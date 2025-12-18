from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
try:
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover - optional dep pre-4.30
    BitsAndBytesConfig = None
import open_clip

from peft import LoraConfig, get_peft_model, TaskType

@dataclass
class ModelCfg:
    qwen_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    device: str = "cuda"
    qdtype: torch.dtype = torch.bfloat16
    clip_arch: str = "ViT-L-14"
    clip_ckpt: str = "openai"
    qwen_quant: str = "bnb-4bit"  # choices: none, bnb-8bit, bnb-4bit

    n_heads: int = 8
    n_layers: int = 1
    gate_init: float = 0.5

    ego_prompt: str = """Describe the scene in detail."
    """

    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_targets: tuple = ("q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj")
    max_vis_tokens: Optional[int] = None

class CLIPBackbone(nn.Module):
    def __init__(self, arch: str, ckpt: str, device: str):
        super().__init__()
        self.model, _, self.preproc = open_clip.create_model_and_transforms(arch, pretrained=ckpt, device=device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=device)
            f = self.model.encode_image(dummy)
            self.d_out = f.shape[-1]

    @torch.no_grad()
    def encode_many(self, images: List[Image.Image], device: str) -> torch.Tensor:
        feats = []
        for im in images:
            t = self.preproc(im.convert("RGB")).unsqueeze(0).to(device)
            f = self.model.encode_image(t)  # (1, d)
            feats.append(f)
        return torch.cat(feats, dim=0)  # (N, d)
    
class VisualTokenCatcher:
    def __init__(self, d_model: int):
        self.d_model = d_model
        self.cands = []
        self.handles = []

    def hook(self, name):
        def fn(m, i, o):
            if torch.is_tensor(o) and o.ndim == 3 and o.shape[-1] == self.d_model:
                self.cands.append((o.detach().to("cpu"), name))
        return fn

    def attach(self, model: nn.Module):
        for n, m in model.named_modules():
            self.handles.append(m.register_forward_hook(self.hook(n)))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def pick(self) -> torch.Tensor:
        if not self.cands:
            raise RuntimeError("No (B,T,d) activations captured. Visual projector not found.")
        best = max(self.cands, key=lambda x: x[0].shape[1])
        return best[0]  # (B, T, d)


class QwenVisionExtractor(nn.Module):
    def __init__(self, qwen_id: str, device: str, dtype: torch.dtype, quant_mode: str = "none"):
        super().__init__()
        self.device = device
        load_kwargs = {"device_map": "auto"}
        if quant_mode == "none":
            load_kwargs["torch_dtype"] = dtype
        else:
            load_kwargs["quantization_config"] = self._build_bnb_config(dtype, quant_mode)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            qwen_id, **load_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(qwen_id, use_fast=False)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.d_model = self.model.config.hidden_size
        self.quant_mode = quant_mode

    def _build_bnb_config(self, dtype: torch.dtype, quant_mode: str):
        if BitsAndBytesConfig is None:
            raise ImportError(
                "bitsandbytes is required for quantized Qwen loading; please install it or set --qwen-quant none."
            )
        qmode = quant_mode.lower()
        if qmode == "bnb-4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
            )
        if qmode == "bnb-8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        raise ValueError(f"Unsupported quantization mode: {quant_mode}")

    @torch.no_grad()
    def visual_tokens(self, images: List[Image.Image]) -> torch.Tensor:
        catcher = VisualTokenCatcher(self.d_model)
        catcher.attach(self.model)
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": "."})
        messages = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(self.device)
        _ = self.model(**inputs)
        catcher.detach()
        vt = catcher.pick()  # (1, T, d)
        return vt
    
class FusionConnector(nn.Module):
    """Project external tokens -> d_model, then cross-attend from Qwen tokens to external tokens.
       Output has shape (B, Tq, d_model). Only this module is trained for fusion.
    """
    def __init__(self, d_ext: int, d_model: int, n_heads: int = 8, n_layers: int = 1, gate_init: float = 0.5):
        super().__init__()
        self.in_proj = nn.Linear(d_ext, d_model)
        self.in_norm = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True, activation="gelu")
            for _ in range(n_layers)
        ])
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.out_norm = nn.LayerNorm(d_model)
        self.gate = nn.Parameter(torch.full((1, 1, d_model), gate_init))

    def forward(self, qwen_tokens: torch.Tensor, ext_tokens: torch.Tensor) -> torch.Tensor:
        x = self.in_norm(self.in_proj(ext_tokens))  # (B, Te, d)
        for blk in self.blocks:
            x = blk(x)
        Q = self.q_proj(qwen_tokens)
        K = self.k_proj(x)
        V = self.v_proj(x)
        attn_out, _ = self.attn(Q, K, V)
        fused = qwen_tokens + torch.sigmoid(self.gate) * attn_out
        return self.out_norm(fused)


class VLMFusionModel(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.qwen = QwenVisionExtractor(cfg.qwen_id, cfg.device, cfg.qdtype, cfg.qwen_quant)
        self.tok = AutoTokenizer.from_pretrained(cfg.qwen_id)
        self.ext = CLIPBackbone(cfg.clip_arch, cfg.clip_ckpt, cfg.device)
        self.connector = FusionConnector(d_ext=self.ext.d_out, d_model=self.qwen.d_model,
                                         n_heads=cfg.n_heads, n_layers=cfg.n_layers, gate_init=cfg.gate_init)
        self.gen_model = self.qwen.model

        if self.cfg.use_lora:
            peft_cfg = LoraConfig(
                r=self.cfg.lora_r,
                lora_alpha=self.cfg.lora_alpha,
                lora_dropout=self.cfg.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=list(self.cfg.lora_targets),
            )
            self.gen_model = get_peft_model(self.gen_model, peft_cfg)

    @torch.no_grad()
    def teacher_tokens(self, images: List[Image.Image]) -> torch.Tensor:
        return self.qwen.visual_tokens(images)  # (1, T, d)

    @torch.no_grad()
    def ext_tokens(self, images: List[Image.Image]) -> torch.Tensor:
        feats = self.ext.encode_many(images, device=self.cfg.device)  # (N, d_ext)
        return feats.unsqueeze(0)  # (1, Te, d_ext)

    def fused_tokens(self, images: List[Image.Image]) -> torch.Tensor:
        with torch.no_grad():
            target_dtype = self.connector.in_proj.weight.dtype
            q = self.teacher_tokens(images).to(self.cfg.device, dtype=target_dtype)  # (1, Tq, d)
            e = self.ext_tokens(images).to(self.cfg.device, dtype=target_dtype)      # (1, Te, d_ext)
            max_vis = self.cfg.max_vis_tokens
            if max_vis is not None and max_vis > 0 and q.size(1) > max_vis:
                q = q[:, :max_vis, :]
        return self.connector(q, e)  # (1, Tq, d)

    def llm_vis_hidden(self, images: List[Image.Image], prompt: Optional[str] = None):
        """Run the LLM on [fused visual prefix ; prompt] and return final hidden states + visual length."""
        fused = self.fused_tokens(images)  # (1, T_vis, d)
        T_vis = fused.size(1)
        txt = (self.cfg.ego_prompt if prompt is None else prompt)
        enc = self.tok(txt, return_tensors="pt").to(self.cfg.device)
        if enc["input_ids"].numel() > 0:
            text_emb = self.gen_model.get_input_embeddings()(enc["input_ids"])  # (1, T_txt, d)
            inputs_embeds = torch.cat([fused, text_emb], dim=1)
        else:
            inputs_embeds = fused
        attn_mask = torch.ones(inputs_embeds.shape[:-1], dtype=torch.long, device=self.cfg.device)
        out = self.gen_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        # For causal LMs in HF, hidden_states is a tuple of layer outputs; take the last
        hidden = out.hidden_states[-1] if isinstance(out.hidden_states, (list, tuple)) else out.last_hidden_state
        return hidden, T_vis
    
    def lm_step(self, images: List[Image.Image], prompt: str, labels_text: Optional[str] = None):
        fused = self.fused_tokens(images)  # (1, T, d)
        enc = self.tok(prompt, return_tensors="pt").to(self.cfg.device)
        text_emb = self.gen_model.get_input_embeddings()(enc["input_ids"])  # (1, Tt, d)
        target_dtype = self.gen_model.lm_head.weight.dtype
        fused = fused.to(self.cfg.device, dtype=target_dtype)
        text_emb = text_emb.to(self.cfg.device, dtype=target_dtype)
        pieces = [fused, text_emb]
        label_ids = None
        if labels_text is not None:
            label_tok = self.tok(labels_text, return_tensors="pt").to(self.cfg.device)
            label_ids = label_tok["input_ids"]
            label_emb = self.gen_model.get_input_embeddings()(label_ids).to(self.cfg.device, dtype=target_dtype)
            pieces.append(label_emb)
        inputs_embeds = torch.cat(pieces, dim=1)
        attn_mask = torch.ones(inputs_embeds.shape[:-1], dtype=torch.long, device=self.cfg.device)
        if labels_text is None:
            return self.gen_model(inputs_embeds=inputs_embeds, attention_mask=attn_mask)
        else:
            pad_len = fused.size(1) + text_emb.size(1)
            pad = torch.full((label_ids.size(0), pad_len), -100, device=label_ids.device)
            labels = torch.cat([pad, label_ids], dim=1)
            return self.gen_model(inputs_embeds=inputs_embeds, attention_mask=attn_mask, labels=labels)

    def generate(self, images: List[Image.Image], prompt: str, max_new_tokens: int = 128) -> str:
        fused = self.fused_tokens(images)
        enc = self.tok(prompt, return_tensors="pt").to(self.cfg.device)
        text_emb = self.gen_model.get_input_embeddings()(enc["input_ids"])  # (1, Tt, d)
        inputs_embeds = torch.cat([fused, text_emb], dim=1)
        attn_mask = torch.ones(inputs_embeds.shape[:-1], dtype=torch.long, device=self.cfg.device)
        out_ids = self.gen_model.generate(inputs_embeds=inputs_embeds, attention_mask=attn_mask,
                                          max_new_tokens=max_new_tokens, do_sample=False)
        return self.tok.batch_decode(out_ids, skip_special_tokens=True)[0]
    
