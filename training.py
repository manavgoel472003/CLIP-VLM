import argparse
import logging
import os

os.environ["MPLBACKEND"] = "Agg"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import ModelCfg, VLMFusionModel
from data import DataCfg, build_dataset


logger = logging.getLogger(__name__)


def log(msg: str, level: int = logging.INFO):
    logger.log(level, msg)
def _ensure_batch(batch):
    if isinstance(batch, list):
        return batch
    return [batch]


def _pad_embeddings(seq_list, device):
    if not seq_list:
        raise ValueError("Cannot pad empty sequence list.")
    max_len = max(seq.size(0) for seq in seq_list)
    d = seq_list[0].size(1)
    dtype = seq_list[0].dtype
    padded = torch.zeros(len(seq_list), max_len, d, device=device, dtype=dtype)
    mask = torch.zeros(len(seq_list), max_len, device=device, dtype=torch.long)
    for i, seq in enumerate(seq_list):
        L = seq.size(0)
        padded[i, :L, :] = seq
        mask[i, :L] = 1
    return padded, mask


def _pad_labels(label_list, device, pad_value=-100):
    max_len = max(lbl.size(0) for lbl in label_list)
    dtype = label_list[0].dtype
    padded = torch.full((len(label_list), max_len), pad_value, device=device, dtype=dtype)
    for i, lbl in enumerate(label_list):
        L = lbl.size(0)
        padded[i, :L] = lbl
    return padded


def distill_epoch(model: VLMFusionModel, loader, opt, device: str):
    model.train()
    total = 0.0
    steps = 0
    for batch in tqdm(loader, desc="distill"):
        samples = _ensure_batch(batch)
        fused_list = []
        teacher_list = []
        for sample in samples:
            images = sample["images"]
            with torch.no_grad():
                teacher = model.teacher_tokens(images).to(device).squeeze(0)
            fused = model.fused_tokens(images).squeeze(0).to(device)
            L = min(fused.size(0), teacher.size(0))
            if L == 0:
                continue
            fused_list.append(fused[:L])
            teacher_list.append(teacher[:L])
        if not fused_list:
            continue
        opt.zero_grad()
        fused_pad, mask = _pad_embeddings(fused_list, device)
        teacher_pad, _ = _pad_embeddings(teacher_list, device)
        mask = mask.unsqueeze(-1).float()
        diff = fused_pad - teacher_pad
        denom = mask.sum().clamp_min(1.0)
        loss = F.smooth_l1_loss(diff * mask, torch.zeros_like(diff), reduction="sum") / denom
        loss.backward()
        nn.utils.clip_grad_norm_(model.connector.parameters(), 1.0)
        opt.step()
        total += loss.item()
        steps += 1
    return total / max(1, steps)


def lm_epoch(model: VLMFusionModel, loader, opt, device: str, prompt: str):
    model.train()
    total = 0.0
    steps = 0
    embed_layer = model.gen_model.get_input_embeddings()
    target_dtype = model.gen_model.lm_head.weight.dtype
    for batch in tqdm(loader, desc="lm"):
        samples = _ensure_batch(batch)
        embed_list = []
        label_list = []
        for sample in samples:
            text = sample["text"]
            if not text:
                continue
            images = sample["images"]
            fused = model.fused_tokens(images).squeeze(0).to(device, dtype=target_dtype)
            prompt_ids = model.tok(prompt, return_tensors="pt").to(device)["input_ids"].squeeze(0)
            prompt_emb = embed_layer(prompt_ids.unsqueeze(0)).squeeze(0).to(device, dtype=target_dtype)
            label_tokens = model.tok(text, return_tensors="pt").to(device)["input_ids"].squeeze(0)
            label_emb = embed_layer(label_tokens.unsqueeze(0)).squeeze(0).to(device, dtype=target_dtype)
            inputs = torch.cat([fused, prompt_emb, label_emb], dim=0)
            labels = torch.cat([
                torch.full((fused.size(0) + prompt_emb.size(0),), -100, device=device, dtype=label_tokens.dtype),
                label_tokens,
            ], dim=0)
            embed_list.append(inputs)
            label_list.append(labels)
        if not embed_list:
            continue
        opt.zero_grad()
        embeds_padded, attn_mask = _pad_embeddings(embed_list, device)
        labels_padded = _pad_labels(label_list, device, pad_value=-100)
        out = model.gen_model(inputs_embeds=embeds_padded, attention_mask=attn_mask, labels=labels_padded)
        loss = out.loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.connector.parameters(), 1.0)
        opt.step()
        total += loss.item()
        steps += 1
    return total / max(1, steps)

# Yet to add
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--nusc-root", default=None, help="nuScenes root directory (required for nuscenes/nuinteract datasets)")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--mode", choices=["distill", "lm"], default="lm")
    ap.add_argument("--dataset", choices=["nuscenes", "nuinteract", "overlap_export"], default="nuscenes")
    ap.add_argument("--batch-size", type=int, default=1)  # keep 1; variable images per sample
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--save", default="./connector.pt")
    ap.add_argument("--cameras", nargs="*", default=[
        "CAM_FRONT","CAM_FRONT_RIGHT","CAM_BACK_RIGHT","CAM_BACK","CAM_BACK_LEFT","CAM_FRONT_LEFT"])
    ap.add_argument("--captions", default=None)
    ap.add_argument("--prompt", default="", help="Optional text prompt prepended during LM fine-tuning/inference.")
    ap.add_argument("--nuinteract-dir", default=None, help="Directory that contains NuInteract dense caption JSON files.")
    ap.add_argument("--overlap-export-dir", default=None, help="Path to dataset produced by export_overlap_dataset.py")
    ap.add_argument("--nuinteract-caption-strategy", choices=["overall", "per_view_concat"], default="overall",
                    help="How to combine NuInteract camera captions into a single target text.")
    ap.add_argument("--max-samples", type=int, default=None, help="Optional limit on dataset size (useful for smoke tests).")
    ap.add_argument("--qwen-id", default="Qwen/Qwen2.5-VL-3B-Instruct",
                    help="Which Qwen VLM checkpoint to use (default: 3B Instruct).")
    ap.add_argument("--qwen-quant", choices=["none", "bnb-8bit", "bnb-4bit"], default="none",
                    help="Load Qwen in 8-bit or 4-bit (bitsandbytes) to reduce VRAM usage.")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-vis-tokens", type=int, default=None,
                    help="Optional cap on fused visual token length to reduce memory usage.")
    ap.add_argument(
        "--require-files",
        action="store_true",
        help="Skip samples where any requested camera file is missing (useful with partial nuScenes blobs).",
    )
    ap.add_argument(
        "--checkpoint-dir",
        default=None,
        help="If set, save connector checkpoints after every epoch to this directory (e.g., a Drive folder).",
    )
    args = ap.parse_args()

    if args.dataset in {"nuscenes", "nuinteract"} and not args.nusc_root:
        ap.error("--nusc-root is required for nuscenes/nuinteract datasets")
    if args.dataset == "nuinteract" and not args.nuinteract_dir:
        ap.error("--nuinteract-dir is required when --dataset nuinteract")
    if args.dataset == "overlap_export" and not args.overlap_export_dir:
        ap.error("--overlap-export-dir is required when --dataset overlap_export")
    setattr(args, "use_lora", True)
    log("LoRA fine-tuning enabled for Qwen (mandatory).")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device} mode={args.mode} dataset={args.dataset} quant={args.qwen_quant}")

    mcfg = ModelCfg(
        device=device,
        qwen_quant=args.qwen_quant,
        use_lora=args.use_lora, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        max_vis_tokens=args.max_vis_tokens,
        qwen_id=args.qwen_id,
    )
    model = VLMFusionModel(mcfg).to(device)
    log("Model instantiated (vision + language backbones ready)")


    # Freeze pre-trained backbones + LLM
    for p in model.qwen.parameters(): p.requires_grad = False
    for p in model.ext.parameters():  p.requires_grad = False

    if args.use_lora:
        # freeze base LM weights but leave LoRA trainable
        for n, p in model.gen_model.named_parameters():
            if "lora_" in n: 
                p.requires_grad = True
            else:
                p.requires_grad = False
    else:
        for p in model.gen_model.parameters(): p.requires_grad = False

    # Optimizers
    train_params = list(model.connector.parameters())
    if args.use_lora:
        train_params += [p for n,p in model.gen_model.named_parameters() if p.requires_grad]

    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=args.wd)

    dcfg = DataCfg(
        nusc_root=args.nusc_root,
        version=args.version,
        cameras=args.cameras,
        captions_json=args.captions,
        dataset=args.dataset,
        nuinteract_dir=args.nuinteract_dir,
        nuinteract_caption_strategy=args.nuinteract_caption_strategy,
        max_samples=args.max_samples,
        require_files=args.require_files,
        overlap_dir=args.overlap_export_dir,
    )
    log("Building dataset ...")
    ds = build_dataset(dcfg)
    log(f"Dataset built with {len(ds)} samples")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=lambda x: x)
    log(f"DataLoader ready (batch_size={args.batch_size})")

    def build_payload():
        return {
            "connector": model.connector.state_dict(),
            "d_ext": model.ext.d_out,
            "d_model": model.qwen.d_model,
            "version": args.version,
            "qwen_id": mcfg.qwen_id,
            "mode": args.mode,
            "prompt": args.prompt,
        }

    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        log(f"Starting epoch {ep}/{args.epochs}")
        if args.mode == "distill":
            loss = distill_epoch(model, loader, opt, device)
        else:
            loss = lm_epoch(model, loader, opt, device, prompt=args.prompt)
        log(f"Epoch {ep} loss: {loss:.4f}")
        if args.checkpoint_dir:
            ckpt_path = os.path.join(args.checkpoint_dir, f"connector_epoch{ep}.pt")
            torch.save(build_payload(), ckpt_path)
            log(f"Saved epoch {ep} checkpoint to {ckpt_path}")

    # Save
    torch.save(build_payload(), args.save)
    log(f"Saved checkpoint to {args.save}")


if __name__ == "__main__":
    main()
