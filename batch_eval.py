import argparse
import json
import logging
import os
import shutil
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

from model import ModelCfg, VLMFusionModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("batch_eval")


def load_manifest(overlap_dir: str) -> Dict:
    manifest_path = os.path.join(overlap_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest.json not found under {overlap_dir}")
    with open(manifest_path, "r") as f:
        return json.load(f)


def pick_tokens(manifest: Dict, limit: int) -> List[str]:
    tokens = [sample.get("token") for sample in manifest.get("samples", []) if sample.get("token")]
    if not tokens:
        raise ValueError("No tokens found in manifest")
    return tokens[:limit]


def resolve_token_images(overlap_dir: str, token: str, cameras: List[str], manifest_map: Dict[str, Dict]) -> Tuple[List[str], Dict]:
    sample = manifest_map.get(token)
    if not sample:
        raise ValueError(f"Token {token} not found in manifest")
    images = sample.get("images", {})
    img_paths = []
    for cam in cameras:
        rel = images.get(cam)
        if not rel:
            raise FileNotFoundError(f"Camera {cam} missing for token {token}")
        img_paths.append(os.path.join(overlap_dir, rel))
    return img_paths, sample


def load_images(paths: List[str]) -> List[Image.Image]:
    out = []
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        out.append(Image.open(path).convert("RGB"))
    return out


def build_inputs_embeds(model: VLMFusionModel, images: List[Image.Image], prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
    fused = model.fused_tokens(images)
    embed_layer = model.gen_model.get_input_embeddings()
    embed_dtype = embed_layer.weight.dtype
    fused = fused.to(embed_dtype)
    enc = model.tok(prompt, return_tensors="pt").to(model.cfg.device)
    text_emb = embed_layer(enc["input_ids"]).to(embed_dtype)
    inputs_embeds = torch.cat([fused, text_emb], dim=1)
    attn_mask = torch.ones(inputs_embeds.shape[:-1], dtype=torch.long, device=model.cfg.device)
    return inputs_embeds, attn_mask


def generate_caption(model: VLMFusionModel, images: List[Image.Image], prompt: str, max_new_tokens: int) -> str:
    inputs_embeds, attn_mask = build_inputs_embeds(model, images, prompt)
    out_ids = model.gen_model.generate(inputs_embeds=inputs_embeds, attention_mask=attn_mask, max_new_tokens=max_new_tokens, do_sample=False)
    return model.tok.batch_decode(out_ids, skip_special_tokens=True)[0]


def init_model(device: str, qwen_id: str, qwen_quant: str, max_vis_tokens: Optional[int], checkpoint: Optional[Dict], use_lora: bool,
               lora_r: int, lora_alpha: int, lora_dropout: float) -> VLMFusionModel:
    cfg = ModelCfg(
        device=device,
        qwen_id=qwen_id,
        qwen_quant=qwen_quant,
        max_vis_tokens=max_vis_tokens,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    model = VLMFusionModel(cfg).to(device)
    if checkpoint:
        connector_state = checkpoint.get("connector")
        if connector_state is None:
            raise KeyError("Checkpoint missing 'connector' weights")
        model.connector.load_state_dict(connector_state)
    model.eval()
    return model


def copy_images(paths: List[str], dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)
    for path in paths:
        shutil.copy2(path, os.path.join(dest_dir, os.path.basename(path)))


def main():
    ap = argparse.ArgumentParser(description="Batch caption evaluation over multiple checkpoints.")
    ap.add_argument("--overlap-export-dir", required=True)
    ap.add_argument("--output-dir", default="preds", help="Base directory to store predictions/results.")
    ap.add_argument("--checkpoints-dir", default="checkpoints", help="Directory containing connector_*.pt files.")
    ap.add_argument("--prompt", default="Describe the driving scene by comparing the front and rear camera views, noting road context, traffic flow, and any differences.")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-vis-tokens", type=int, default=2048)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--qwen-quant", choices=["none", "bnb-8bit", "bnb-4bit"], default="none")
    ap.add_argument("--qwen-id", default=None)
    ap.add_argument("--cameras", nargs="+", default=["CAM_FRONT", "CAM_BACK"])
    ap.add_argument("--num-tokens", type=int, default=5)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    args = ap.parse_args()

    manifest = load_manifest(args.overlap_export_dir)
    tokens = pick_tokens(manifest, args.num_tokens)
    logger.info("Evaluating %d tokens: %s", len(tokens), ", ".join(tokens))

    checkpoints = sorted([os.path.join(args.checkpoints_dir, f) for f in os.listdir(args.checkpoints_dir) if f.endswith(".pt")])
    checkpoints.insert(0, None)

    results = []
    os.makedirs(args.output_dir, exist_ok=True)

    manifest_map = {sample["token"]: sample for sample in manifest.get("samples", []) if sample.get("token")}

    for ckpt_path in checkpoints:
        label = "no_checkpoint" if ckpt_path is None else os.path.splitext(os.path.basename(ckpt_path))[0]
        logger.info("=== Running %s ===", label)
        ckpt_data = None
        qwen_id = args.qwen_id
        use_lora = False
        if ckpt_path:
            logger.info("Loading checkpoint %s", ckpt_path)
            ckpt_data = torch.load(ckpt_path, map_location="cpu")
            qwen_id = qwen_id or ckpt_data.get("qwen_id") or "Qwen/Qwen2.5-VL-3B-Instruct"
            use_lora = True
        else:
            qwen_id = qwen_id or "Qwen/Qwen2.5-VL-3B-Instruct"

        model = init_model(
            device=args.device,
            qwen_id=qwen_id,
            qwen_quant=args.qwen_quant,
            max_vis_tokens=args.max_vis_tokens,
            checkpoint=ckpt_data,
            use_lora=use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )

        ckpt_output_dir = os.path.join(args.output_dir, label)

        for token in tokens:
            logger.info("Token %s", token)
            img_paths, _ = resolve_token_images(args.overlap_export_dir, token, args.cameras, manifest_map)
            images = load_images(img_paths)
            ref_caption = manifest_map[token].get("caption")
            with torch.no_grad():
                pred = generate_caption(model, images, prompt=args.prompt, max_new_tokens=args.max_new_tokens)
            score = None
            if ref_caption:
                score = SequenceMatcher(None, pred.lower(), ref_caption.lower()).ratio()
            token_dir = os.path.join(ckpt_output_dir, token)
            os.makedirs(token_dir, exist_ok=True)
            copy_images(img_paths, token_dir)
            with open(os.path.join(token_dir, "caption.txt"), "w") as f:
                f.write(pred + "\n")
            meta = {
                "token": token,
                "checkpoint": label,
                "prediction": pred,
                "reference": ref_caption,
                "score": score,
                "image_files": [os.path.basename(p) for p in img_paths],
            }
            with open(os.path.join(token_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            results.append(meta)

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved aggregated results to %s", results_path)


if __name__ == "__main__":
    main()