"""
Colab-friendly inference helper.

This script mirrors the training setup used in CLIP_VLM.ipynb: it loads the
frozen CLIP + Qwen backbones, optionally applies a fine-tuned connector
checkpoint, and then runs caption generation on either explicit image paths
or tokens from an exported overlap dataset.

Usage inside Colab:
  !python colab_inference.py \
       --checkpoint /content/drive/MyDrive/CLIP-VLM/checkpoints/connector_final.pt \
       --overlap-export-dir /content/CLIP-VLM/overlap_front_back_trainval \
       --token 7dfafb95b1cb4b6592411a1afedb4893 \
       --cameras CAM_FRONT CAM_BACK \
       --prompt "Describe the scene." \
       --display
"""

import argparse
import json
import logging
import os
from typing import List, Tuple

import torch
from PIL import Image

from model import ModelCfg, VLMFusionModel

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - matplotlib is optional
    plt = None


logger = logging.getLogger("colab_inference")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def load_images(paths: List[str]) -> List[Image.Image]:
    images = []
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        images.append(Image.open(path).convert("RGB"))
    return images


def resolve_token_images(overlap_dir: str, token: str, cameras: List[str]) -> Tuple[List[str], dict]:
    manifest_path = os.path.join(overlap_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest.json not found under {overlap_dir}")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    for sample in manifest.get("samples", []):
        if sample.get("token") != token:
            continue
        images = sample.get("images", {})
        img_paths = []
        for cam in cameras:
            rel = images.get(cam)
            if not rel:
                raise FileNotFoundError(f"Camera {cam} missing for token {token}")
            img_paths.append(os.path.join(overlap_dir, rel))
        return img_paths, sample
    raise ValueError(f"Token {token} not found in manifest.")


def maybe_show(images: List[Image.Image], cameras: List[str], enabled: bool):
    if not enabled:
        return
    if plt is None:
        logger.warning("matplotlib not available; skipping visualization.")
        return
    cols = len(images)
    plt.figure(figsize=(4 * cols, 4))
    for idx, (im, cam) in enumerate(zip(images, cameras), start=1):
        plt.subplot(1, cols, idx)
        plt.imshow(im)
        plt.axis("off")
        plt.title(cam)
    plt.tight_layout()
    plt.show()


def build_model(args, ckpt):
    qwen_id = args.qwen_id or (ckpt.get("qwen_id") if ckpt else None) or "Qwen/Qwen2.5-VL-3B-Instruct"
    use_lora = bool(ckpt)
    cfg = ModelCfg(
        device=args.device,
        qwen_id=qwen_id,
        qwen_quant=args.qwen_quant,
        max_vis_tokens=args.max_vis_tokens,
        use_lora=use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model = VLMFusionModel(cfg).to(args.device)
    if ckpt:
        connector_state = ckpt.get("connector")
        if connector_state is None:
            raise KeyError("Checkpoint missing 'connector' weights.")
        model.connector.load_state_dict(connector_state)
        logger.info("Loaded connector weights from %s", args.checkpoint)
    else:
        logger.info("Running with frozen pretrained CLIP + Qwen (no connector).")
    model.eval()
    return model


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
    out_ids = model.gen_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return model.tok.batch_decode(out_ids, skip_special_tokens=True)[0]


def parse_args():
    ap = argparse.ArgumentParser(description="Colab helper for CLIP-VLM inference.")
    ap.add_argument("--checkpoint", help="Path to fine-tuned connector checkpoint (.pt). Optional.")
    ap.add_argument("--image-paths", nargs="+", help="Image paths to run inference on (bypass manifest).")
    ap.add_argument("--overlap-export-dir", default="/content/CLIP-VLM/overlap_front_back_trainval",
                    help="Exported overlap dataset directory (for token lookup).")
    ap.add_argument("--token", help="Token inside manifest.json to visualize.")
    ap.add_argument("--cameras", nargs="+", default=["CAM_FRONT", "CAM_BACK"],
                    help="Camera ordering to load when resolving a token.")
    ap.add_argument("--prompt", default="Describe the scene.", help="Prompt prepended before generation.")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--qwen-id", default=None, help="Override Qwen checkpoint (defaults to checkpoint metadata).")
    ap.add_argument("--qwen-quant", choices=["none", "bnb-8bit", "bnb-4bit"], default="none")
    ap.add_argument("--max-vis-tokens", type=int, default=None, help="Optional cap on visual tokens.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--display", action="store_true", help="Show camera views inline (matplotlib).")
    ap.add_argument("--save-caption", default=None, help="Optional path to save generated caption text.")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    args = ap.parse_args()

    if not args.image_paths and not (args.overlap_export_dir and args.token):
        ap.error("Provide --image-paths or both --overlap-export-dir and --token.")
    return args


def main():
    args = parse_args()
    logger.info("Starting inference with device=%s quant=%s checkpoint=%s", args.device, args.qwen_quant, args.checkpoint)
    ckpt = None
    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(args.checkpoint)
        logger.info("Loading checkpoint from %s", args.checkpoint)
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    else:
        logger.info("No checkpoint provided; using base pretrained model only.")

    logger.info("Building model (qwen_id=%s, max_vis_tokens=%s)", args.qwen_id or (ckpt.get("qwen_id") if ckpt else None), args.max_vis_tokens)
    model = build_model(args, ckpt)

    if args.image_paths:
        logger.info("Using explicit image paths (count=%d)", len(args.image_paths))
        image_paths = args.image_paths
        sample_info = {}
    else:
        logger.info("Resolving token %s from %s", args.token, args.overlap_export_dir)
        image_paths, sample_info = resolve_token_images(args.overlap_export_dir, args.token, args.cameras)
    logger.info("Using images: %s", ", ".join(image_paths))
    images = load_images(image_paths)
    maybe_show(images, args.cameras, args.display)

    logger.info("Generating caption with prompt='%s' max_new_tokens=%d", args.prompt, args.max_new_tokens)
    with torch.no_grad():
        caption = generate_caption(model, images, prompt=args.prompt, max_new_tokens=args.max_new_tokens)
    logger.info("Caption generation finished.")

    print("\n=== Generated Caption ===\n")
    print(caption)

    if args.display and plt is not None:
        cols = len(images)
        plt.figure(figsize=(4 * cols, 6))
        for idx, (im, cam) in enumerate(zip(images, args.cameras), start=1):
            plt.subplot(2, cols, idx)
            plt.imshow(im)
            plt.axis("off")
            plt.title(cam)
        for idx in range(cols):
            plt.subplot(2, cols, cols + idx + 1)
            plt.axis("off")
            sub = caption if idx == 0 else ""
            plt.text(0.5, 0.5, sub, fontsize=12, ha="center", va="center", wrap=True)
        plt.tight_layout()
        plt.show()

    if args.save_caption:
        save_dir = os.path.dirname(args.save_caption)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(args.save_caption, "w") as f:
            f.write(caption + "\n")
        logger.info("Saved caption to %s", args.save_caption)

    if sample_info:
        gt = sample_info.get("caption")
        if gt:
            print("\n=== Reference Caption (dataset) ===\n")
            print(gt)


if __name__ == "__main__":
    main()
