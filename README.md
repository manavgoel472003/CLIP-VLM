# CLIP-VLM

Fine-tune a CLIP + Qwen VLM on nuScenes multi-camera images. Supports your own captions or the NuInteract dense-caption release. Works even if you only unpack part of nuScenes (skip missing images with a flag).

## What’s happening
- Images: all chosen camera views per sample are fed through CLIP, fused, and projected to Qwen’s hidden size.
- Modes:  
  - `distill` — match fused visual tokens to Qwen’s frozen vision tower (no text needed).  
  - `lm` — train the LLM to emit captions for each sample.
- Freezing: CLIP and base Qwen weights stay frozen; only the connector (and LoRA layers if enabled) train.
- Prompts: `--prompt` text is prepended to every target caption in LM mode to steer style.

## Data setup
- nuScenes root must include `samples/`, `sweeps/`, `maps/`, and a `v1.0-*` metadata folder.
- Partial blobs: add `--require-files` to drop any sample missing a requested camera image (helps when you only have part of trainval).
- Cameras: choose with `--cameras` (defaults to all 6 surround).
- Custom captions (nuScenes): JSON mapping `sample_token -> text` via `--captions`. Samples without captions are skipped in LM mode.

## NuInteract (DriveMonkey) captions
- Point `--nuinteract-dir` at `all_caption_public/`.
- GPT-only: the loader uses `gpt_caption` fields only; rows without GPT captions are skipped.
- Caption strategies (`--nuinteract-caption-strategy`):  
  - `overall` (default): use `OVERALL` plus FRONT/BACK snippets.  
  - `per_view_concat`: concatenate per-camera captions in the order of `--cameras`; if none, fall back to `OVERALL`.
- Missing images: dropped when `--require-files` is set.

## Training flow
1) Configure: `--dataset [nuscenes|nuinteract]`, `--version v1.0-mini|v1.0-trainval`, cameras, prompt, `--require-files` if using partial blobs.  
2) Build dataset (`data.py`): resolves image paths, checks files if requested, builds one caption string per sample according to the chosen strategy.  
3) Init model (`model.py`): CLIP + Qwen; optional quant (`--qwen-quant bnb-8bit|bnb-4bit`) and LoRA (`--use-lora`).  
4) Train (`training.py`):  
   - `distill`: Smooth L1 + cosine between fused tokens and teacher vision tokens.  
   - `lm`: cross-entropy on the caption (with prompt prepended).  
   - Loader uses batch size 1 (variable number of images).  
5) Save: connector weights and metadata to `--save`.

## Quick commands
nuScenes with your captions:
```bash
uv run python training.py \
  --dataset nuscenes \
  --nusc-root /path/to/nuscenes \
  --version v1.0-mini \
  --captions /path/to/captions.json \
  --mode lm \
  --prompt "You are the driver. Reply with the next safe instruction." \
  --epochs 3
```

NuInteract GPT captions (partially unpacked trainval):
```bash
uv run python training.py \
  --dataset nuinteract \
  --nusc-root /path/to/nuscenes \
  --version v1.0-trainval \
  --nuinteract-dir /path/to/all_caption_public \
  --nuinteract-caption-strategy per_view_concat \
  --cameras CAM_FRONT CAM_FRONT_RIGHT CAM_BACK_RIGHT CAM_BACK CAM_BACK_LEFT CAM_FRONT_LEFT \
  --require-files \
  --mode lm \
  --epochs 3
```

## Flag tips
- `--require-files`: drop samples missing any requested camera image (use for partial blobs).
- `--max-samples N`: small smoke tests.
- `--qwen-quant bnb-8bit|bnb-4bit`: smaller VRAM; pairs well with `--use-lora`.

