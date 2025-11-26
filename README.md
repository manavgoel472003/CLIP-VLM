# CLIP-VLM

Fine-tune a CLIP + Qwen VLM on nuScenes surround cameras. Bring your own captions or use NuInteract’s dense ones. If you only grabbed part of nuScenes, flip a flag and we’ll skip missing images.

## What’s going on
- We run all chosen camera views through CLIP, fuse them, and project to Qwen’s hidden size.
- Modes:  
  - `distill`: match fused visuals to Qwen’s frozen vision tower (no text needed).  
  - `lm`: teach the LLM to write captions for each sample.
- We freeze CLIP and base Qwen; only the connector (and LoRA layers if you enable them) trains.
- `--prompt` gets prepended to every target caption in LM mode so you can steer tone.

## Data setup (keep it light)
- nuScenes folder needs `samples/`, `sweeps/`, `maps/`, and a `v1.0-*` metadata folder.
- Missing blobs? Add `--require-files` to drop any sample missing a requested camera image.
- Pick cameras with `--cameras` (defaults to all 6 surround).
- Your captions (nuScenes): JSON mapping `sample_token -> text` via `--captions`. Samples without captions are skipped in LM mode.

## NuInteract (DriveMonkey) captions
- Point `--nuinteract-dir` at `all_caption_public/`.
- GPT-only: we read `gpt_caption` fields and ignore others. Rows without GPT captions are skipped.
- Strategies (`--nuinteract-caption-strategy`):  
  - `overall` (default): use `OVERALL` plus FRONT/BACK snippets.  
  - `per_view_concat`: glue per-camera captions in the order of `--cameras`; if none, fall back to `OVERALL`.
- With `--require-files`, any sample missing a requested JPG is dropped.

### Export just the overlap (front/back + GPT/Gemini captions)
Need the NuInteract/nuScenes intersection as a self-contained folder? Use `export_overlap_dataset.py` to copy only the overlapping tokens, front/back camera JPGs, and GPT captions:
```bash
uv run python export_overlap_dataset.py \
  --nusc-root /path/to/nuscenes \
  --version v1.0-mini \
  --nuinteract-dir /path/to/all_caption_public \
  --output-dir /tmp/nu_overlap_front_back \
  --require-files \
  --overwrite
```
- Defaults to copying `CAM_FRONT` and `CAM_BACK`; pass `--cameras` to change the views.
- Output layout: `images/<token>/CAM_FRONT.jpg`, `images/<token>/CAM_BACK.jpg`, plus `captions.json` and `manifest.json` (combined/front/back GPT text only).
- Use `--max-samples` for a quick sanity check before exporting the full set.

## Train from exported overlap (no nuScenes tree needed)
After exporting the overlap folder (contains `images/`, `manifest.json`, `captions.json`), you can train directly on it:
```bash
uv run python training.py \
  --dataset overlap_export \
  --overlap-export-dir /path/to/overlap_front_back_trainval \
  --mode lm \
  --epochs 3 \
  --prompt "Describe the scene." \
  --max-samples 100  # optional smoke test
```
- Leave `--nusc-root` unset for this mode; manifests already include image paths.
- `--cameras` defaults to the manifest order, but you can override if you only want a subset.

Want to run everything on Colab? See `COLAB_TRAINING.md` for a cell-by-cell walkthrough (GPU runtime setup, installing from `pyproject.toml`, copying the exported dataset via Drive, and saving checkpoints back to Drive).

## Training flow (plain version)
1) Configure: dataset (`nuscenes|nuinteract`), version (`v1.0-mini|v1.0-trainval`), cameras, prompt, and `--require-files` if you’re on partial blobs.  
2) Build dataset (`data.py`): resolve image paths, check files if asked, build one caption string per sample using the chosen strategy.  
3) Init model (`model.py`): CLIP + Qwen; optional quant (`--qwen-quant bnb-8bit|bnb-4bit`) and LoRA (`--use-lora`).  
4) Train (`training.py`):  
   - `distill`: Smooth L1 + cosine between fused tokens and teacher vision tokens.  
   - `lm`: cross-entropy on the caption (with prompt prepended).  
   - Batch size 1 (variable number of images).  
5) Save: connector weights + metadata go to `--save`.

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

NuInteract GPT captions (partial trainval is fine):
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

## Handy flags
- `--require-files`: drop samples missing any requested camera image (use this with partial blobs).
- `--max-samples N`: quick smoke test.
- `--qwen-quant bnb-8bit|bnb-4bit`: trims VRAM; pairs well with `--use-lora`.
