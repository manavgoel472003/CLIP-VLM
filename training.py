import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import ModelCfg, VLMFusionModel
from data import DataCfg, NuScenesMultiCamDataset, EgoNuScenesSeq

def distill_epoch(model: VLMFusionModel, loader, opt, device: str):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="distill"):
        images = batch["images"]
        with torch.no_grad():
            teacher = model.teacher_tokens(images).to(device)
        fused = model.fused_tokens(images)
        loss = F.smooth_l1_loss(fused, teacher) + 0.1 * (1 - F.cosine_similarity(fused.flatten(1), teacher.flatten(1)).mean())
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.connector.parameters(), 1.0)
        opt.step()
        total += loss.item()
    return total / max(1, len(loader))

def lm_epoch(model: VLMFusionModel, loader, opt, device: str):
    model.train()
    total=0.0
    for batch in tqdm(loader, desc="lm"):
        images, text = batch["images"], batch["text"]
        if text is None:
            continue
        out = model.lm_step(images, prompt="", labels_text=text)
        loss = out.loss
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.connector.parameters(), 1.0)
        opt.step()
    return total / max(1, len(loader))

def yaw_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = pred - target
    return torch.atan2(torch.sin(diff), torch.cos(diff)).abs().mean()

def ego_epoch(model: VLMFusionModel, loader, opt, device: str):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="ego"):
        images_seq = batch["images_seq"]   # list of length L, each is list[PIL]
        target = torch.tensor(batch["target"], dtype=torch.float32, device=device).unsqueeze(0)  # (1,3)
        pred = model.ego_predict(images_seq) 
        loss = F.smooth_l1_loss(pred[:, :2], target[:, :2]) + 0.5 * yaw_loss(pred[:, 2], target[:, 2])
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(model.connector.parameters()) + list(model.ego_head.parameters()), 1.0)
        opt.step()
        total += loss.item()
    return total / max(1, len(loader))

# Yet to add
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nusc-root", required=True)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--mode", choices=["distill", "lm", "ego"], default="ego")
    ap.add_argument("--batch-size", type=int, default=1)  # keep 1; variable images per sample
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--save", default="./connector.pt")
    ap.add_argument("--cameras", nargs="*", default=[
        "CAM_FRONT","CAM_FRONT_RIGHT","CAM_BACK_RIGHT","CAM_BACK","CAM_BACK_LEFT","CAM_FRONT_LEFT"])
    ap.add_argument("--captions", default=None)
    ap.add_argument("--seq-len", type=int, default=3)
    ap.add_argument("--use-lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    args = ap.parse_args()
    

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mcfg = ModelCfg(
        device=device,
        use_lora=args.use_lora, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout
    )
    model = VLMFusionModel(mcfg).to(device)


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
    if args.mode == "ego":
        train_params += list(model.ego_head.parameters())
    if args.use_lora:
        train_params += [p for n,p in model.gen_model.named_parameters() if p.requires_grad]

    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=args.wd)

    if args.mode == "distill" or args.mode == "lm":
        dcfg = DataCfg(nusc_root=args.nusc_root, version=args.version, cameras=args.cameras, captions_json=args.captions)
        ds = NuScenesMultiCamDataset(dcfg)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=lambda x: x[0])
        for ep in range(1, args.epochs + 1):
            if args.mode == "distill":
                loss = distill_epoch(model, loader, opt, device)
            else:
                loss = lm_epoch(model, loader, opt, device)
            print(f"epoch {ep} loss: {loss:.4f}")
    else:
        ds = EgoNuScenesSeq(args.nusc_root, version=args.version, cameras=args.cameras, seq_len=args.seq_len)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=lambda x: x[0])
        for ep in range(1, args.epochs + 1):
            loss = ego_epoch(model, loader, opt, device)
            print(f"epoch {ep} ego-loss: {loss:.4f}")

    # Save
    payload = {
        "connector": model.connector.state_dict(),
        "ego_head": model.ego_head.state_dict(),
        "d_ext": model.ext.d_out,
        "d_model": model.qwen.d_model,
        "vision": args.vision,
        "qwen_id": mcfg.qwen_id,
        "mode": args.mode,
    }
    torch.save(payload, args.save)
    print(f"Saved: {args.save}")


if __name__ == "__main__":
    main()
