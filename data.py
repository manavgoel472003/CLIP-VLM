import os
import math
from dataclasses import dataclass
from typing import List, Optional, Dict

from PIL import Image
from nuscenes.nuscenes import NuScenes
from torch.utils.data import Dataset

@dataclass
class DataCfg:
    nusc_root: str
    version: str = "v1.0-mini"
    cameras: List[str] = None
    captions_json: Optional[str] = None

class NuScenesMultiCamDataset(Dataset):

    def __init__(self, cfg: DataCfg):
        self.cfg = cfg
        self.nusc = NuScenes(version=self.version, dataroot=cfg.nusc_root, verbose=False)
        self.cams = cfg.cameras or [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_FRONT_LEFT",
        ]
        self.samples = []
        for scene in self.nusc.scene:
            tok = scene["first_sample_token"]
            while tok:
                sample = self.nusc.get("sample", tok)
                img_paths = []
                for cam in self.cams:
                    if cam in sample["data"]:
                        sd = self.nusc.get("sample_data", sample["data"][cam])
                        img_paths.append(os.path.join(self.nusc.dataroot, sd["filename"]))
                if img_paths:
                    self.samples.append({"token": tok, "img_paths": img_paths})
                tok = sample["next"]
        self.captions: Dict[str, str] = {}
        if cfg.captions_json and os.path.exists(cfg.captions_json):
            import json
            with open(cfg.captions_json) as f:
                self.captions = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        row = self.samples[i]
        images = [Image.open(p).convert("RGB") for p in row["img_paths"]]
        text = self.captions.get(row["token"], None)
        return {"images": images, "text": text, "token": row["token"]}

# Utils

def q_to_yaw(qw, qx, qy, qz) -> float:
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def ego_delta(prev_pose, next_pose):
    tx1, ty1, tz1, qw1, qx1, qy1, qz1 = prev_pose
    tx2, ty2, tz2, qw2, qx2, qy2, qz2 = next_pose
    yaw1 = q_to_yaw(qw1, qx1, qy1, qz1)
    yaw2 = q_to_yaw(qw2, qx2, qy2, qz2)
    dxg = tx2 - tx1
    dyg = ty2 - ty1
    dyaw = (yaw2 - yaw1 + math.pi) % (2 * math.pi) - math.pi
    c, s = math.cos(-yaw1), math.sin(-yaw1)
    dx = c * dxg - s * dyg
    dy = s * dxg + c * dyg
    return (dx, dy, dyaw)

class EgoNuScenesSeq(Dataset):
    """Temporal sequences for ego-motion.
    Returns: {"images_seq": list of list[PIL], "target": (dx,dy,dyaw), "tokens": [sample_tokens]}
    """
    def __init__(self, nusc_root: str, version: str = "v1.0-mini", cameras: Optional[List[str]] = None, seq_len: int = 3):
        self.nusc = NuScenes(version=version, dataroot=nusc_root, verbose=False)
        self.cams = cameras or [
            "CAM_FRONT","CAM_FRONT_RIGHT","CAM_BACK_RIGHT",
            "CAM_BACK","CAM_BACK_LEFT","CAM_FRONT_LEFT",
        ]
        self.seq_len = seq_len
        # Build sample list with image paths and ego pose
        samples = []
        for scene in self.nusc.scene:
            tok = scene["first_sample_token"]
            while tok:
                sm = self.nusc.get("sample", tok)
                img_paths = []
                pose = None
                for cam in self.cams:
                    if cam in sm["data"]:
                        sd = self.nusc.get("sample_data", sm["data"][cam])
                        img_paths.append(os.path.join(self.nusc.dataroot, sd["filename"]))
                        if pose is None:
                            ep = self.nusc.get("ego_pose", sd["ego_pose_token"])
                            t = ep["translation"]
                            r = ep["rotation"]
                            pose = (t[0], t[1], t[2], r[0], r[1], r[2], r[3])
                if img_paths and pose is not None:
                    samples.append({"token": tok, "img_paths": img_paths, "pose": pose, "next": sm["next"]})
                tok = sm["next"]
        # Build sequences
        self.seqs = []
        for i in range(self.seq_len - 1, len(samples) - 1):
            src = samples[i - (self.seq_len - 1): i + 1]
            nxt = samples[i + 1]
            self.seqs.append((src, nxt))

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        src, nxt = self.seqs[i]
        images_seq = [[Image.open(p).convert("RGB") for p in step["img_paths"]] for step in src]
        target = ego_delta(src[-1]["pose"], nxt["pose"])  # (dx, dy, dyaw)
        tokens = [step["token"] for step in src] + [nxt["token"]]
        return {"images_seq": images_seq, "target": target, "tokens": tokens}
