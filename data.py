import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from PIL import Image
from nuscenes.nuscenes import NuScenes
from torch.utils.data import Dataset

DEFAULT_CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]


logger = logging.getLogger(__name__)


@dataclass
class DataCfg:
    nusc_root: str
    version: str = "v1.0-mini"
    cameras: Optional[List[str]] = None
    captions_json: Optional[str] = None
    dataset: str = "nuscenes"
    nuinteract_dir: Optional[str] = None
    nuinteract_caption_strategy: str = "overall"
    max_samples: Optional[int] = None
    require_files: bool = False


def resolve_nusc_root(nusc_root: str, version: str) -> str:
    """Allow users to pass either the nuScenes root or the version folder directly."""
    root = os.path.abspath(nusc_root)
    version_dir = os.path.join(root, version)
    if os.path.isdir(version_dir):
        return root
    parent = os.path.dirname(root.rstrip(os.sep))
    if os.path.basename(root.rstrip(os.sep)) == version and os.path.isdir(os.path.join(parent, version)):
        return parent
    raise FileNotFoundError(
        f"Could not find nuScenes version '{version}' under '{nusc_root}'. "
        "Pass the dataset root (directory that contains samples/, sweeps/, maps/, v1.0-*/)."
    )

class NuScenesMultiCamDataset(Dataset):

    def __init__(self, cfg: DataCfg):
        self.cfg = cfg
        self.nusc_root = resolve_nusc_root(cfg.nusc_root, cfg.version)
        self.nusc = NuScenes(version=cfg.version, dataroot=self.nusc_root, verbose=False)
        self.cams = cfg.cameras or list(DEFAULT_CAMERAS)
        self.require_files = bool(cfg.require_files)
        logger.info(
            "Initializing NuScenesMultiCamDataset version=%s with cameras=%s",
            cfg.version,
            ",".join(self.cams),
        )
        skipped = 0
        self.samples = []
        for scene in self.nusc.scene:
            tok = scene["first_sample_token"]
            while tok:
                sample = self.nusc.get("sample", tok)
                img_paths = []
                missing = False
                for cam in self.cams:
                    if cam in sample["data"]:
                        sd = self.nusc.get("sample_data", sample["data"][cam])
                        path = os.path.join(self.nusc.dataroot, sd["filename"])
                        if self.require_files and not os.path.exists(path):
                            missing = True
                            break
                        if os.path.exists(path):
                            img_paths.append(path)
                if missing:
                    skipped += 1
                elif img_paths and (not self.require_files or len(img_paths) == len(self.cams)):
                    self.samples.append({"token": tok, "img_paths": img_paths})
                tok = sample["next"]
        self.captions: Dict[str, str] = {}
        if cfg.captions_json and os.path.exists(cfg.captions_json):
            with open(cfg.captions_json) as f:
                self.captions = json.load(f)
            logger.info("Loaded %d captions from %s", len(self.captions), cfg.captions_json)
        elif cfg.captions_json:
            logger.warning("Caption file %s not found; proceeding without captions", cfg.captions_json)
        if skipped:
            logger.info("Skipped %d samples missing required camera files", skipped)
        logger.info("NuScenesMultiCamDataset ready with %d samples", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        row = self.samples[i]
        images = [Image.open(p).convert("RGB") for p in row["img_paths"]]
        text = self.captions.get(row["token"], None)
        return {"images": images, "text": text, "token": row["token"]}


class NuInteractDenseCaptionDataset(Dataset):
    """Load dense captions released with NuInteract for multi-view training."""

    def __init__(self, cfg: DataCfg):
        if not cfg.nuinteract_dir:
            raise ValueError("nuinteract_dir must be set when dataset='nuinteract'.")
        self.cfg = cfg
        self.cams = cfg.cameras or list(DEFAULT_CAMERAS)
        self.require_files = bool(cfg.require_files)
        self.nusc_root = resolve_nusc_root(cfg.nusc_root, cfg.version)
        self.nusc = NuScenes(version=cfg.version, dataroot=self.nusc_root, verbose=False)
        self.samples = []
        skipped_missing = 0
        json_files = []
        for root, _, files in os.walk(cfg.nuinteract_dir):
            for fname in files:
                if fname.endswith(".json") and fname != "token_name.json":
                    json_files.append(os.path.join(root, fname))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found under {cfg.nuinteract_dir} for NuInteract.")
        json_files.sort()
        logger.info("Found %d NuInteract JSON shards under %s", len(json_files), cfg.nuinteract_dir)
        for path in json_files:
            try:
                data = json.load(open(path))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to load %s: %s", path, exc)
                continue
            for row in data:
                token = row.get("token")
                img_paths = self._resolve_cam_paths(token, row)
                captions = self._select_caption_dict(row)
                if not captions:
                    continue
                text = self._build_caption(captions)
                if not img_paths or not text:
                    continue
                if self.require_files and len(img_paths) != len(self.cams):
                    skipped_missing += 1
                    continue
                if self.require_files and not all(os.path.exists(p) for p in img_paths):
                    skipped_missing += 1
                    continue
                self.samples.append({
                    "token": token,
                    "img_paths": img_paths,
                    "text": text,
                })
                if cfg.max_samples and len(self.samples) >= cfg.max_samples:
                    break
            if cfg.max_samples and len(self.samples) >= cfg.max_samples:
                break
        if not self.samples:
            raise RuntimeError("NuInteract dataset produced zero usable samples. Check paths and strategy.")
        if skipped_missing:
            logger.info("Skipped %d NuInteract samples missing required camera files", skipped_missing)
        logger.info("NuInteract dataset ready with %d samples", len(self.samples))

    def _resolve_cam_paths(self, token: Optional[str], row: Dict[str, Any]) -> List[str]:
        paths = self._paths_from_nuscenes(token)
        if paths:
            return paths
        json_paths = self._paths_from_json(row)
        if json_paths and all(os.path.exists(p) for p in json_paths):
            return json_paths
        return []

    def _paths_from_nuscenes(self, token: Optional[str]) -> List[str]:
        if not token:
            return []
        try:
            sample = self.nusc.get("sample", token)
        except Exception:
            logger.debug("Sample token %s missing from nuScenes index", token)
            return []
        img_paths = []
        for cam in self.cams:
            if cam not in sample["data"]:
                continue
            sd = self.nusc.get("sample_data", sample["data"][cam])
            img_paths.append(os.path.join(self.nusc.dataroot, sd["filename"]))
        return img_paths

    def _paths_from_json(self, row: Dict[str, Any]) -> List[str]:
        cam_field = row.get("cam_path") or row.get("cam_paths")
        cam_map: Dict[str, str] = {}
        if isinstance(cam_field, dict):
            cam_map = cam_field
        elif isinstance(cam_field, list):
            for path in cam_field:
                name = self._infer_cam_name(path)
                if name and path:
                    cam_map.setdefault(name, path)
        elif isinstance(cam_field, str):
            name = self._infer_cam_name(cam_field)
            if name:
                cam_map[name] = cam_field
        img_paths = []
        for cam in self.cams:
            rel = cam_map.get(cam)
            if not rel:
                continue
            img_paths.append(rel if os.path.isabs(rel) else os.path.join(self.nusc.dataroot, rel))
        return img_paths

    @staticmethod
    def _infer_cam_name(path: str) -> Optional[str]:
        for cam in DEFAULT_CAMERAS:
            if cam in path:
                return cam
        return None

    def _select_caption_dict(self, row: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Use GPT captions only; skip samples without them."""
        captions = row.get("gpt_caption")
        if captions:
            return captions
        return None

    def _build_caption(self, captions: Dict[str, str]) -> Optional[str]:
        if not captions:
            return None
        strategy = (self.cfg.nuinteract_caption_strategy or "overall").lower()
        if strategy == "overall":
            front = captions.get("FRONT") or captions.get("Cam_Front")
            back = captions.get("BACK") or captions.get("Cam_Back")
            overall = captions.get("OVERALL")
            text_parts = [p.strip() for p in [overall, front, back] if p]
            return " ".join(text_parts).strip() if text_parts else None
        if strategy == "per_view_concat":
            pieces = []
            for cam in self.cams:
                part = captions.get(cam) or captions.get(cam.replace("CAM_", ""))
                if part:
                    pieces.append(part.strip())
            if not pieces and captions.get("OVERALL"):
                pieces.append(captions["OVERALL"].strip())
            return " ".join(pieces).strip() if pieces else None
        raise ValueError(f"Unknown NuInteract caption strategy: {self.cfg.nuinteract_caption_strategy}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        images = [Image.open(p).convert("RGB") for p in sample["img_paths"] if os.path.exists(p)]
        if not images:
            raise FileNotFoundError(f"No camera images found for sample {sample['token']}")
        return {"images": images, "text": sample["text"], "token": sample["token"]}


def build_dataset(cfg: DataCfg) -> Dataset:
    logger.info("Building dataset type=%s", cfg.dataset)
    if cfg.dataset == "nuscenes":
        ds = NuScenesMultiCamDataset(cfg)
    elif cfg.dataset == "nuinteract":
        ds = NuInteractDenseCaptionDataset(cfg)
    else:
        raise ValueError(f"Unsupported dataset type: {cfg.dataset}")
    logger.info("Dataset %s ready with %d samples", cfg.dataset, len(ds))
    return ds
