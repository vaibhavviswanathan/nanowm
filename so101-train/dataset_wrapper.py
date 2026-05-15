"""
SO-101 World-Model Dataset Wrapper
==================================

Unified PyTorch Dataset that loads multiple LeRobot v3.0 sources and yields
clips ready for NanoWM-B/2 training.

Responsibilities:
  - Multi-source loading with per-source action-key + camera-key remap
  - 30fps -> 15fps stride subsampling
  - Combined-stats action normalization (see compute_stats.py)
  - Clip-coherent augmentations: ColorJitter, RandomResizedCrop, GaussianNoise,
    ActionNoise, CameraDropout

Output schema per item:
  {
    "frames":    Tensor [T, V, C, H, W] in [0, 1]
    "view_mask": BoolTensor [V]              # False where view was dropped
    "actions":   Tensor [T, 6]                # normalized canonical joint positions
    "task":      str                          # natural-language task description
    "source":    str                          # source repo_id for debugging
  }

Adjust ACTION_KEY_MAPS / CAMERA_KEY_MAPS for additional sources (e.g. your own
SO-101 task data). The canonical joint order is fixed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import ColorJitter

from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ---------------------------------------------------------------------------
# Canonical schema
# ---------------------------------------------------------------------------

CANONICAL_JOINTS: Tuple[str, ...] = (
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
)
CANONICAL_VIEWS: Tuple[str, ...] = ("wrist", "top")


# Maps source repo_id -> ordered list of raw action key names, in canonical order.
# Add an entry here for your own SO-101 task dataset.
ACTION_KEY_MAPS: Dict[str, List[str]] = {
    "lerobot/svla_so100_pickplace": [
        "main_shoulder_pan", "main_shoulder_lift", "main_elbow_flex",
        "main_wrist_flex", "main_wrist_roll", "main_gripper",
    ],
    "lerobot/svla_so100_stacking": [
        "main_shoulder_pan", "main_shoulder_lift", "main_elbow_flex",
        "main_wrist_flex", "main_wrist_roll", "main_gripper",
    ],
    "lerobot/svla_so100_sorting": [
        "main_shoulder_pan", "main_shoulder_lift", "main_elbow_flex",
        "main_wrist_flex", "main_wrist_roll", "main_gripper",
    ],
    "lerobot/svla_so101_pickplace": [
        "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
        "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
    ],
    # Add your own task data here, e.g.:
    # "your-username/so101_my_task": [
    #     "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    #     "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
    # ],
}

# Maps source repo_id -> {canonical_view_name: raw_dataset_key}.
# If a view doesn't exist for a source, omit it (will return zeros + mask False).
CAMERA_KEY_MAPS: Dict[str, Dict[str, str]] = {
    "lerobot/svla_so100_pickplace": {
        "top": "observation.images.top",
        "wrist": "observation.images.wrist",
    },
    "lerobot/svla_so100_stacking": {
        "top": "observation.images.top",
        "wrist": "observation.images.wrist",
    },
    "lerobot/svla_so100_sorting": {
        "top": "observation.images.top",
        "wrist": "observation.images.wrist",
    },
    "lerobot/svla_so101_pickplace": {
        "top": "observation.images.top",
        "wrist": "observation.images.wrist",
    },
}


# ---------------------------------------------------------------------------
# Clip-coherent augmentations
# ---------------------------------------------------------------------------

class ClipCoherentAug:
    """
    Spatial augs use the SAME random params across all frames in a clip
    (preserves temporal coherence). Sensor-noise aug is per-frame.
    """

    def __init__(
        self,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.2,
        hue: float = 0.05,
        crop_scale: Tuple[float, float] = (0.9, 1.0),
        noise_std: float = 0.015,
    ):
        self._cj = ColorJitter(brightness, contrast, saturation, hue)
        self.crop_scale = crop_scale
        self.noise_std = noise_std

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        # clip: [T, C, H, W] in [0, 1]
        T_, C, H, W = clip.shape

        # --- ColorJitter, same params across clip ---
        fn_idx, b, c, s, h = ColorJitter.get_params(
            self._cj.brightness, self._cj.contrast,
            self._cj.saturation, self._cj.hue,
        )
        out = clip.clone()
        for fn_id in fn_idx:
            if fn_id == 0 and b is not None:
                out = TF.adjust_brightness(out, b)
            elif fn_id == 1 and c is not None:
                out = TF.adjust_contrast(out, c)
            elif fn_id == 2 and s is not None:
                out = TF.adjust_saturation(out, s)
            elif fn_id == 3 and h is not None:
                out = TF.adjust_hue(out, h)

        # --- RandomResizedCrop, same crop across clip ---
        lo, hi = self.crop_scale
        scale = lo + (hi - lo) * torch.rand(1).item()
        new_h, new_w = max(1, int(H * scale)), max(1, int(W * scale))
        top = torch.randint(0, H - new_h + 1, (1,)).item()
        left = torch.randint(0, W - new_w + 1, (1,)).item()
        out = out[:, :, top:top + new_h, left:left + new_w]
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

        # --- Gaussian noise, per-frame ---
        out = out + torch.randn_like(out) * self.noise_std
        return out.clamp_(0.0, 1.0)


# ---------------------------------------------------------------------------
# Main dataset
# ---------------------------------------------------------------------------

class SO101WorldModelDataset(Dataset):
    """
    Unified loader for the SO-101 world-model pretrain/finetune mix.

    Args:
      source_repo_ids:     list of LeRobot v3.0 dataset repo_ids; each must have
                           an entry in ACTION_KEY_MAPS and CAMERA_KEY_MAPS.
      clip_length:         number of frames per clip (at target_fps).
      target_fps:          frames per second after subsampling.
      image_size:          square crop / resize target.
      stats_path:          path to combined_stats.json (from compute_stats.py).
      augment:             whether to apply augmentations.
      camera_dropout_p:    probability of dropping one (random) view per clip.
      action_noise_std:    Gaussian std (in normalized action units) added to actions.
      streaming:           if True, use StreamingLeRobotDataset (no local cache).
    """

    def __init__(
        self,
        source_repo_ids: List[str],
        clip_length: int = 16,
        target_fps: int = 15,
        image_size: int = 256,
        stats_path: Optional[str] = None,
        augment: bool = True,
        camera_dropout_p: float = 0.15,
        action_noise_std: float = 0.01,
        streaming: bool = False,
    ):
        super().__init__()
        self.clip_length = clip_length
        self.target_fps = target_fps
        self.image_size = image_size
        self.augment = augment
        self.camera_dropout_p = camera_dropout_p
        self.action_noise_std = action_noise_std

        # Lazy import to avoid hard dep when not streaming
        if streaming:
            from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
            loader_cls = StreamingLeRobotDataset
        else:
            loader_cls = LeRobotDataset

        self.sources = []
        for repo_id in source_repo_ids:
            if repo_id not in ACTION_KEY_MAPS:
                raise KeyError(f"No ACTION_KEY_MAPS entry for {repo_id}")
            if repo_id not in CAMERA_KEY_MAPS:
                raise KeyError(f"No CAMERA_KEY_MAPS entry for {repo_id}")
            ds = loader_cls(repo_id)
            src_fps = int(getattr(ds.meta, "fps", 30)) if hasattr(ds, "meta") else 30
            self.sources.append({
                "repo_id": repo_id,
                "dataset": ds,
                "action_keys": ACTION_KEY_MAPS[repo_id],
                "camera_keys": CAMERA_KEY_MAPS[repo_id],
                "src_fps": src_fps,
                "stride": max(1, src_fps // target_fps),
            })

        self.index = self._build_index()

        if stats_path and Path(stats_path).exists():
            stats = json.loads(Path(stats_path).read_text())
            self.action_mean = torch.tensor(stats["action_mean"], dtype=torch.float32)
            self.action_std = torch.tensor(stats["action_std"], dtype=torch.float32)
        else:
            print("[SO101WMDataset] WARN: no stats_path; using identity normalization. "
                  "Run compute_stats.py first.")
            self.action_mean = torch.zeros(6)
            self.action_std = torch.ones(6)

        self.augmenter = ClipCoherentAug() if augment else None

    # -- index ---------------------------------------------------------------

    def _build_index(self) -> List[Tuple[int, int, int]]:
        """Pre-compute (source_idx, start_frame, stride) for every valid clip."""
        index: List[Tuple[int, int, int]] = []
        for src_idx, src in enumerate(self.sources):
            ds = src["dataset"]
            stride = src["stride"]
            clip_span_src = self.clip_length * stride

            # LeRobot v3.0 exposes episode boundaries via episode_data_index
            ep_index = ds.episode_data_index
            n_ep = len(ep_index["from"])
            for ep_idx in range(n_ep):
                ep_start = int(ep_index["from"][ep_idx])
                ep_end = int(ep_index["to"][ep_idx])
                max_start = ep_end - clip_span_src
                # Step by stride; gives reasonable coverage without combinatorial blowup
                for start in range(ep_start, max_start + 1, stride):
                    index.append((src_idx, start, stride))
        return index

    def __len__(self) -> int:
        return len(self.index)

    # -- item ----------------------------------------------------------------

    def __getitem__(self, idx: int):
        src_idx, start, stride = self.index[idx]
        src = self.sources[src_idx]
        ds = src["dataset"]

        frame_ids = [start + i * stride for i in range(self.clip_length)]
        frames_per_view: Dict[str, List[torch.Tensor]] = {v: [] for v in CANONICAL_VIEWS}
        actions: List[torch.Tensor] = []
        task_str = ""

        for fi in frame_ids:
            sample = ds[fi]

            # --- action: gather + concat in canonical order ---
            # Some datasets store the 6-dim action as a single "action" tensor;
            # others store each joint as a separate column. Handle both.
            if "action" in sample and isinstance(sample["action"], torch.Tensor) \
                    and sample["action"].numel() == 6:
                a = sample["action"].float()
            else:
                a = torch.stack([sample[k].float() for k in src["action_keys"]], dim=-1)
            actions.append(a)

            # --- per-view image fetch + resize ---
            for v in CANONICAL_VIEWS:
                cam_key = src["camera_keys"].get(v)
                if cam_key is None or cam_key not in sample:
                    img = torch.zeros(3, self.image_size, self.image_size)
                else:
                    img = sample[cam_key]  # [C, H, W], float in [0, 1]
                    if img.dtype != torch.float32:
                        img = img.float() / 255.0
                    if img.shape[-2] != self.image_size or img.shape[-1] != self.image_size:
                        img = F.interpolate(
                            img.unsqueeze(0),
                            size=(self.image_size, self.image_size),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)
                frames_per_view[v].append(img)

            if not task_str and "task" in sample:
                task_str = sample["task"] if isinstance(sample["task"], str) else ""

        actions_t = torch.stack(actions)  # [T, 6]
        actions_t = (actions_t - self.action_mean) / (self.action_std + 1e-6)

        view_clips = [torch.stack(frames_per_view[v]) for v in CANONICAL_VIEWS]  # each [T, C, H, W]
        view_mask = torch.ones(len(CANONICAL_VIEWS), dtype=torch.bool)

        if self.augment:
            for i, clip in enumerate(view_clips):
                view_clips[i] = self.augmenter(clip)

            # Mutually-exclusive camera dropout
            if torch.rand(1).item() < self.camera_dropout_p and len(view_clips) > 1:
                drop_idx = int(torch.randint(0, len(view_clips), (1,)).item())
                view_clips[drop_idx] = torch.zeros_like(view_clips[drop_idx])
                view_mask[drop_idx] = False

            if self.action_noise_std > 0:
                actions_t = actions_t + torch.randn_like(actions_t) * self.action_noise_std

        frames = torch.stack(view_clips, dim=1)  # [T, V, C, H, W]

        return {
            "frames": frames,
            "view_mask": view_mask,
            "actions": actions_t,
            "task": task_str,
            "source": src["repo_id"],
        }


# ---------------------------------------------------------------------------
# Convenience: pretrain mix
# ---------------------------------------------------------------------------

PRETRAIN_SOURCES: List[str] = [
    "lerobot/svla_so100_pickplace",
    "lerobot/svla_so100_stacking",
    "lerobot/svla_so100_sorting",
    "lerobot/svla_so101_pickplace",
]


def build_pretrain_dataset(stats_path: str = "combined_stats.json", **kwargs):
    return SO101WorldModelDataset(
        source_repo_ids=PRETRAIN_SOURCES,
        stats_path=stats_path,
        **kwargs,
    )
