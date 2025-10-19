import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import print_on_rank_zero


class TalkingFaceDataset(Dataset):
    """Dataset wrapper for Talking Face windows exported by the preprocess script."""

    def __init__(
        self,
        root_path: str = os.path.join("BlinkFormer", "data_preprocess", "talking_face"),
        mode: str = "train",
        seq_length: int = 13,
        transform=None,
        eye_size: Tuple[int, int] | None = (48, 48),
        config=None,
    ) -> None:
        super().__init__()

        self.config = config
        self.seq_length = seq_length
        self.transform = transform
        self.eye_size = eye_size

        mode = mode.lower()
        self.mode = mode
        if mode not in {"train", "val", "test"}:
            raise ValueError("mode must be one of train/val/test")

        self.root_path = os.path.abspath(os.path.expanduser(root_path))
        index_path = os.path.join(self.root_path, f"{mode}_data.json")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Missing split file: {index_path}. Run the preprocessing script first.")

        with open(index_path, "r", encoding="utf-8") as handle:
            self.labels_dict: Dict[str, int] = json.load(handle)

        self.sample_ids: List[str] = sorted(self.labels_dict.keys())
        if not self.sample_ids:
            print_on_rank_zero(
                f"Warning: split '{mode}' in {self.root_path} has no samples. Check preprocessing ratios or rerun the script."
            )

        print_on_rank_zero(f"{mode} samples: {len(self.sample_ids)} from {self.root_path}")

        if self.seq_length not in (10, 13):
            raise ValueError("seq_length must be 10 or 13")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int):
        sample_id = self.sample_ids[index]
        label = self.labels_dict[sample_id]

        npy_path = os.path.join(self.root_path, f"{sample_id}.npy")
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"Eye tensor missing for {sample_id}: expected {npy_path}")

        frames = np.load(npy_path)
        if frames.shape[0] != self.seq_length:
            raise ValueError(
                f"Sample {sample_id} expected {self.seq_length} frames but found {frames.shape[0]}"
            )

        eye_tensor = torch.from_numpy(frames)
        if self.transform is not None:
            eye_tensor = self.transform(eye_tensor)

        anno_path = os.path.join(self.root_path, f"{sample_id}.json")
        if os.path.exists(anno_path):
            with open(anno_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            blink_strengths = np.asarray(meta.get("blink_strength", [0.0] * self.seq_length), dtype=np.float32)
        else:
            blink_strengths = np.zeros(self.seq_length, dtype=np.float32)

        return eye_tensor, label, [], sample_id, blink_strengths
