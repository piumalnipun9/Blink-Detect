"""Preprocess the Talking Face dataset into BlinkFormer-ready samples.

Each sample is a 13-frame eye tensor saved as .npy along with metadata so we
can reuse the existing SynthBlink/HUST loaders.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

SequenceLike = Sequence[int] | np.ndarray

LEFT_EYE_IDX = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_IDX = [42, 43, 44, 45, 46, 47]


def parse_pts(path: pathlib.Path) -> np.ndarray:
    with path.open("r", encoding="ascii") as handle:
        lines = [line.strip() for line in handle]
    start = lines.index("{") + 1
    end = lines.index("}")
    coords = [tuple(map(float, line.split())) for line in lines[start:end]]
    return np.asarray(coords, dtype=np.float32)


def crop_eye(frame: np.ndarray, landmarks: np.ndarray, indices: SequenceLike, size: int) -> np.ndarray:
    eye_pts = landmarks[indices]
    center = eye_pts.mean(axis=0)
    half = size // 2
    x_min = int(max(center[0] - half, 0))
    y_min = int(max(center[1] - half, 0))
    x_max = x_min + size
    y_max = y_min + size
    crop = frame[y_min:y_max, x_min:x_max]
    if crop.shape[0] != size or crop.shape[1] != size:
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    return crop


def blink_strength_from_landmarks(prev: np.ndarray, current: np.ndarray, next_: np.ndarray) -> float:
    # Hacky proxy: ratio between vertical eyelid distance and horizontal width.
    # Works okay for synthetic/Talking Face where 68-point landmarks exist.
    def eye_ratio(landmarks: np.ndarray, indices: SequenceLike) -> float:
        eye = landmarks[indices]
        horizontal = np.linalg.norm(eye[0] - eye[3])
        vertical = (np.linalg.norm(eye[1] - eye[5]) + np.linalg.norm(eye[2] - eye[4])) / 2
        if horizontal <= 1e-6:
            return 0.0
        return vertical / horizontal

    ratio_now = eye_ratio(current, LEFT_EYE_IDX) + eye_ratio(current, RIGHT_EYE_IDX)
    ratio_prev = eye_ratio(prev, LEFT_EYE_IDX) + eye_ratio(prev, RIGHT_EYE_IDX)
    ratio_next = eye_ratio(next_, LEFT_EYE_IDX) + eye_ratio(next_, RIGHT_EYE_IDX)
    baseline = (ratio_prev + ratio_next) / 2
    strength = max(baseline - ratio_now, 0.0)
    return float(np.clip(strength, 0.0, 1.0))


def sliding_windows(items: List[Tuple[pathlib.Path, pathlib.Path]], window: int, stride: int) -> List[List[Tuple[pathlib.Path, pathlib.Path]]]:
    chunks: List[List[Tuple[pathlib.Path, pathlib.Path]]] = []
    for start in range(0, len(items) - window + 1, stride):
        slice_ = items[start : start + window]
        chunks.append(slice_)
    return chunks


def process_split(image_dir: pathlib.Path, points_dir: pathlib.Path, output_dir: pathlib.Path, window: int = 13, stride: int = 1) -> Dict[str, int]:
    samples: Dict[str, int] = {}
    frames = sorted(image_dir.glob("franck_*.jpg"))
    paired = [(frame, points_dir.joinpath(frame.stem + ".pts")) for frame in frames]
    chunks = sliding_windows(paired, window=window, stride=stride)

    window_id = 0
    for chunk in tqdm(chunks, desc=f"{image_dir.name}"):
        images = []
        landmarks = []
        valid = True
        for img_path, pts_path in chunk:
            if not pts_path.exists():
                valid = False
                break
            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                valid = False
                break
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            images.append(image_rgb)
            landmarks.append(parse_pts(pts_path))
        if not valid:
            continue

        left_eye = [crop_eye(img, lm, LEFT_EYE_IDX, 48) for img, lm in zip(images, landmarks)]
        right_eye = [crop_eye(img, lm, RIGHT_EYE_IDX, 48) for img, lm in zip(images, landmarks)]
        stack = np.stack([np.concatenate([l, r], axis=2) for l, r in zip(left_eye, right_eye)], axis=0)  # [T, 48, 96, 3]
        stack = stack[:, :, :, :3]  # left crop only by default

        # Format to [T, 3, 48, 48]
        tensor = np.stack(left_eye, axis=0)  # using left eye only
        tensor = tensor.transpose(0, 3, 1, 2)

        # Blink strength per frame via heuristic
        blink_strength = []
        for idx in range(window):
            prev_idx = max(idx - 1, 0)
            next_idx = min(idx + 1, window - 1)
            strength = blink_strength_from_landmarks(
                landmarks[prev_idx], landmarks[idx], landmarks[next_idx]
            )
            blink_strength.append(strength)
        blink_strength_arr = np.asarray(blink_strength, dtype=np.float32)

        # Binary label based on strength threshold
        blink_label = int(np.max(blink_strength_arr) > 0.15)

        sample_id = f"talkingface_{image_dir.name}_{window_id:05d}"
        np.save(output_dir / f"{sample_id}.npy", tensor.astype(np.float32))
        meta = {
            "id": sample_id,
            "label": blink_label,
            "blink_strength": blink_strength_arr.tolist(),
            "frames": [str(path.relative_to(image_dir.parent)) for path, _ in chunk],
        }
        with (output_dir / f"{sample_id}.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f)

        samples[sample_id] = blink_label
        window_id += 1

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Talking Face into BlinkFormer format.")
    parser.add_argument("--images", type=pathlib.Path, default=pathlib.Path("images/images"), help="Directory with Talking Face frames")
    parser.add_argument("--points", type=pathlib.Path, default=pathlib.Path("points"), help="Directory with 68-point .pts")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("data_preprocess/talking_face"))
    parser.add_argument("--window", type=int, default=13)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    mapping = process_split(args.images, args.points, args.output, window=args.window, stride=args.stride)

    with (args.output / "train_data.json").open("w", encoding="utf-8") as handle:
        json.dump(mapping, handle, indent=2)
    # For simplicity split same mapping into val/test
    with (args.output / "val_data.json").open("w", encoding="utf-8") as handle:
        json.dump({}, handle)
    with (args.output / "test_data.json").open("w", encoding="utf-8") as handle:
        json.dump({}, handle)


if __name__ == "__main__":
    main()
