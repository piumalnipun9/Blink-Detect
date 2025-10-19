import pathlib
import numpy as np
from typing import Tuple, Dict

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class TalkingFaceDataset(Dataset):
    """Loads Talking Face frames and their 68-point landmarks."""

    def __init__(self, root: str, resize: Tuple[int, int] = (96, 96)) -> None:
        self.root = pathlib.Path(root)
        self.resize = resize
        self.images = sorted(self.root.glob("*.jpg"))
        if not self.images:
            raise RuntimeError(f"No .jpg files found in {self.root}")
        # points are assumed one level up in a ``points`` directory
        self.points: Dict[str, pathlib.Path] = {
            img.stem: self.root.parent.joinpath("points", f"{img.stem}.pts")
            for img in self.images
        }

    def __len__(self) -> int:
        return len(self.images)

    def _load_pts(self, path: pathlib.Path) -> np.ndarray:
        with path.open("r", encoding="ascii") as handle:
            lines = [line.strip() for line in handle]
        start = lines.index("{") + 1
        end = lines.index("}")
        coords = [tuple(map(float, line.split())) for line in lines[start:end]]
        return np.asarray(coords, dtype=np.float32)

    def __getitem__(self, index: int):
        img_path = self.images[index]
        pts_path = self.points[img_path.stem]

        image = Image.open(img_path).convert("RGB").resize(self.resize)
        array = np.array(image)
        # convert to CHW tensor, normalized to [0, 1]
        image_tensor = torch.from_numpy(np.transpose(array, (2, 0, 1))).float() / 255.0
        landmarks = torch.from_numpy(self._load_pts(pts_path))

        return image_tensor, landmarks


class EyePatchExtractor(nn.Module):
    """Extracts left/right eye crops around landmark centers."""

    def __init__(self, crop_size: int = 48) -> None:
        super().__init__()
        self.crop_size = crop_size

    def forward(self, images: torch.Tensor, landmarks: torch.Tensor) -> torch.Tensor:
        left_eye_idx = [36, 37, 38, 39, 40, 41]
        right_eye_idx = [42, 43, 44, 45, 46, 47]

        patches = []
        for img, pts in zip(images, landmarks):
            def crop(indices):
                eye_pts = pts[indices]
                center = eye_pts.mean(dim=0)
                half = self.crop_size // 2
                x_min = int(max(center[0] - half, 0))
                y_min = int(max(center[1] - half, 0))
                x_max = x_min + self.crop_size
                y_max = y_min + self.crop_size
                patch = img[:, y_min:y_max, x_min:x_max]
                # ensure consistent output size even near borders
                patch = nn.functional.interpolate(
                    patch.unsqueeze(0),
                    size=(self.crop_size, self.crop_size),
                    mode="bilinear",
                    align_corners=False,
                )
                return patch.squeeze(0)

            left_patch = crop(left_eye_idx)
            right_patch = crop(right_eye_idx)
            patches.append(torch.stack([left_patch, right_patch], dim=0))

        return torch.stack(patches, dim=0)  # [batch, 2, C, H, W]


class SimpleConvNet(nn.Module):
    """Example single-layer conv net over extracted eye patches."""

    def __init__(self, in_channels: int = 3, num_filters: int = 16) -> None:
        super().__init__()
        self.extractor = EyePatchExtractor()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # 48x48 patches become 24x24 after pooling
        self.head = nn.Linear(num_filters * 2 * 24 * 24, 1)

    def forward(self, images: torch.Tensor, landmarks: torch.Tensor) -> torch.Tensor:
        eye_patches = self.extractor(images, landmarks)
        batch_size, eyes, channels, height, width = eye_patches.shape
        eye_patches = eye_patches.view(batch_size * eyes, channels, height, width)
        features = self.conv(eye_patches)
        features = features.view(batch_size, eyes, -1).flatten(1)
        return self.head(features)


if __name__ == "__main__":
    dataset = TalkingFaceDataset(root="images/images")
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    model = SimpleConvNet()
    for images, landmarks in loader:
        outputs = model(images, landmarks)
        print(outputs.shape)
        break
