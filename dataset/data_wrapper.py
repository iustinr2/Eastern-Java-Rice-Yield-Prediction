from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def to_float(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.float()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return torch.tensor(x, dtype=torch.float32)


def fix_sentinel(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 5:
        raise ValueError(f"Sentinel image data must be 5d, got {tuple(x.shape)}")

    return x.permute(0, 1, 4, 2, 3).contiguous()


def pool_weather(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x

    if x.ndim == 2:
        return x.mean(dim=0)

    if x.ndim >= 3:
        size = x.shape[-1]
        return x.reshape(-1, size).mean(dim=0)

    raise ValueError(f"Incorrect weather shape: {tuple(x.shape)}")


def pool_sentinel(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 5:
        raise ValueError(f"Incorrect sentinel shape: {tuple(x.shape)}")

    return x.mean(dim=(0, 1, 3, 4))


class MultiModalRiceWrapper(Dataset):
    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.base_dataset[idx]

        sentinel = to_float(item["sentinel"])

        if sentinel.ndim != 5:
            raise ValueError(f"Sentinel image data must be 5d, got {tuple(sentinel.shape)}")

        if sentinel.shape[2] <= 16 and sentinel.shape[-1] > 16:
            sentinel_tensor = sentinel.contiguous()
        else:
            sentinel_tensor = fix_sentinel(sentinel)

        weather_tensor = to_float(item["weather"])
        weather_feat = pool_weather(weather_tensor)
        sentinel_feat = pool_sentinel(sentinel_tensor)

        out = dict(item)
        out["sentinel_tensor"] = sentinel_tensor
        out["weather_tensor"] = weather_tensor
        out["sentinel_feat"] = sentinel_feat
        out["weather_feat"] = weather_feat
        out["target_norm"] = torch.tensor(float(item["target_norm"]), dtype=torch.float32)
        out["target"] = torch.tensor(float(item["target"]), dtype=torch.float32)
        out["target_raw"] = torch.tensor(float(item["target_raw"]), dtype=torch.float32)

        return out