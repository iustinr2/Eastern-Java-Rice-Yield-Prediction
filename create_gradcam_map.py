from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torchf
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader


repo_root = Path("/vol/home/s3881946/Downloads/MMST-ViT-main")
finetune_file = repo_root / "main_finetune_mmst_vit.py"
checkpoint_root = repo_root / "RESULTS/multiseed_finetune_results_temporal"
output_dir = repo_root / "gradcam_outputs_temporal"

experiment_name = "standard_random_grouped"
seed_values = [0, 1, 2, 3, 4, 5, 42, 123, 777, 2025]
max_samples_per_seed = 37
max_scan_batches = 500

target_layer_name = "pvt_backbone.backbone.patch_embed1.proj"

output_tile_pixels = 4096
individual_tile_pixels = 1536

heatmap_cmap = "turbo"
overlay_alpha = 0.38

use_log_before_minmax = True
log_scale_factor = 1800.0

smooth_before_minmax = True
smooth_kernel_size = 5
smooth_sigma = 0.95

low_percentile = 1.0
high_percentile = 99.0
display_gamma = 1
cam_eps = 1e-20

apply_sentinel_valid_mask = True
valid_mask_threshold_ratio = 0.16
valid_mask_min_threshold = 1e-6
valid_mask_soft_power = 1.0

land_mask_gray_percentile = 12.0
land_mask_signal_percentile = 10.0
land_mask_blur_kernel_size = 9
land_mask_blur_sigma = 1.5
land_mask_hard_threshold = 0.30

background_gamma = 0.78
background_gray_overlay = False

tile_standardization_mode = "hybrid"
global_standardization_weight = 0.60
per_tile_standardization_weight = 0.40

save_individual_sample_maps = False

tile_names = ["tl", "tr", "bl", "br"]


def show(text: str):
    print(text, flush=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_finetune():
    if not finetune_file.exists():
        raise FileNotFoundError(f"fine tune file not found: {finetune_file}")

    spec = importlib.util.spec_from_file_location(
        f"finetune_module_for_gradcam_{experiment_name}",
        str(finetune_file),
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"could not import fine tune file: {finetune_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


ft = load_finetune()


def get_checkpoint(seed: int):
    return checkpoint_root / f"seed_{seed}" / "best_finetune_checkpoint.pt"


def get_seeds():
    seeds = []

    for seed in seed_values:
        path = get_checkpoint(seed)

        if path.exists():
            seeds.append(seed)

    if not seeds:
        raise FileNotFoundError(f"no checkpoints found: {checkpoint_root}")

    return seeds


def read_weights(path: Path):
    checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        weights = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        weights = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        weights = checkpoint
    else:
        raise TypeError(f"bad checkpoint type: {type(checkpoint)}")

    clean = {}

    for key, value in weights.items():
        key = str(key)

        if key.startswith("module."):
            key = key[len("module."):]

        clean[key] = value

    return clean


def load_checkpoint(model: nn.Module, path: Path):
    weights = read_weights(path)
    model_weights = model.state_dict()

    valid = {}

    for key, value in weights.items():
        if key not in model_weights:
            continue

        if tuple(model_weights[key].shape) != tuple(value.shape):
            continue

        valid[key] = value

    model.load_state_dict(valid, strict=False)


def get_regency_count(train_loader: DataLoader, val_loader: DataLoader):
    max_id = -1

    for loader in [train_loader, val_loader]:
        df = getattr(loader.dataset, "df", None)

        if df is None or "regency_idx" not in df.columns:
            continue

        max_id = max(max_id, int(df["regency_idx"].max()))

    if max_id < 0:
        return 1

    return max_id + 1


def get_config(seed: int):
    cfg = ft.FineTuneConfig(seed=seed)
    cfg.num_workers = 0
    cfg.batch_size = 1
    return cfg


def get_seed_data(seed: int):
    cfg = get_config(seed)
    set_seed(seed)

    model, train_loader, val_loader, scaler, baseline = ft.prepare_data_and_model(cfg)
    regency_count = get_regency_count(train_loader, val_loader)

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return cfg, train_loader, val_loader, regency_count


def get_batches(val_loader: DataLoader, max_samples: int):
    batches = []

    for index, batch in enumerate(val_loader):
        if index >= max_scan_batches:
            break

        batches.append(batch)

        if len(batches) >= max_samples:
            break

    if not batches:
        raise RuntimeError("no validation batches selected")

    return batches


def copy_batch(batch: dict[str, Any]):
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def get_module(model: nn.Module, name: str):
    modules = dict(model.named_modules())

    if name not in modules:
        found = "\n".join(list(modules.keys())[-180:])
        raise KeyError(f"target layer not found: {name}\n{found}")

    return modules[name]


class gradcam_hook:
    def __init__(self, layer: nn.Module):
        self.activations = []
        self.gradients = []
        self.forward_handle = layer.register_forward_hook(self.forward_hook)
        self.backward_handle = layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, inputs, output):
        self.activations.append(output)

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0])

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def join_tensors(values: list[torch.Tensor], reverse: bool = False):
    if not values:
        raise RuntimeError("gradcam hook did not capture tensors")

    tensors = list(values)

    if reverse and len(tensors) > 1:
        tensors = list(reversed(tensors))

    if len(tensors) == 1:
        return tensors[0]

    if all(tensor.ndim == tensors[0].ndim for tensor in tensors):
        return torch.cat(tensors, dim=0)

    raise RuntimeError("captured gradcam tensors had inconsistent dimensions")


def get_raw_cam(hook: gradcam_hook):
    activations = join_tensors(hook.activations, reverse=False).detach()
    gradients = join_tensors(hook.gradients, reverse=True).detach()

    if activations.ndim == 4:
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.abs((weights * activations).sum(dim=1))
        return cam.detach().cpu().numpy().astype(np.float32)

    if activations.ndim == 3:
        weights = gradients.mean(dim=1, keepdim=True)
        cam = torch.abs((weights * activations).sum(dim=1))
        arr = cam.detach().cpu().numpy().astype(np.float32)

        side = int(math.sqrt(arr.shape[-1]))

        if side * side != arr.shape[-1]:
            raise ValueError(f"cannot reshape cam: {arr.shape}")

        return arr.reshape(arr.shape[0], side, side).astype(np.float32)

    raise ValueError(f"bad activation shape: {tuple(activations.shape)}")


def get_shape(batch: dict[str, Any]):
    sentinel = batch["sentinel"]

    if sentinel.ndim == 6:
        return int(sentinel.shape[0]), int(sentinel.shape[1]), int(sentinel.shape[2])

    if sentinel.ndim == 5:
        return 1, int(sentinel.shape[0]), int(sentinel.shape[1])

    raise ValueError(f"bad sentinel shape: {tuple(sentinel.shape)}")


def get_tile_cams(raw: np.ndarray, batch: dict[str, Any]):
    raw = np.asarray(raw, dtype=np.float32)

    if raw.ndim != 3:
        raise ValueError(f"bad raw cam shape: {raw.shape}")

    batch_size, time_count, tile_count = get_shape(batch)

    if tile_count != 4:
        raise ValueError(f"expected 4 sentinel tiles, got {tile_count}")

    count = raw.shape[0]
    height = raw.shape[-2]
    width = raw.shape[-1]

    if count == batch_size * time_count * tile_count:
        return raw.reshape(batch_size, time_count, tile_count, height, width).mean(axis=(0, 1)).astype(np.float32)

    if count == time_count * tile_count:
        return raw.reshape(time_count, tile_count, height, width).mean(axis=0).astype(np.float32)

    if count == tile_count:
        return raw.astype(np.float32)

    if count == 1:
        return np.repeat(raw, 4, axis=0).astype(np.float32)

    raise RuntimeError(f"could not extract tile cams: raw={raw.shape}")


def get_gradcam(model: nn.Module, batch: dict[str, Any], device: torch.device):
    model.eval()

    layer = get_module(model, target_layer_name)
    hook = gradcam_hook(layer)

    try:
        device_batch = ft.move_batch_to_device(copy_batch(batch), device)

        model.zero_grad(set_to_none=True)
        pred = ft.forward_mmst_model(model, device_batch)
        score = pred.reshape(-1)[0]
        score.backward(retain_graph=False)

        raw = get_raw_cam(hook)

    finally:
        hook.close()

    tile_cams = get_tile_cams(raw, batch)
    tile_cams = np.nan_to_num(
        np.abs(tile_cams),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    return tile_cams, float(score.detach().cpu().item())


def start_model(model: nn.Module, batch: dict[str, Any], device: torch.device):
    model.to(device)
    model.eval()

    device_batch = ft.move_batch_to_device(copy_batch(batch), device)

    with torch.no_grad():
        _ = ft.forward_mmst_model(model, device_batch)


def resize_2d_shape(arr: np.ndarray, shape: tuple[int, int], mode: str = "bilinear"):
    arr = np.asarray(arr, dtype=np.float32)
    tensor = torch.from_numpy(arr).view(1, 1, arr.shape[0], arr.shape[1])

    if mode in {"bilinear", "bicubic"}:
        out = torchf.interpolate(tensor, size=shape, mode=mode, align_corners=False)
    else:
        out = torchf.interpolate(tensor, size=shape, mode=mode)

    return out.squeeze().detach().cpu().numpy().astype(np.float32)


def resize_2d(arr: np.ndarray, size: int, mode: str = "bicubic"):
    return resize_2d_shape(arr, (size, size), mode)


def resize_rgb_shape(rgb: np.ndarray, shape: tuple[int, int], mode: str = "bicubic"):
    rgb = np.asarray(rgb, dtype=np.float32)
    rgb = np.clip(rgb, 0.0, 1.0)

    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)

    if mode in {"bilinear", "bicubic"}:
        out = torchf.interpolate(tensor, size=shape, mode=mode, align_corners=False)
    else:
        out = torchf.interpolate(tensor, size=shape, mode=mode)

    out = out.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)

    return np.clip(out, 0.0, 1.0)


def resize_rgb(rgb: np.ndarray, size: int, mode: str = "bicubic"):
    return resize_rgb_shape(rgb, (size, size), mode)


def blur_2d(arr: np.ndarray, size: int, sigma: float):
    arr = np.asarray(arr, dtype=np.float32)

    coords = torch.arange(size, dtype=torch.float32) - size // 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")

    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, size, size)

    tensor = torch.from_numpy(arr).view(1, 1, arr.shape[0], arr.shape[1])
    pad = size // 2
    tensor = torchf.pad(tensor, (pad, pad, pad, pad), mode="reflect")

    out = torchf.conv2d(tensor, kernel)

    return out.squeeze().detach().cpu().numpy().astype(np.float32)


def smooth_2d(arr: np.ndarray):
    if not smooth_before_minmax:
        return arr.astype(np.float32)

    return blur_2d(arr, int(smooth_kernel_size), float(smooth_sigma))


def get_tile_tensor(batch: dict[str, Any], tile_index: int):
    sentinel = batch["sentinel"].detach().cpu().float()

    if sentinel.ndim == 6:
        return sentinel[:, :, tile_index]

    if sentinel.ndim == 5:
        return sentinel[:, tile_index]

    raise ValueError(f"bad sentinel shape: {tuple(sentinel.shape)}")


def get_signal_rgb(batch: dict[str, Any], tile_index: int):
    tile = get_tile_tensor(batch, tile_index)

    if tile.ndim != 5:
        raise ValueError(f"expected tile tensor 5d, got {tuple(tile.shape)}")

    if tile.shape[2] <= 16 and tile.shape[-1] > 16:
        signal = tile.abs().mean(dim=(0, 1, 2)).numpy().astype(np.float32)
        mean_tile = tile.mean(dim=(0, 1))

        if mean_tile.shape[0] >= 3:
            rgb = mean_tile[:3].permute(1, 2, 0).numpy().astype(np.float32)
        else:
            gray = mean_tile[0].numpy().astype(np.float32)
            rgb = np.stack([gray, gray, gray], axis=-1)

    elif tile.shape[-1] <= 16 and tile.shape[2] > 16:
        signal = tile.abs().mean(dim=(0, 1, 4)).numpy().astype(np.float32)
        mean_tile = tile.mean(dim=(0, 1))

        if mean_tile.shape[-1] >= 3:
            rgb = mean_tile[..., :3].numpy().astype(np.float32)
        else:
            gray = mean_tile[..., 0].numpy().astype(np.float32)
            rgb = np.stack([gray, gray, gray], axis=-1)

    else:
        raise ValueError(f"could not infer tile layout: {tuple(tile.shape)}")

    return signal, rgb


def get_valid_mask(signal: np.ndarray, rgb: np.ndarray):
    signal = np.nan_to_num(
        np.asarray(signal, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    rgb = np.nan_to_num(
        np.asarray(rgb, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if rgb.max() > 2.0:
        rgb = rgb / 255.0

    rgb = np.clip(rgb, 0.0, None)

    gray = (
        0.2989 * rgb[..., 0]
        + 0.5870 * rgb[..., 1]
        + 0.1140 * rgb[..., 2]
    ).astype(np.float32)

    positive_signal = signal[signal > 0]
    positive_gray = gray[gray > 0]

    if positive_signal.size == 0 or positive_gray.size == 0:
        return np.zeros_like(signal, dtype=np.float32)

    signal_ref = float(np.percentile(positive_signal, 95.0))
    signal_threshold = max(
        valid_mask_min_threshold,
        valid_mask_threshold_ratio * signal_ref,
        float(np.percentile(positive_signal, land_mask_signal_percentile)),
    )

    gray_threshold = float(np.percentile(positive_gray, land_mask_gray_percentile))

    mask = ((signal > signal_threshold) & (gray > gray_threshold)).astype(np.float32)

    if land_mask_blur_kernel_size is not None and land_mask_blur_kernel_size > 1:
        mask = blur_2d(mask, int(land_mask_blur_kernel_size), float(land_mask_blur_sigma))

    mask = np.clip(mask, 0.0, 1.0)

    if land_mask_hard_threshold is not None:
        mask = (mask > float(land_mask_hard_threshold)).astype(np.float32)

    if valid_mask_soft_power is not None and valid_mask_soft_power > 0:
        mask = np.power(mask, valid_mask_soft_power).astype(np.float32)

    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def normalize_rgb(rgb: np.ndarray, mask: np.ndarray):
    rgb = np.nan_to_num(
        np.asarray(rgb, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    mask = np.asarray(mask, dtype=np.float32)

    if rgb.max() > 2.0:
        rgb = rgb / 255.0

    rgb = np.clip(rgb, 0.0, None)

    valid = mask > 0.10
    out = np.zeros_like(rgb, dtype=np.float32)

    for channel_index in range(3):
        channel = rgb[..., channel_index]

        if valid.sum() >= 10:
            values = channel[valid]
            low = float(np.percentile(values, 2.0))
            high = float(np.percentile(values, 98.0))
        else:
            low = float(channel.min())
            high = float(channel.max())

        if high - low <= cam_eps:
            clean = np.zeros_like(channel, dtype=np.float32)
        else:
            clean = (channel - low) / (high - low)
            clean = np.clip(clean, 0.0, 1.0).astype(np.float32)

        out[..., channel_index] = clean

    out[~valid] = 0.0

    if background_gamma is not None and background_gamma > 0:
        out = np.power(out, background_gamma).astype(np.float32)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def get_backgrounds(batch: dict[str, Any], cam_shape: tuple[int, int]):
    backgrounds = []
    masks = []

    for tile_index in range(4):
        signal, rgb = get_signal_rgb(batch, tile_index)
        image_mask = get_valid_mask(signal, rgb)
        background = normalize_rgb(rgb, image_mask)

        cam_mask = resize_2d_shape(image_mask, cam_shape, mode="area")
        cam_mask = np.clip(cam_mask, 0.0, 1.0).astype(np.float32)

        backgrounds.append(background)
        masks.append(cam_mask)

    return (
        np.stack(backgrounds, axis=0).astype(np.float32),
        np.stack(masks, axis=0).astype(np.float32),
    )


def normalize_tile(tile_cam: np.ndarray, valid_mask: np.ndarray | None = None):
    tile_cam = np.nan_to_num(
        np.abs(np.asarray(tile_cam, dtype=np.float32)),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=np.float32)

        if valid_mask.shape != tile_cam.shape:
            valid_mask = resize_2d_shape(valid_mask, tile_cam.shape, mode="area")

        valid_mask = np.clip(valid_mask, 0.0, 1.0)
        valid_pixels = valid_mask > 0.10
    else:
        valid_mask = np.ones_like(tile_cam, dtype=np.float32)
        valid_pixels = np.ones_like(tile_cam, dtype=bool)

    if use_log_before_minmax:
        tile_cam = np.log1p(log_scale_factor * tile_cam)

    tile_cam = smooth_2d(tile_cam)

    if valid_pixels.sum() < 10:
        display = np.zeros_like(tile_cam, dtype=np.float32)
        low = 0.0
        high = 0.0
    else:
        values = tile_cam[valid_pixels]
        low = float(np.percentile(values, low_percentile))
        high = float(np.percentile(values, high_percentile))

        if high - low <= cam_eps:
            display = np.zeros_like(tile_cam, dtype=np.float32)
        else:
            display = ((tile_cam - low) / (high - low)).clip(0.0, 1.0).astype(np.float32)

            if display_gamma is not None and display_gamma > 0:
                display = np.power(display, display_gamma).astype(np.float32)

            display = display * valid_mask
            display[~valid_pixels] = 0.0

            if valid_pixels.any():
                valid_after = display[valid_pixels]

                if valid_after.max() - valid_after.min() > cam_eps:
                    display[valid_pixels] = (
                        (valid_after - valid_after.min())
                        / (valid_after.max() - valid_after.min())
                    )

            display[~valid_pixels] = 0.0
            display = np.clip(display, 0.0, 1.0).astype(np.float32)

    stats = {
        "pre_percentile_min": float(low),
        "pre_percentile_max": float(high),
        "display_min": float(display.min()),
        "display_max": float(display.max()),
        "display_mean": float(display.mean()),
        "display_std": float(display.std()),
        "valid_pixel_fraction": float(valid_pixels.mean()),
    }

    return display.astype(np.float32), stats


def normalize_tiles(tile_maps: np.ndarray, tile_masks: np.ndarray | None, tile_pixels: int):
    resized_maps = []
    resized_masks = []

    for tile_index in range(4):
        cam = resize_2d(tile_maps[tile_index], tile_pixels, mode="bicubic")

        if tile_masks is not None:
            mask = resize_2d(tile_masks[tile_index], tile_pixels, mode="area")
            mask = np.clip(mask, 0.0, 1.0)
        else:
            mask = np.ones_like(cam, dtype=np.float32)

        resized_maps.append(cam.astype(np.float32))
        resized_masks.append(mask.astype(np.float32))

    resized_maps = np.stack(resized_maps, axis=0).astype(np.float32)
    resized_masks = np.stack(resized_masks, axis=0).astype(np.float32)

    transformed = np.nan_to_num(
        np.abs(resized_maps),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if use_log_before_minmax:
        transformed = np.log1p(log_scale_factor * transformed)

    transformed = np.stack([smooth_2d(transformed[i]) for i in range(4)], axis=0).astype(np.float32)

    valid_pixels = resized_masks > 0.10

    if valid_pixels.sum() < 10:
        global_display = np.zeros_like(transformed, dtype=np.float32)
        global_low = 0.0
        global_high = 0.0
    else:
        values = transformed[valid_pixels]
        global_low = float(np.percentile(values, low_percentile))
        global_high = float(np.percentile(values, high_percentile))

        if global_high - global_low <= cam_eps:
            global_display = np.zeros_like(transformed, dtype=np.float32)
        else:
            global_display = (transformed - global_low) / (global_high - global_low)
            global_display = np.clip(global_display, 0.0, 1.0).astype(np.float32)

            if display_gamma is not None and display_gamma > 0:
                global_display = np.power(global_display, display_gamma).astype(np.float32)

            global_display = global_display * resized_masks
            global_display[~valid_pixels] = 0.0

    display_tiles = []
    stats = []

    for tile_index in range(4):
        tile_display, tile_stats = normalize_tile(
            resized_maps[tile_index],
            valid_mask=resized_masks[tile_index],
        )

        if tile_standardization_mode == "global":
            final = global_display[tile_index]
        elif tile_standardization_mode == "per_tile":
            final = tile_display
        else:
            final = (
                global_standardization_weight * global_display[tile_index]
                + per_tile_standardization_weight * tile_display
            )

        final = final * resized_masks[tile_index]
        final[resized_masks[tile_index] <= 0.10] = 0.0

        if final.max() > cam_eps:
            final = final / final.max()

        final = np.clip(final, 0.0, 1.0).astype(np.float32)

        tile_stats["standardization_mode"] = tile_standardization_mode
        tile_stats["global_standardization_weight"] = float(global_standardization_weight)
        tile_stats["per_tile_standardization_weight"] = float(per_tile_standardization_weight)
        tile_stats["global_pre_percentile_min"] = float(global_low)
        tile_stats["global_pre_percentile_max"] = float(global_high)
        tile_stats["mask_mean"] = float(resized_masks[tile_index].mean())

        display_tiles.append(final)
        stats.append(tile_stats)

    return join_tiles(display_tiles), display_tiles, stats


def map_to_rgb(display_map: np.ndarray):
    display_map = np.clip(display_map, 0.0, 1.0).astype(np.float32)
    rgba = cm.get_cmap(heatmap_cmap)(display_map)
    return (rgba[..., :3] * 255.0).clip(0, 255).astype(np.uint8)


def to_uint8(rgb: np.ndarray):
    return (np.clip(rgb, 0.0, 1.0) * 255.0).clip(0, 255).astype(np.uint8)


def make_gray(rgb: np.ndarray):
    rgb = np.asarray(rgb, dtype=np.float32)

    gray = (
        0.2989 * rgb[..., 0]
        + 0.5870 * rgb[..., 1]
        + 0.1140 * rgb[..., 2]
    )

    return np.stack([gray, gray, gray], axis=-1).astype(np.float32)


def blend(background_rgb: np.ndarray, heatmap_rgb: np.ndarray, alpha: float):
    background = background_rgb.astype(np.float32)
    heatmap = heatmap_rgb.astype(np.float32)
    out = (1.0 - alpha) * background + alpha * heatmap

    return np.clip(out, 0, 255).astype(np.uint8)


def join_tiles(tiles: list[np.ndarray]):
    height = tiles[0].shape[0]
    width = tiles[0].shape[1]

    if tiles[0].ndim == 2:
        out = np.zeros((height * 2, width * 2), dtype=np.float32)
    else:
        out = np.zeros((height * 2, width * 2, tiles[0].shape[-1]), dtype=tiles[0].dtype)

    out[0:height, 0:width] = tiles[0]
    out[0:height, width:2 * width] = tiles[1]
    out[height:2 * height, 0:width] = tiles[2]
    out[height:2 * height, width:2 * width] = tiles[3]

    return out


def get_font(size: int):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for font_path in paths:
        path = Path(font_path)

        if path.exists():
            return ImageFont.truetype(str(path), size)

    return ImageFont.load_default()


def draw_labels(image_array: np.ndarray, title: str | None = None, add_colorbar: bool = True):
    image = Image.fromarray(image_array).convert("RGB")
    draw = ImageDraw.Draw(image)

    width, height = image.size
    half_width = width // 2
    half_height = height // 2

    line_width = max(8, width // 900)
    label_font = get_font(max(48, width // 125))

    draw.line(
        [(half_width, 0), (half_width, height)],
        fill=(255, 255, 255),
        width=line_width,
    )
    draw.line(
        [(0, half_height), (width, half_height)],
        fill=(255, 255, 255),
        width=line_width,
    )

    margin = max(48, width // 170)
    pad_x = max(26, width // 360)
    pad_y = max(18, width // 480)

    labels = [
        ("tl / tile 0", half_width // 2, margin),
        ("tr / tile 1", half_width + half_width // 2, margin),
        ("bl / tile 2", half_width // 2, half_height + margin),
        ("br / tile 3", half_width + half_width // 2, half_height + margin),
    ]

    for text, center_x, top_y in labels:
        box = draw.textbbox((0, 0), text, font=label_font)
        text_width = box[2] - box[0]
        text_height = box[3] - box[1]

        x0 = int(center_x - text_width / 2 - pad_x)
        y0 = int(top_y)
        x1 = int(center_x + text_width / 2 + pad_x)
        y1 = int(top_y + text_height + 2 * pad_y)

        draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0))
        draw.text(
            (int(center_x - text_width / 2), int(y0 + pad_y)),
            text,
            font=label_font,
            fill=(255, 255, 255),
        )

    if not add_colorbar:
        return np.asarray(image, dtype=np.uint8)

    footer_height = max(150, width // 28)
    canvas = Image.new("RGB", (width, height + footer_height), (0, 0, 0))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)

    bar_width = int(width * 0.62)
    bar_height = max(32, width // 180)
    bar_x = (width - bar_width) // 2
    bar_y = height + max(22, footer_height // 5)

    gradient = np.tile(
        np.linspace(0.0, 1.0, bar_width, dtype=np.float32)[None, :],
        (bar_height, 1),
    )

    colorbar = Image.fromarray(map_to_rgb(gradient)).convert("RGB")
    canvas.paste(colorbar, (bar_x, bar_y))

    outline_width = max(2, width // 2500)
    draw.rectangle(
        (bar_x, bar_y, bar_x + bar_width, bar_y + bar_height),
        outline=(255, 255, 255),
        width=outline_width,
    )

    tick_font = get_font(max(28, width // 230))
    small_font = get_font(max(30, width // 215))

    tick_y = bar_y + bar_height + max(12, footer_height // 18)

    left_text = "0.0 lower relevance"
    mid_text = "0.5"
    right_text = "1.0 higher relevance"
    center_text = "relative attribution / model relevance scale"

    def draw_centered(text: str, x: int, y: int, font, fill=(255, 255, 255)):
        box = draw.textbbox((0, 0), text, font=font)
        draw.text((int(x - (box[2] - box[0]) / 2), y), text, font=font, fill=fill)

    draw.text((bar_x, tick_y), left_text, font=tick_font, fill=(255, 255, 255))
    draw_centered(mid_text, bar_x + bar_width // 2, tick_y, tick_font)

    right_box = draw.textbbox((0, 0), right_text, font=tick_font)
    draw.text(
        (bar_x + bar_width - (right_box[2] - right_box[0]), tick_y),
        right_text,
        font=tick_font,
        fill=(255, 255, 255),
    )

    center_y = tick_y + max(36, width // 180)
    draw_centered(center_text, width // 2, center_y, small_font)

    return np.asarray(canvas, dtype=np.uint8)


def save_rgb(path: Path, rgb: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def make_images(display_tiles: list[np.ndarray], background_tiles: np.ndarray, tile_pixels: int):
    heatmap_tiles = []
    background_tiles_uint8 = []
    overlay_tiles = []

    for tile_index in range(4):
        display = display_tiles[tile_index]
        heatmap_rgb = map_to_rgb(display)

        background = resize_rgb(background_tiles[tile_index], tile_pixels, mode="bicubic")

        if background_gray_overlay:
            background = make_gray(background)

        background_uint8 = to_uint8(background)
        overlay = blend(background_uint8, heatmap_rgb, overlay_alpha)

        heatmap_tiles.append(heatmap_rgb)
        background_tiles_uint8.append(background_uint8)
        overlay_tiles.append(overlay)

    return (
        join_tiles(heatmap_tiles),
        join_tiles(background_tiles_uint8),
        join_tiles(overlay_tiles),
    )


def safe_filename(text: str):
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
    out = "".join(ch if ch in allowed else "_" for ch in str(text)).strip("_")

    while "__" in out:
        out = out.replace("__", "_")

    return out or "sample"


def get_regency_flag():
    if hasattr(ft, "use_regency_embedding"):
        return bool(ft.use_regency_embedding)

    if hasattr(ft, "USE_REGENCY_EMBEDDING"):
        return bool(ft.USE_REGENCY_EMBEDDING)

    return True


def run_gradcam():
    output_dir.mkdir(parents=True, exist_ok=True)

    used_seeds = get_seeds()
    seed_total = len(used_seeds)

    all_tiles = []
    all_masks = []
    all_backgrounds = []
    manifest = []

    for seed_index, seed in enumerate(used_seeds, 1):
        set_seed(seed)

        cfg, train_loader, val_loader, regency_count = get_seed_data(seed)
        device = torch.device(cfg.device)

        batches = get_batches(val_loader, max_samples_per_seed)
        sample_total = len(batches)

        model = ft.build_model(
            cfg.repo_root,
            num_regencies=regency_count,
            use_regency_embedding=get_regency_flag(),
        )

        start_model(model, batches[0], device)
        load_checkpoint(model, get_checkpoint(seed))

        model.to(device)
        model.eval()

        for sample_index, batch in enumerate(batches, 1):
            sample_id = str(batch["sample_id"][0])

            tile_cams, pred_norm = get_gradcam(model, batch, device)

            background_tiles, tile_masks = get_backgrounds(
                batch,
                cam_shape=tile_cams.shape[-2:],
            )

            normalized_tiles = []

            for tile_index in range(4):
                display_small, _ = normalize_tile(
                    tile_cams[tile_index],
                    valid_mask=tile_masks[tile_index] if apply_sentinel_valid_mask else None,
                )
                normalized_tiles.append(display_small)

            normalized_tiles = np.stack(normalized_tiles, axis=0).astype(np.float32)

            all_tiles.append(normalized_tiles)
            all_masks.append(tile_masks.astype(np.float32))
            all_backgrounds.append(background_tiles.astype(np.float32))

            record = {
                "seed": int(seed),
                "sample_index": int(sample_index - 1),
                "sample_id": sample_id,
                "prediction_norm": float(pred_norm),
                "raw_tile_cam_stats": [
                    {
                        "tile": int(tile_index),
                        "min": float(tile_cams[tile_index].min()),
                        "max": float(tile_cams[tile_index].max()),
                        "mean": float(tile_cams[tile_index].mean()),
                        "std": float(tile_cams[tile_index].std()),
                    }
                    for tile_index in range(4)
                ],
            }

            manifest.append(record)

            if save_individual_sample_maps:
                sample_display_2x2, sample_display_tiles, sample_stats = normalize_tiles(
                    tile_cams,
                    tile_masks if apply_sentinel_valid_mask else None,
                    individual_tile_pixels,
                )

                sample_heatmap, sample_background, sample_overlay = make_images(
                    sample_display_tiles,
                    background_tiles,
                    individual_tile_pixels,
                )

                sample_dir = output_dir / f"seed_{seed}" / f"{sample_index - 1:03d}_{safe_filename(sample_id)}"
                sample_dir.mkdir(parents=True, exist_ok=True)

                np.save(sample_dir / "raw_tile_cams.npy", tile_cams.astype(np.float32))
                np.save(sample_dir / "display_equal_tile_gradcam.npy", sample_display_2x2.astype(np.float32))

                save_rgb(sample_dir / "heatmap_2x2.png", sample_heatmap)
                save_rgb(sample_dir / "sentinel_background_2x2.png", sample_background)
                save_rgb(sample_dir / "overlay_on_sentinel_2x2.png", sample_overlay)

                save_rgb(
                    sample_dir / "labelled_overlay_on_sentinel_2x2.png",
                    draw_labels(
                        sample_overlay,
                        title=f"{experiment_name} | seed {seed} | {sample_id}",
                    ),
                )

                with open(sample_dir / "stats.json", "w", encoding="utf-8") as file:
                    json.dump({**record, "display_tile_stats": sample_stats}, file, indent=2)

            show(f"sample {sample_index}/{sample_total} done")

        del model
        del train_loader
        del val_loader

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        show(f"seed {seed_index}/{seed_total} done")

    if not all_tiles:
        raise RuntimeError("no gradcam maps were computed")

    combined_tile_maps = np.stack(all_tiles, axis=0).mean(axis=0).astype(np.float32)
    combined_tile_masks = np.stack(all_masks, axis=0).mean(axis=0).astype(np.float32)
    combined_background_tiles = np.stack(all_backgrounds, axis=0).mean(axis=0).astype(np.float32)

    combined_display_2x2, combined_display_tiles, combined_stats = normalize_tiles(
        combined_tile_maps,
        combined_tile_masks if apply_sentinel_valid_mask else None,
        output_tile_pixels,
    )

    combined_heatmap, combined_background, combined_overlay = make_images(
        combined_display_tiles,
        combined_background_tiles,
        output_tile_pixels,
    )

    combined_dir = output_dir / "combined_all_selected_samples"
    combined_dir.mkdir(parents=True, exist_ok=True)

    np.save(combined_dir / "combined_raw_normalized_tile_maps.npy", combined_tile_maps.astype(np.float32))
    np.save(combined_dir / "combined_tile_masks.npy", combined_tile_masks.astype(np.float32))
    np.save(combined_dir / "combined_background_tiles.npy", combined_background_tiles.astype(np.float32))
    np.save(combined_dir / "combined_display_2x2.npy", combined_display_2x2.astype(np.float32))

    save_rgb(combined_dir / "combined_heatmap_2x2.png", combined_heatmap)
    save_rgb(combined_dir / "combined_sentinel_background_2x2.png", combined_background)
    save_rgb(combined_dir / "combined_overlay_on_sentinel_2x2.png", combined_overlay)

    save_rgb(
        combined_dir / "combined_labelled_heatmap_2x2.png",
        draw_labels(
            combined_heatmap,
            title=f"{experiment_name} | combined gradcam",
        ),
    )

    save_rgb(
        combined_dir / "combined_labelled_overlay_on_sentinel_2x2.png",
        draw_labels(
            combined_overlay,
            title=f"{experiment_name} | gradcam overlay on Sentinel-2",
        ),
    )

    metadata = {
        "experiment_name": experiment_name,
        "finetune_file": str(finetune_file),
        "checkpoint_root": str(checkpoint_root),
        "output_dir": str(output_dir),
        "target_layer_name": target_layer_name,
        "used_seeds": used_seeds,
        "max_samples_per_seed": max_samples_per_seed,
        "n_maps_combined": len(manifest),
        "display_settings": {
            "output_tile_pixels": output_tile_pixels,
            "individual_tile_pixels": individual_tile_pixels,
            "heatmap_cmap": heatmap_cmap,
            "overlay_alpha": overlay_alpha,
            "use_log_before_minmax": use_log_before_minmax,
            "log_scale_factor": log_scale_factor,
            "smooth_before_minmax": smooth_before_minmax,
            "smooth_kernel_size": smooth_kernel_size,
            "smooth_sigma": smooth_sigma,
            "robust_percentile_low": low_percentile,
            "robust_percentile_high": high_percentile,
            "display_gamma": display_gamma,
            "apply_sentinel_valid_mask": apply_sentinel_valid_mask,
            "valid_mask_threshold_ratio": valid_mask_threshold_ratio,
            "valid_mask_min_threshold": valid_mask_min_threshold,
            "valid_mask_soft_power": valid_mask_soft_power,
            "land_mask_gray_percentile": land_mask_gray_percentile,
            "land_mask_signal_percentile": land_mask_signal_percentile,
            "land_mask_blur_kernel_size": land_mask_blur_kernel_size,
            "land_mask_blur_sigma": land_mask_blur_sigma,
            "land_mask_hard_threshold": land_mask_hard_threshold,
            "background_gamma": background_gamma,
            "background_gray_overlay": background_gray_overlay,
            "tile_standardization_mode": tile_standardization_mode,
            "global_standardization_weight": global_standardization_weight,
            "per_tile_standardization_weight": per_tile_standardization_weight,
        },
        "combined_tile_stats": combined_stats,
        "manifest": manifest,
        "files": {
            "combined_heatmap": str(combined_dir / "combined_heatmap_2x2.png"),
            "combined_labelled_heatmap": str(combined_dir / "combined_labelled_heatmap_2x2.png"),
            "combined_sentinel_background": str(combined_dir / "combined_sentinel_background_2x2.png"),
            "combined_overlay_on_sentinel": str(combined_dir / "combined_overlay_on_sentinel_2x2.png"),
            "combined_labelled_overlay_on_sentinel": str(combined_dir / "combined_labelled_overlay_on_sentinel_2x2.png"),
            "combined_npy": str(combined_dir / "combined_raw_normalized_tile_maps.npy"),
        },
    }

    with open(output_dir / "gradcam_manifest.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    with open(combined_dir / "combined_stats.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


if __name__ == "__main__":
    run_gradcam()