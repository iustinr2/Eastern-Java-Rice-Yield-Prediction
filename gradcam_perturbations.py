from __future__ import annotations

import importlib.util
import json
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
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader


repo_root = Path("/vol/home/s3881946/Downloads/MMST-ViT-main")
gradcam_file = repo_root / "create_gradcam_map.py"
checkpoint_root = repo_root / "RESULTS/multiseed_finetune_results_temporal"
output_dir = repo_root / "perturbation_outputs_temporal"

experiment_name = "standard_random_grouped"
seed_values = [0, 1, 2, 3, 4, 5, 42, 123, 777, 2025]
max_samples_per_seed = 37
max_scan_batches = 500

perturbation_patch_size = 32
perturbation_stride = 32
perturbation_fill_mode = "tile_mean"
perturbation_score_mode = "absolute_change"

output_tile_pixels = 4096
individual_tile_pixels = 1536

heatmap_cmap = "turbo"
overlay_alpha = 0.42
visual_red_gamma = 0.75

use_log_before_minmax = True
log_scale_factor = 300.0

smooth_before_minmax = True
smooth_kernel_size = 3
smooth_sigma = 0.65

low_percentile = 2.0
high_percentile = 99.7
display_gamma = 1.25
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
rank_blend_weight = 0.45

save_individual_sample_maps = False


def show(text: str):
    print(text, flush=True)


def load_file(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")

    spec = importlib.util.spec_from_file_location(name, str(path))

    if spec is None or spec.loader is None:
        raise ImportError(f"could not import file: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


gc = load_file(gradcam_file, "gradcam_helpers")
ft = gc.ft

if hasattr(gc, "log"):
    gc.log = lambda *args, **kwargs: None

if hasattr(gc, "show"):
    gc.show = lambda *args, **kwargs: None


def use_func(*names):
    for name in names:
        if hasattr(gc, name):
            return getattr(gc, name)

    raise AttributeError(f"missing helper function: {names}")


get_checkpoint = use_func("get_checkpoint", "checkpoint_path")
get_seeds = use_func("get_seeds", "available_seeds")
load_checkpoint = use_func("load_checkpoint", "load_weights")
get_batches = use_func("get_batches", "collect_validation_batches")
copy_batch = use_func("copy_batch", "clone_batch")
get_regency_flag = use_func("get_regency_flag", "get_regency_embedding_flag")
start_model = use_func("start_model", "initialize_lazy_model")

resize_2d_shape = use_func("resize_2d_shape", "resize_2d_to_shape")
resize_2d = use_func("resize_2d")
resize_rgb = use_func("resize_rgb")
smooth_2d = use_func("smooth_2d")

get_tile_tensor = use_func("get_tile_tensor")
get_signal_rgb = use_func("get_signal_rgb", "tile_signal_and_rgb")
get_valid_mask = use_func("get_valid_mask", "valid_mask_from_signal_and_rgb")
normalize_rgb = use_func("normalize_rgb", "normalize_background_rgb")
get_backgrounds = use_func("get_backgrounds", "extract_background_and_masks")

to_uint8 = use_func("to_uint8", "rgb_float_to_uint8")
make_gray = use_func("make_gray", "make_gray_reference")
blend = use_func("blend", "blend_overlay")
join_tiles = use_func("join_tiles", "assemble_2x2")
save_rgb = use_func("save_rgb")
safe_filename = use_func("safe_filename")


def sync_gradcam_settings():
    values = {
        "repo_root": repo_root,
        "checkpoint_root": checkpoint_root,
        "output_dir": output_dir,
        "experiment_name": experiment_name,
        "seed_values": seed_values,
        "max_samples_per_seed": max_samples_per_seed,
        "max_scan_batches": max_scan_batches,
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
        "low_percentile": low_percentile,
        "high_percentile": high_percentile,
        "display_gamma": display_gamma,
        "cam_eps": cam_eps,
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
        "save_individual_sample_maps": save_individual_sample_maps,
    }

    for name, value in values.items():
        setattr(gc, name, value)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_seed_data(seed: int):
    helper = use_func("get_seed_data", "prepare_seed_objects")
    data = helper(seed)

    if len(data) == 4:
        cfg, train_loader, val_loader, regency_count = data
        return cfg, train_loader, val_loader, regency_count

    if len(data) == 6:
        cfg, train_loader, val_loader, scaler, baseline, regency_count = data
        return cfg, train_loader, val_loader, regency_count

    raise ValueError("bad seed data output")


def get_sentinel_layout(x: torch.Tensor):
    if x.ndim == 6:
        if x.shape[3] <= 16 and x.shape[-1] > 16:
            return "b_t_tile_c_h_w"

        if x.shape[-1] <= 16 and x.shape[3] > 16:
            return "b_t_tile_h_w_c"

    if x.ndim == 5:
        if x.shape[2] <= 16 and x.shape[-1] > 16:
            return "t_tile_c_h_w"

        if x.shape[-1] <= 16 and x.shape[2] > 16:
            return "t_tile_h_w_c"

    raise ValueError(f"could not infer sentinel layout: {tuple(x.shape)}")


def get_height_width(x: torch.Tensor):
    layout = get_sentinel_layout(x)

    if layout.endswith("c_h_w"):
        return int(x.shape[-2]), int(x.shape[-1])

    return int(x.shape[-3]), int(x.shape[-2])


def get_patch_fill(tile: torch.Tensor, patch: torch.Tensor, channel_last: bool):
    if perturbation_fill_mode == "zero":
        return torch.zeros_like(patch)

    if perturbation_fill_mode == "tile_mean":
        if channel_last:
            return tile.mean(dim=(-3, -2), keepdim=True)

        return tile.mean(dim=(-2, -1), keepdim=True)

    raise ValueError(f"bad perturbation fill mode: {perturbation_fill_mode}")


def perturb_patch(x: torch.Tensor, tile_index: int, y0: int, y1: int, x0: int, x1: int):
    out = x.clone()
    layout = get_sentinel_layout(out)

    if layout == "b_t_tile_c_h_w":
        tile = out[:, :, tile_index]
        patch = tile[:, :, :, y0:y1, x0:x1]
        out[:, :, tile_index, :, y0:y1, x0:x1] = get_patch_fill(tile, patch, False)

    elif layout == "b_t_tile_h_w_c":
        tile = out[:, :, tile_index]
        patch = tile[:, :, y0:y1, x0:x1, :]
        out[:, :, tile_index, y0:y1, x0:x1, :] = get_patch_fill(tile, patch, True)

    elif layout == "t_tile_c_h_w":
        tile = out[:, tile_index]
        patch = tile[:, :, y0:y1, x0:x1]
        out[:, tile_index, :, y0:y1, x0:x1] = get_patch_fill(tile, patch, False)

    elif layout == "t_tile_h_w_c":
        tile = out[:, tile_index]
        patch = tile[:, y0:y1, x0:x1, :]
        out[:, tile_index, y0:y1, x0:x1, :] = get_patch_fill(tile, patch, True)

    else:
        raise ValueError(f"bad sentinel layout: {layout}")

    return out


def get_score(model: nn.Module, batch: dict[str, Any]):
    with torch.no_grad():
        pred = ft.forward_mmst_model(model, batch)

    return float(pred.reshape(-1)[0].detach().cpu().item())


def get_delta(base_score: float, new_score: float):
    if perturbation_score_mode == "absolute_change":
        return abs(base_score - new_score)

    if perturbation_score_mode == "prediction_drop":
        return max(0.0, base_score - new_score)

    if perturbation_score_mode == "prediction_increase":
        return max(0.0, new_score - base_score)

    raise ValueError(f"bad perturbation score mode: {perturbation_score_mode}")


def get_perturbation(model: nn.Module, batch: dict[str, Any], device: torch.device):
    model.eval()

    device_batch = ft.move_batch_to_device(copy_batch(batch), device)

    if "sentinel" not in device_batch:
        raise KeyError("batch does not contain sentinel")

    sentinel = device_batch["sentinel"].detach()
    height, width = get_height_width(sentinel)
    base_score = get_score(model, device_batch)

    maps = np.zeros((4, height, width), dtype=np.float32)
    counts = np.zeros((4, height, width), dtype=np.float32)

    for tile_index in range(4):
        for y0 in range(0, height, perturbation_stride):
            y1 = min(y0 + perturbation_patch_size, height)

            for x0 in range(0, width, perturbation_stride):
                x1 = min(x0 + perturbation_patch_size, width)

                batch_copy = dict(device_batch)
                batch_copy["sentinel"] = perturb_patch(
                    sentinel,
                    tile_index,
                    y0,
                    y1,
                    x0,
                    x1,
                )

                delta = get_delta(base_score, get_score(model, batch_copy))

                maps[tile_index, y0:y1, x0:x1] += float(delta)
                counts[tile_index, y0:y1, x0:x1] += 1.0

    maps = maps / np.maximum(counts, 1.0)
    maps = np.nan_to_num(maps, nan=0.0, posinf=0.0, neginf=0.0)

    return maps.astype(np.float32), base_score


def transform_map(tile_map: np.ndarray):
    out = np.nan_to_num(
        np.abs(np.asarray(tile_map, dtype=np.float32)),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if use_log_before_minmax:
        out = np.log1p(log_scale_factor * out)

    return smooth_2d(out).astype(np.float32)


def rank_normalize(display: np.ndarray, valid_mask: np.ndarray):
    display = np.asarray(display, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=np.float32) > 0.10

    out = np.zeros_like(display, dtype=np.float32)

    if valid.sum() < 2:
        out[valid] = display[valid]
        return out

    values = display[valid]
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(values), endpoint=True, dtype=np.float32)
    out[valid] = ranks

    return out


def standardize_tiles(tile_maps: np.ndarray, tile_masks: np.ndarray | None):
    transformed = []
    masks = []

    for tile_index in range(4):
        tile = transform_map(tile_maps[tile_index])
        transformed.append(tile)

        if tile_masks is None:
            mask = np.ones_like(tile, dtype=np.float32)
        else:
            mask = np.asarray(tile_masks[tile_index], dtype=np.float32)

        if mask.shape != tile.shape:
            mask = resize_2d_shape(mask, tile.shape, mode="area")

        masks.append(np.clip(mask, 0.0, 1.0).astype(np.float32))

    transformed = np.stack(transformed, axis=0)
    masks = np.stack(masks, axis=0)

    valid = masks > 0.10
    stats = []

    if valid.sum() < 10:
        display_tiles = [np.zeros_like(transformed[0], dtype=np.float32) for _ in range(4)]
        return join_tiles(display_tiles), display_tiles, stats

    values = transformed[valid]
    low = float(np.percentile(values, low_percentile))
    high = float(np.percentile(values, high_percentile))

    display_tiles = []

    for tile_index in range(4):
        if high - low <= cam_eps:
            base = np.zeros_like(transformed[tile_index], dtype=np.float32)
        else:
            base = np.clip((transformed[tile_index] - low) / (high - low), 0.0, 1.0)

        if display_gamma is not None and display_gamma > 0:
            base = np.power(base, display_gamma).astype(np.float32)

        final = (
            (1.0 - rank_blend_weight) * base
            + rank_blend_weight * rank_normalize(base, masks[tile_index])
        )

        final = final * masks[tile_index]
        final[masks[tile_index] <= 0.10] = 0.0
        final = np.clip(final, 0.0, 1.0).astype(np.float32)

        display_tiles.append(final)

        stats.append(
            {
                "tile": int(tile_index),
                "global_pre_percentile_min": low,
                "global_pre_percentile_max": high,
                "display_min": float(final.min()),
                "display_max": float(final.max()),
                "display_mean": float(final.mean()),
                "display_std": float(final.std()),
                "mask_mean": float(masks[tile_index].mean()),
            }
        )

    return join_tiles(display_tiles), display_tiles, stats


def map_to_rgb(display_map: np.ndarray):
    out = np.clip(np.asarray(display_map, dtype=np.float32), 0.0, 1.0)
    out = np.power(out, float(visual_red_gamma))
    rgba = cm.get_cmap(heatmap_cmap)(out)

    return (rgba[..., :3] * 255.0).clip(0, 255).astype(np.uint8)


def draw_centered(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, font):
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(
        (int(x - (box[2] - box[0]) / 2), y),
        text,
        font=font,
        fill=(255, 255, 255),
    )


def get_font(size: int):
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for font_path in font_paths:
        path = Path(font_path)

        if path.exists():
            return ImageFont.truetype(str(path), size)

    return ImageFont.load_default()


def draw_labels(image_array: np.ndarray, add_colorbar: bool = True):
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

    canvas.paste(Image.fromarray(map_to_rgb(gradient)).convert("RGB"), (bar_x, bar_y))

    draw.rectangle(
        (bar_x, bar_y, bar_x + bar_width, bar_y + bar_height),
        outline=(255, 255, 255),
        width=max(2, width // 2500),
    )

    tick_font = get_font(max(28, width // 230))
    small_font = get_font(max(30, width // 215))
    tick_y = bar_y + bar_height + max(12, footer_height // 18)

    draw.text(
        (bar_x, tick_y),
        "0.0 lower perturbation effect",
        font=tick_font,
        fill=(255, 255, 255),
    )

    draw_centered(draw, "0.5", bar_x + bar_width // 2, tick_y, tick_font)

    right_text = "1.0 higher perturbation effect"
    box = draw.textbbox((0, 0), right_text, font=tick_font)

    draw.text(
        (bar_x + bar_width - (box[2] - box[0]), tick_y),
        right_text,
        font=tick_font,
        fill=(255, 255, 255),
    )

    draw_centered(
        draw,
        "relative occlusion sensitivity scale",
        width // 2,
        tick_y + max(36, width // 180),
        small_font,
    )

    return np.asarray(canvas, dtype=np.uint8)


def make_images(display_tiles: list[np.ndarray], background_tiles: np.ndarray, tile_pixels: int):
    heatmaps = []
    backgrounds = []
    overlays = []

    for tile_index in range(4):
        display = resize_2d(display_tiles[tile_index], tile_pixels, mode="nearest")
        heatmap = map_to_rgb(display)
        background = resize_rgb(background_tiles[tile_index], tile_pixels, mode="bicubic")

        if background_gray_overlay:
            background = make_gray(background)

        background_uint8 = to_uint8(background)
        overlay = blend(background_uint8, heatmap, overlay_alpha)

        heatmaps.append(heatmap)
        backgrounds.append(background_uint8)
        overlays.append(overlay)

    return join_tiles(heatmaps), join_tiles(backgrounds), join_tiles(overlays)


def run_perturbation():
    sync_gradcam_settings()
    output_dir.mkdir(parents=True, exist_ok=True)

    used_seeds = get_seeds()
    seed_total = len(used_seeds)

    all_maps = []
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

            tile_maps, base_score = get_perturbation(model, batch, device)

            background_tiles, tile_masks = get_backgrounds(
                batch,
                tile_maps.shape[-2:],
            )

            if not apply_sentinel_valid_mask:
                tile_masks = np.ones_like(tile_masks, dtype=np.float32)

            _, display_tiles, display_stats = standardize_tiles(tile_maps, tile_masks)

            all_maps.append(np.stack(display_tiles, axis=0).astype(np.float32))
            all_masks.append(tile_masks.astype(np.float32))
            all_backgrounds.append(background_tiles.astype(np.float32))

            record = {
                "seed": int(seed),
                "sample_index": int(sample_index - 1),
                "sample_id": sample_id,
                "base_prediction_norm": float(base_score),
                "perturbation_patch_size": int(perturbation_patch_size),
                "perturbation_stride": int(perturbation_stride),
                "perturbation_fill_mode": perturbation_fill_mode,
                "perturbation_score_mode": perturbation_score_mode,
                "raw_tile_perturbation_stats": [
                    {
                        "tile": int(tile_index),
                        "min": float(tile_maps[tile_index].min()),
                        "max": float(tile_maps[tile_index].max()),
                        "mean": float(tile_maps[tile_index].mean()),
                        "std": float(tile_maps[tile_index].std()),
                    }
                    for tile_index in range(4)
                ],
            }

            manifest.append(record)

            if save_individual_sample_maps:
                sample_heatmap, sample_background, sample_overlay = make_images(
                    display_tiles,
                    background_tiles,
                    individual_tile_pixels,
                )

                sample_dir = output_dir / f"seed_{seed}" / f"{sample_index - 1:03d}_{safe_filename(sample_id)}"
                sample_dir.mkdir(parents=True, exist_ok=True)

                np.save(
                    sample_dir / "raw_perturbation_tile_maps.npy",
                    tile_maps.astype(np.float32),
                )

                save_rgb(sample_dir / "heatmap_2x2.png", sample_heatmap)
                save_rgb(sample_dir / "sentinel_background_2x2.png", sample_background)
                save_rgb(sample_dir / "overlay_on_sentinel_2x2.png", sample_overlay)

                save_rgb(
                    sample_dir / "labelled_overlay_on_sentinel_2x2.png",
                    draw_labels(sample_overlay),
                )

                with open(sample_dir / "stats.json", "w", encoding="utf-8") as file:
                    json.dump({**record, "display_tile_stats": display_stats}, file, indent=2)

            show(f"sample {sample_index}/{sample_total} done")

        del model
        del train_loader
        del val_loader

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        show(f"seed {seed_index}/{seed_total} done")

    if not all_maps:
        raise RuntimeError("no perturbation maps were computed")

    combined_maps = np.stack(all_maps, axis=0).mean(axis=0).astype(np.float32)
    combined_masks = np.stack(all_masks, axis=0).mean(axis=0).astype(np.float32)
    combined_backgrounds = np.stack(all_backgrounds, axis=0).mean(axis=0).astype(np.float32)

    _, combined_display_tiles, combined_stats = standardize_tiles(
        combined_maps,
        combined_masks if apply_sentinel_valid_mask else None,
    )

    combined_heatmap, combined_background, combined_overlay = make_images(
        combined_display_tiles,
        combined_backgrounds,
        output_tile_pixels,
    )

    combined_dir = output_dir / "combined_all_selected_samples"
    combined_dir.mkdir(parents=True, exist_ok=True)

    np.save(
        combined_dir / "combined_raw_normalized_perturbation_tile_maps.npy",
        combined_maps.astype(np.float32),
    )
    np.save(
        combined_dir / "combined_tile_masks.npy",
        combined_masks.astype(np.float32),
    )
    np.save(
        combined_dir / "combined_background_tiles.npy",
        combined_backgrounds.astype(np.float32),
    )

    save_rgb(combined_dir / "combined_perturbation_heatmap_2x2.png", combined_heatmap)
    save_rgb(combined_dir / "combined_sentinel_background_2x2.png", combined_background)
    save_rgb(combined_dir / "combined_overlay_on_sentinel_2x2.png", combined_overlay)

    save_rgb(
        combined_dir / "combined_labelled_overlay_on_sentinel_2x2.png",
        draw_labels(combined_overlay),
    )

    metadata = {
        "experiment_name": experiment_name,
        "gradcam_file": str(gradcam_file),
        "checkpoint_root": str(checkpoint_root),
        "output_dir": str(output_dir),
        "used_seeds": used_seeds,
        "max_samples_per_seed": max_samples_per_seed,
        "n_maps_combined": len(manifest),
        "perturbation_patch_size": perturbation_patch_size,
        "perturbation_stride": perturbation_stride,
        "perturbation_fill_mode": perturbation_fill_mode,
        "perturbation_score_mode": perturbation_score_mode,
        "display_settings": {
            "output_tile_pixels": output_tile_pixels,
            "heatmap_cmap": heatmap_cmap,
            "overlay_alpha": overlay_alpha,
            "visual_red_gamma": visual_red_gamma,
            "use_log_before_minmax": use_log_before_minmax,
            "log_scale_factor": log_scale_factor,
            "smooth_before_minmax": smooth_before_minmax,
            "smooth_kernel_size": smooth_kernel_size,
            "smooth_sigma": smooth_sigma,
            "robust_percentile_low": low_percentile,
            "robust_percentile_high": high_percentile,
            "display_gamma": display_gamma,
            "rank_blend_weight": rank_blend_weight,
            "apply_sentinel_valid_mask": apply_sentinel_valid_mask,
        },
        "combined_tile_stats": combined_stats,
        "manifest": manifest,
        "files": {
            "combined_heatmap": str(combined_dir / "combined_perturbation_heatmap_2x2.png"),
            "combined_sentinel_background": str(combined_dir / "combined_sentinel_background_2x2.png"),
            "combined_overlay_on_sentinel": str(combined_dir / "combined_overlay_on_sentinel_2x2.png"),
            "combined_labelled_overlay_on_sentinel": str(combined_dir / "combined_labelled_overlay_on_sentinel_2x2.png"),
            "combined_npy": str(combined_dir / "combined_raw_normalized_perturbation_tile_maps.npy"),
        },
    }

    with open(output_dir / "perturbation_manifest.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    with open(combined_dir / "combined_stats.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


if __name__ == "__main__":
    run_perturbation()