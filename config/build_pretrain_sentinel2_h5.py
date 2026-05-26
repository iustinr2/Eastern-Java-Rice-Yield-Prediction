# build_sentinel2_h5.py

import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


tile_order = ["tl", "tr", "bl", "br"]

input_dir = "/vol/home/s3881946/Downloads/Processed_Pretrain_Data"
output_h5 = "/vol/home/s3881946/Downloads/H5_Loader_Input/sentinel_pretrain.h5"
output_manifest = "/vol/home/s3881946/Downloads/JSON_Loader_Input/sentinel_pretrain_manifest.json"

expected_timeframe_count = 29


def normalize_timeframe_name(name: str) -> int:
    value = name.strip().upper()

    if value.startswith("TF_"):
        value = value[3:]
    elif value.startswith("TF"):
        value = value[2:]
    elif value.startswith("T"):
        value = value[1:]

    return int(value)


def get_timeframe_dirs(root_dir: Path):
    timeframe_entries = []

    for candidate_path in root_dir.iterdir():
        if not candidate_path.is_dir():
            continue

        try:
            timeframe_index = normalize_timeframe_name(candidate_path.name)
            timeframe_entries.append((timeframe_index, candidate_path))
        except Exception:
            continue

    timeframe_entries = sorted(timeframe_entries, key=lambda item: item[0])

    if len(timeframe_entries) != expected_timeframe_count:
        print(
            f"Expected {expected_timeframe_count} timeframe folders instead of {len(timeframe_entries)}."
        )

    detected_indices = [idx for idx, _ in timeframe_entries]
    expected_indices = list(range(1, expected_timeframe_count + 1))
    missing_indices = sorted(set(expected_indices) - set(detected_indices))

    if missing_indices:
        print(f"Missing timeframe indices: {missing_indices}")

    return timeframe_entries


def find_png_in_tile_folder(tile_dir: Path):
    if not tile_dir.exists() or not tile_dir.is_dir():
        raise FileNotFoundError(f"Missing tile folder: {tile_dir}")

    png_candidates = sorted(tile_dir.rglob("*.png"))

    if len(png_candidates) == 0:
        raise FileNotFoundError(f"No PNG found inside tile folder: {tile_dir}")

    for candidate_path in png_candidates:
        if candidate_path.name.lower() == "ndvi.png":
            return candidate_path

    if len(png_candidates) > 1:
        print( f"Warning: multiple PNG files found in {tile_dir}")

    return png_candidates[0]


def find_tile_files(timeframe_dir: Path):
    tile_image_paths = {}

    for tile_name in tile_order:
        tile_dir = timeframe_dir / tile_name
        tile_image_paths[tile_name] = find_png_in_tile_folder(tile_dir)

    return tile_image_paths


def load_and_stack_tiles(tile_image_paths):
    tile_arrays = []
    reference_shape = None

    for tile_name in tile_order:
        image_path = tile_image_paths[tile_name]

        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image, dtype=np.uint8)

        if reference_shape is None:
            reference_shape = image_array.shape
        elif image_array.shape != reference_shape:
            raise ValueError(
                f"Tile shape mismatch. Expected {reference_shape} instead of {image_array.shape} for {image_path}"
            )

        tile_arrays.append(image_array)

    tile_stack = np.stack(tile_arrays, axis=0)

    return tile_stack


def build_h5(input_dir_path: str, output_h5_path: str, output_manifest_path: str):
    input_root = Path(input_dir_path)

    if not input_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")

    timeframe_entries = get_timeframe_dirs(input_root)

    if not timeframe_entries:
        raise ValueError(f"No timeframe folders found under: {input_root}")

    manifest_records = []

    h5_path = Path(output_h5_path)
    manifest_path = Path(output_manifest_path)

    h5_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if h5_path.exists():
        print(f"Warning: removing existing H5 file: {h5_path}")
        h5_path.unlink()

    successful_timeframe_count = 0
    failed_timeframe_count = 0

    with h5py.File(h5_path, "w") as h5_file:
        for timeframe_index, timeframe_dir in timeframe_entries:
            group_name = f"TF_{timeframe_index:02d}"

            try:
                tile_image_paths = find_tile_files(timeframe_dir)

                tile_stack = load_and_stack_tiles(tile_image_paths)

                group = h5_file.create_group(group_name)
                group.create_dataset(
                    "image",
                    data=tile_stack,
                    compression="gzip",
                )

                group.attrs["timeframe_index"] = int(timeframe_index)
                group.attrs["source_dir"] = str(timeframe_dir)
                group.attrs["tile_order"] = ",".join(tile_order)

                for tile_name in tile_order:
                    group.attrs[f"source_{tile_name}"] = str(tile_image_paths[tile_name])

                manifest_records.append(
                    {
                        "id": group_name,
                        "group": group_name,
                        "timeframe_index": int(timeframe_index),
                        "file_path": str(h5_path.resolve()),
                        "source_dir": str(timeframe_dir),
                        "tile_order": tile_order,
                        "tiles": {
                            tile_name: str(tile_image_paths[tile_name])
                            for tile_name in tile_order
                        },
                        "image_shape": list(tile_stack.shape),
                        "dtype": str(tile_stack.dtype),
                    }
                )

                successful_timeframe_count += 1

            except Exception as exc:
                failed_timeframe_count += 1
                print(f"Failed to build {group_name}: {exc}")

    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest_records, manifest_file, indent=2)

    print(f"H5 output path {h5_path}")
    print(f"Manifest output path {manifest_path}")

    print(f"Successful timeframe groups {successful_timeframe_count}")
    print(f"Failed timeframe groups {failed_timeframe_count}")


if __name__ == "__main__":
    build_h5(input_dir, output_h5, output_manifest)