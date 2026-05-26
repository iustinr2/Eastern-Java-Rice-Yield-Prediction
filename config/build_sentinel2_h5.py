import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


tile_order = ["tl", "tr", "bl", "br"]

input_dir = "/vol/home/s3881946/Downloads/Thesis_Sentinel2_Data"
output_h5 = "/vol/home/s3881946/Downloads/H5_Loader_Input/sentinel_input.h5"
output_manifest = "/vol/home/s3881946/Downloads/JSON_Loader_Input/sentinel_manifest.json"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=input_dir, help="Input folder")
    parser.add_argument("--output-h5", default=output_h5, help="Output H5 file")
    parser.add_argument("--output-manifest", default=output_manifest, help="Output JSON file")
    return parser.parse_args()


def get_number(name: str) -> int:
    text = name.strip().lower()

    if text.startswith("t"):
        text = text[1:]

    return int(text)


def get_dirs(root: Path):
    dirs = []

    for path in root.iterdir():
        if not path.is_dir():
            continue

        try:
            num = get_number(path.name)
            dirs.append((num, path))
        except Exception:
            pass

    dirs = sorted(dirs, key=lambda x: x[0])

    if len(dirs) != 24:
        print(f"Expected 24 folders, found {len(dirs)}")

    return dirs


def find_tiles(folder: Path):
    found = {}

    for path in folder.rglob("*.png"):
        name = path.stem.lower()

        for tile in tile_order:
            if name == tile or name.endswith(f"_{tile}") or name.startswith(f"{tile}_"):
                found[tile] = path

    missing = [tile for tile in tile_order if tile not in found]

    if missing:
        raise FileNotFoundError(f"Missing tiles {missing}: {folder}")

    return found


def load_tiles(files):
    imgs = []

    for tile in tile_order:
        img = Image.open(files[tile]).convert("RGB")
        imgs.append(np.array(img, dtype=np.uint8))

    return np.stack(imgs, axis=0)


def build_h5(in_dir: str, h5_out: str, json_out: str):
    root = Path(in_dir)
    dirs = get_dirs(root)

    h5_path = Path(h5_out)
    json_path = Path(json_out)

    h5_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    data = []

    with h5py.File(h5_path, "w") as hf:
        for num, folder in dirs:
            group = f"TF_{num:02d}"
            files = find_tiles(folder)
            arr = load_tiles(files)

            grp = hf.create_group(group)
            grp.create_dataset("image", data=arr, compression="gzip")
            grp.attrs["timeframe_index"] = num
            grp.attrs["source_dir"] = str(folder)

            data.append(
                {
                    "id": group,
                    "group": group,
                    "timeframe_index": num,
                    "file_path": str(h5_path.resolve()),
                    "source_dir": str(folder),
                }
            )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"H5 saved: {h5_path}")
    print(f"Json saved: {json_path}")


def main():
    args = get_args()
    build_h5(args.input_dir, args.output_h5, args.output_manifest)


if __name__ == "__main__":
    main()