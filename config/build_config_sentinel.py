import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import reproject


skip_bands = {0, 1, 3, 8, 9, 10, 11}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Input folder")
    parser.add_argument("--output-dir", required=True, help="Output folder")
    parser.add_argument(
        "--output-size",
        nargs=2,
        type=int,
        default=[224, 224],
        metavar=("width", "height"),
        help="Image size, default: 224 224",
    )
    return parser.parse_args()


def unzip_files(in_dir: Path):
    zips = list(in_dir.rglob("*.zip"))

    for zip_path in zips:
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = zf.namelist()
                roots = sorted(
                    {n.split("/")[0] for n in names if ".SAFE/" in n or n.endswith(".SAFE")}
                )

                if roots:
                    safe_path = zip_path.parent / roots[0]
                    if safe_path.exists():
                        continue

                zf.extractall(zip_path.parent)

        except zipfile.BadZipFile:
            print(f"Bad zip skipped: {zip_path}")

    return sorted(in_dir.rglob("*.SAFE"))


def find_file(safe_dir: Path, pattern: str):
    files = list(safe_dir.rglob(pattern))

    if not files:
        return None

    if len(files) > 1:
        files = sorted(files, key=lambda p: len(str(p)))

    return files[0]


def find_files(safe_dir: Path):
    b04 = find_file(safe_dir, "*_B04_10m.jp2")
    b08 = find_file(safe_dir, "*_B08_10m.jp2")
    scl = find_file(safe_dir, "*_SCL_20m.jp2")

    if not (b04 and b08 and scl):
        return None

    return {"b04": b04, "b08": b08, "scl": scl}


def get_name(safe_dir: Path):
    return safe_dir.name.replace(".SAFE", "")


def align_scl(scl_src, target):
    arr = np.zeros((target["height"], target["width"]), dtype=np.uint8)

    reproject(
        source=rasterio.band(scl_src, 1),
        destination=arr,
        src_transform=scl_src.transform,
        src_crs=scl_src.crs,
        dst_transform=target["transform"],
        dst_crs=target["crs"],
        dst_nodata=0,
        resampling=Resampling.nearest,
    )

    return arr


def get_ndvi(red, nir):
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)
    total = nir + red

    ndvi = np.full(red.shape, np.nan, dtype=np.float32)
    ok = total != 0
    ndvi[ok] = (nir[ok] - red[ok]) / total[ok]

    return ndvi


def get_mask(scl):
    return np.isin(scl, list(skip_bands))


def to_rgb(ndvi, ok):
    ndvi = np.clip(ndvi, -1.0, 1.0)
    arr = ((ndvi + 1.0) / 2.0) * 255.0
    arr = np.nan_to_num(arr, nan=0.0)
    arr[~ok] = 0
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    return np.stack([arr, arr, arr], axis=-1)


def resize_img(arr, size):
    if arr.ndim not in (2, 3):
        raise ValueError(f"Incorrect shape: {arr.shape}")

    width, height = size
    img = Image.fromarray(arr)
    img = img.resize((width, height), resample=Image.BILINEAR)

    return np.array(img)


def save_png(rgb, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out_path)


def process(safe_dir: Path, out_dir: Path, out_size):
    files = find_files(safe_dir)

    if files is None:
        print(f"Skipped missing files: {safe_dir.name}")
        return False

    name = get_name(safe_dir)
    out_path = out_dir / name / "ndvi.png"

    try:
        with rasterio.open(files["b04"]) as b04_src, rasterio.open(files["b08"]) as b08_src, rasterio.open(files["scl"]) as scl_src:
            red = b04_src.read(1).astype(np.float32)
            nir = b08_src.read(1).astype(np.float32)

            target = {
                "height": b04_src.height,
                "width": b04_src.width,
                "transform": b04_src.transform,
                "crs": b04_src.crs,
            }

            scl = align_scl(scl_src, target)

            if red.shape != nir.shape or red.shape != scl.shape:
                print(f"Incorrect shape mismatch: {safe_dir.name}")
                return False

            ndvi = get_ndvi(red, nir)
            mask = get_mask(scl)
            correct_mask = (~mask) & np.isfinite(ndvi) & (red > 0) & (nir > 0)

            rgb = to_rgb(ndvi, correct_mask)
            rgb = resize_img(rgb, tuple(out_size))

            save_png(rgb, out_path)

        return True

    except Exception as e:
        print(f"Failed: {safe_dir.name} - {e}")
        return False


def main():
    args = get_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    if not in_dir.exists():
        print(f"Input folder not found: {in_dir}")
        sys.exit(1)

    safe_dirs = unzip_files(in_dir)

    if not safe_dirs:
        print("No safe products found")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    good = 0
    bad = 0

    for i, safe_dir in enumerate(safe_dirs, 1):
        if process(safe_dir, out_dir, args.output_size):
            good += 1
        else:
            bad += 1

    print(f"Finished: {good} correct files, {bad} failed files")


if __name__ == "__main__":
    main()