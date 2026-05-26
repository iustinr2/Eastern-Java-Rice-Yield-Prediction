from pathlib import Path
import tempfile
import zipfile

import geopandas as gpd
import pandas as pd


# "east java", "central java", or "yogyakarta"
selected_province = "yogyakarta"

tiles_path = Path("/vol/home/s3881946/Downloads/tiles.geojson")
regencies_zip_path = Path("/vol/home/s3881946/Downloads/geoBoundaries-IDN-ADM2-all.zip")
output_root = Path("/vol/home/s3881946/Downloads/Processed_BPS_Data")
years = ["2021", "2022", "2023", "2024", "2025"]
tile_names = ["tl", "tr", "bl", "br"]

province_configs = {
    "east_java": {
        "bps_root": Path("/vol/home/s3881946/Downloads/BPS_Data"),
        "output_prefix": "east_java",
        "overlap_file": "east_java_tile_overlap_table.csv",
        "regency_columns": [
            "Kabupaten/Kota Se Jawa Timur",
            "kabupaten/kota se jawa timur",
            "Kabupaten/Kota",
            "kabupaten/kota",
            "kabupaten kota",
            "regency",
            "district",
            "kabupaten",
            "kabkot",
            "name",
            "NAME_2",
            "WADMKK",
        ],
    },
    "central_java": {
        "bps_root": Path("/vol/home/s3881946/Downloads/BPS_Data_CJ"),
        "output_prefix": "central_java",
        "overlap_file": "central_java_tile_overlap_table.csv",
        "regency_columns": [
            "District/City",
            "district/city",
            "district city",
            "Kabupaten/Kota",
            "kabupaten/kota",
            "kabupaten kota",
            "regency",
            "district",
            "kabupaten",
            "kabkot",
            "name",
            "NAME_2",
            "WADMKK",
        ],
    },
    "yogyakarta": {
        "bps_root": Path("/vol/home/s3881946/Downloads/BPS_Data_YK"),
        "output_prefix": "yogyakarta",
        "overlap_file": "yogyakarta_tile_overlap_table.csv",
        "regency_columns": [
            "Kabupaten/Kota Se Daerah Istimewa Yogyakarta",
            "kabupaten/kota se daerah istimewa yogyakarta",
            "Kabupaten/Kota Se DI Yogyakarta",
            "kabupaten/kota se di yogyakarta",
            "Kabupaten/Kota Se Yogyakarta",
            "kabupaten/kota se yogyakarta",
            "Kabupaten/Kota",
            "kabupaten/kota",
            "kabupaten kota",
            "regency",
            "district",
            "kabupaten",
            "kabkot",
            "name",
            "NAME_2",
            "WADMKK",
        ],
    },
}


def get_config():
    if selected_province not in province_configs:
        raise KeyError(f"Unknown selected_province: {selected_province}")

    return province_configs[selected_province]


def find_column(df, names, label):
    for name in names:
        if name in df.columns:
            return name

    lower = {str(col).strip().lower(): col for col in df.columns}

    for name in names:
        key = str(name).strip().lower()
        if key in lower:
            return lower[key]

    raise KeyError(f"Missing {label} column")


def clean_columns(df):
    df = df.copy()
    df.columns = [str(name).strip() for name in df.columns]
    return df


def clean_text(value):
    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def clean_regency(value):
    text = clean_text(value)

    if text.startswith("kabupaten "):
        text = text.replace("kabupaten ", "", 1)
    elif text.startswith("kota "):
        text = text.replace("kota ", "", 1) + "_city"

    text = text.replace("/", " ").replace("-", " ")
    return " ".join(text.split())


def is_area_only_csv(csv_path, df):
    text = clean_text(csv_path.name + " " + " ".join(str(c) for c in df.columns))
    has_area = any(word in text for word in ["luas panen", "luas_panen", "harvested area", "harvested_area", "panen"])
    has_production = any(word in text for word in ["produksi", "production", "gkg", "ton"])
    return has_area and not has_production


def read_regencies(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Regency zip not found: {path}")

    with tempfile.TemporaryDirectory() as folder:
        with zipfile.ZipFile(path, "r") as file:
            file.extractall(folder)

        geojson_path = Path(folder) / "geoBoundaries-IDN-ADM2.geojson"

        if not geojson_path.exists():
            raise FileNotFoundError(f"Geojson not found in zip: {path}")

        data = gpd.read_file(geojson_path)

    return data


def make_overlap(tiles, regencies):
    if "tile_id" not in tiles.columns:
        raise KeyError("Missing tile_id column")

    for name in ["shapeName", "geometry"]:
        if name not in regencies.columns:
            raise KeyError(f"Missing regency column: {name}")

    if tiles.crs is None:
        tiles = tiles.set_crs("EPSG:4326")

    if regencies.crs is None:
        regencies = regencies.set_crs("EPSG:4326")

    regencies = regencies.to_crs(tiles.crs)

    regencies = regencies[
        ["shapeName", "shapeISO", "shapeID", "shapeGroup", "shapeType", "geometry"]
    ].copy()
    regencies = regencies.rename(columns={"shapeName": "regency_name"})
    tiles = tiles[["tile_id", "geometry"]].copy()

    area_crs = "EPSG:23830"

    tiles_area = tiles.to_crs(area_crs)
    regencies_area = regencies.to_crs(area_crs)

    regencies_area["regency_area_m2"] = regencies_area.geometry.area

    overlaps = gpd.overlay(regencies_area, tiles_area, how="intersection")

    if overlaps.empty:
        raise ValueError("No tile and regency overlap found")

    overlaps["overlap_area_m2"] = overlaps.geometry.area
    overlaps["pct_of_regency"] = overlaps["overlap_area_m2"] / overlaps["regency_area_m2"] * 100.0

    data = regencies_area[["regency_name"]].copy()

    for tile in tile_names:
        data[f"{tile}_pct"] = 0.0

    for tile in tile_names:
        part = overlaps[overlaps["tile_id"] == tile][["regency_name", "pct_of_regency"]].copy()

        if part.empty:
            continue

        part = part.groupby("regency_name", as_index=False)["pct_of_regency"].sum()
        part = part.rename(columns={"pct_of_regency": f"{tile}_pct_new"})

        data = data.merge(part, on="regency_name", how="left")
        data[f"{tile}_pct"] = data[f"{tile}_pct_new"].fillna(data[f"{tile}_pct"])
        data = data.drop(columns=[f"{tile}_pct_new"])

    for name in ["tl_pct", "tr_pct", "bl_pct", "br_pct"]:
        data[name] = data[name].fillna(0.0).round(4)

    data["total_tile_pct"] = (
        data["tl_pct"] + data["tr_pct"] + data["bl_pct"] + data["br_pct"]
    ).round(4)

    data["regency_key"] = data["regency_name"].apply(clean_regency)

    return data


def process_years(overlap_data, config):
    rows = []
    bps_root = config["bps_root"]
    output_prefix = config["output_prefix"]
    regency_columns = config["regency_columns"]
    overlap_count = int(overlap_data[overlap_data["total_tile_pct"] > 0]["regency_key"].nunique())

    for year in years:
        year_dir = bps_root / year
        out_dir = output_root / year
        out_dir.mkdir(parents=True, exist_ok=True)

        if not year_dir.exists():
            continue

        csv_files = sorted(year_dir.glob("*.csv"))

        if not csv_files:
            continue

        year_rows = 0

        for csv_path in csv_files:
            bps = pd.read_csv(csv_path)
            bps = clean_columns(bps)

            if is_area_only_csv(csv_path, bps):
                continue

            regency_col = find_column(bps, regency_columns, "regency")

            bps = bps.copy()
            bps["regency_key"] = bps[regency_col].apply(clean_regency)

            merged = bps.merge(
                overlap_data[
                    [
                        "regency_key",
                        "regency_name",
                        "tl_pct",
                        "tr_pct",
                        "bl_pct",
                        "br_pct",
                        "total_tile_pct",
                    ]
                ],
                on="regency_key",
                how="left",
            )

            for name in ["tl_pct", "tr_pct", "bl_pct", "br_pct", "total_tile_pct"]:
                merged[name] = merged[name].fillna(0.0)

            filtered = merged[merged["total_tile_pct"] > 0].copy()
            filtered = filtered.drop_duplicates(subset=["regency_key"], keep="first")
            filtered = filtered.drop(columns=["regency_key"])

            output_csv = out_dir / f"{output_prefix}_{csv_path.name}"
            filtered.to_csv(output_csv, index=False)
            year_rows += len(filtered)

            rows.append(
                {
                    "province": selected_province,
                    "year": year,
                    "input_file": str(csv_path),
                    "output_file": str(output_csv),
                    "original_rows": len(bps),
                    "filtered_rows": len(filtered),
                }
            )

        if year_rows > 0:
            print(f"{year} csv data condensed to {len(filtered)} regency rows")

    return rows

def main():
    config = get_config()
    output_root.mkdir(parents=True, exist_ok=True)

    tiles = gpd.read_file(tiles_path)
    tiles = clean_columns(tiles)

    regencies = read_regencies(regencies_zip_path)
    regencies = clean_columns(regencies)

    overlap_data = make_overlap(tiles, regencies)

    overlap_path = output_root / config["overlap_file"]
    overlap_data.to_csv(overlap_path, index=False)

    rows = process_years(overlap_data, config)

    summary = pd.DataFrame(rows)
    summary_path = output_root / f"{config['output_prefix']}_processing_summary.csv"
    summary.to_csv(summary_path, index=False)


if __name__ == "__main__":
    main()