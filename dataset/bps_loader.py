from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


regency_names = {
    "kabupaten bojonegoro": "bojonegoro",
    "kabupaten tuban": "tuban",
    "kabupaten lamongan": "lamongan",
    "kabupaten gresik": "gresik",
    "kabupaten sidoarjo": "sidoarjo",
    "kabupaten mojokerto": "mojokerto",
    "kabupaten jombang": "jombang",
    "kabupaten nganjuk": "nganjuk",
    "kabupaten madiun": "madiun",
    "kabupaten magetan": "magetan",
    "kabupaten ponorogo": "ponorogo",
    "kabupaten pacitan": "pacitan",
    "kabupaten trenggalek": "trenggalek",
    "kabupaten tulungagung": "tulungagung",
    "kabupaten blitar": "blitar",
    "kabupaten malang": "malang",
    "kabupaten lumajang": "lumajang",
    "kabupaten jember": "jember",
    "kabupaten bondowoso": "bondowoso",
    "kabupaten situbondo": "situbondo",
    "kabupaten banyuwangi": "banyuwangi",
    "kabupaten probolinggo": "probolinggo",
    "kabupaten pasuruan": "pasuruan",
    "kabupaten bangkalan": "bangkalan",
    "kabupaten sampang": "sampang",
    "kabupaten pamekasan": "pamekasan",
    "kabupaten sumenep": "sumenep",
    "kota surabaya": "surabaya",
    "kota malang": "malang_city",
    "kota batu": "batu",
}

csv_names = (
    "processed_bps.csv",
    "processed.csv",
    "rice.csv",
    "bps_rice.csv",
    "bps_processed.csv",
)


production_names = [
    "padi_production_tons",
    "padi_production",
    "padi",
    "produksi padi",
    "produksi_padi",
    "produksi padi (gkg)",
    "produksi padi gkg",
    "produksi padi - produksi padi (ton) (ton)",
    "produksi (ton)",
    "produksi",
    "production",
    "gkg",
    "gkg (ton)",
    "padi (gkg)",
    "padi gkg",
    "produksi padi sawah",
]


area_names = [
    "harvested_area_ha",
    "harvested_area",
    "luas panen",
    "luas_panen",
    "luas panen (ha)",
    "luas panen ha",
]


def clean_text(value):
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()
    return " ".join(text.split())


def to_number(series):
    text = series.astype(str).str.strip()
    text = text.str.replace("\u00a0", "", regex=False)
    text = text.str.replace(" ", "", regex=False)

    both = text.str.contains(",", regex=False) & text.str.contains(".", regex=False)
    converted = text.copy()
    converted[both] = converted[both].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    converted[~both] = converted[~both].str.replace(",", "", regex=False)

    return pd.to_numeric(converted, errors="coerce")


def merge_columns(df, names, new_name):
    found = [name for name in names if name in df.columns]

    if not found:
        raise KeyError(f"Missing column for {new_name}")

    data = None

    for name in found:
        current = df[name]
        data = current if data is None else data.fillna(current)

    df[new_name] = data
    return df


def find_csvs(root, year):
    folder = Path(root) / str(year)

    if not folder.is_dir():
        raise FileNotFoundError(f"Year folder not found: {folder}")

    files = sorted(path for path in folder.iterdir() if path.suffix.lower() == ".csv")

    if not files:
        raise FileNotFoundError(f"No csv file found: {folder}")

    preferred = [folder / name for name in csv_names if (folder / name).is_file()]

    return preferred if preferred else files


def clean_columns(df):
    df = df.copy()
    names = {}

    for column in df.columns:
        name = clean_text(column)

        if name in {"unnamed: 1", "regency", "kabupaten/kota", "kabupaten", "kota"}:
            names[column] = "regency_name"
        elif name in {"province", "provinsi"}:
            names[column] = "province"
        elif name in {"year", "tahun"}:
            names[column] = "year"
        elif name in production_names:
            names[column] = "padi_production_tons"
        elif name in {"beras", "beras_production", "beras_production_tons", "produksi beras", "produksi_beras", "produksi padi - produksi beras (ton) (ton)"}:
            names[column] = "beras_production_tons"
        elif name in area_names:
            names[column] = "harvested_area_ha"
        elif name in {"tl_pct", "tr_pct", "bl_pct", "br_pct", "total_tile_pct"}:
            names[column] = name

    if names:
        df = df.rename(columns=names)

    if "regency_name" not in df.columns and "Unnamed: 1" in df.columns:
        df = df.rename(columns={"Unnamed: 1": "regency_name"})

    return df


def clean_regency(name):
    text = clean_text(name)

    if text in regency_names:
        return regency_names[text]

    text = text.replace("kabupaten ", "").replace("kota ", "")
    text = text.replace("/", " ").replace("-", " ")

    return " ".join(text.split())


def check_columns(df, default_year=None, source_path=None):
    df = df.copy()

    if "province" not in df.columns:
        source_name = str(source_path).lower() if source_path is not None else ""

        if "yogyakarta" in source_name or "di_yogyakarta" in source_name:
            df["province"] = "di yogyakarta"
        elif "central_java" in source_name or "jawa_tengah" in source_name:
            df["province"] = "jawa tengah"
        elif "east_java" in source_name or "jawa_timur" in source_name:
            df["province"] = "jawa timur"
        else:
            raise KeyError("Missing column: province")

    if "regency_name" not in df.columns:
        raise KeyError("Missing column: regency_name")

    if "year" not in df.columns:
        if default_year is None:
            raise KeyError("Missing column: year")
        df["year"] = int(default_year)

    if "padi_production_tons" not in df.columns:
        df = merge_columns(df, production_names, "padi_production_tons")

    if "beras_production_tons" not in df.columns:
        df["beras_production_tons"] = np.nan

    if "harvested_area_ha" not in df.columns:
        df["harvested_area_ha"] = np.nan

    if "total_tile_pct" not in df.columns:
        df["total_tile_pct"] = 100.0

    for name in ["tl_pct", "tr_pct", "bl_pct", "br_pct"]:
        if name not in df.columns:
            df[name] = 0.0

    for name in [
        "padi_production_tons",
        "beras_production_tons",
        "harvested_area_ha",
        "total_tile_pct",
        "tl_pct",
        "tr_pct",
        "bl_pct",
        "br_pct",
    ]:
        df[name] = to_number(df[name])

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    return df


def make_targets(df):
    df = df.copy()

    df["province"] = df["province"].map(clean_text)
    df["regency"] = df["regency_name"].map(clean_regency)

    df["tile_weight"] = df["total_tile_pct"] / 100.0
    df["tile_weight"] = df["tile_weight"].clip(lower=0.0, upper=1.0)

    df["weighted_padi_production_tons"] = df["padi_production_tons"] * df["tile_weight"]
    df["weighted_beras_production_tons"] = df["beras_production_tons"] * df["tile_weight"]
    df["weighted_harvested_area_ha"] = df["harvested_area_ha"] * df["tile_weight"]

    data = (
        df.groupby(["province", "regency", "year"], as_index=False)
        .agg(
            padi_production_tons=("weighted_padi_production_tons", "sum"),
            beras_production_tons=("weighted_beras_production_tons", "sum"),
            harvested_area_ha=("weighted_harvested_area_ha", "sum"),
            total_tile_pct=("total_tile_pct", "sum"),
            n_rows=("regency", "size"),
        )
        .copy()
    )

    data["target_raw"] = data["padi_production_tons"]
    data["target"] = np.log1p(data["target_raw"])

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["year", "target", "target_raw"])
    data["year"] = data["year"].astype(int)
    data = data[data["target_raw"] > 0].reset_index(drop=True)

    return data


def load_data(root, years):
    tables = []

    for year in years:
        for path in find_csvs(root, int(year)):
            try:
                df = pd.read_csv(path)
                df = clean_columns(df)
                df = check_columns(df, default_year=int(year), source_path=path)
                tables.append(df)
            except KeyError:
                continue

    if not tables:
        raise ValueError("No production data loaded")

    df = pd.concat(tables, ignore_index=True)
    data = make_targets(df)

    if data.empty:
        raise ValueError("No valid production samples found")

    data["sample_id"] = (
        data["province"].astype(str)
        + "__"
        + data["regency"].astype(str)
        + "__"
        + data["year"].astype(str)
    )
    data["group_key"] = data["regency"].astype(str) + "__" + data["year"].astype(str)

    return data


@dataclass
class Scaler:
    mean: float
    std: float

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        return x * self.std + self.mean


def fit_scaler(train_df, target_col="target"):
    mean = float(train_df[target_col].mean())
    std = float(train_df[target_col].std())

    if not np.isfinite(std) or std < 1e-8:
        std = 1.0

    return Scaler(mean=mean, std=std)


def apply_scaler(df, scaler, target_col="target"):
    df = df.copy()
    df["target_norm"] = scaler.transform(df[target_col].to_numpy(dtype=np.float32))
    return df


class RiceDataset(Dataset):
    def __init__(self, samples_df):
        self.df = samples_df.reset_index(drop=True).copy()

        needed = {"province", "regency", "year", "sample_id", "target", "target_raw"}
        missing = needed - set(self.df.columns)

        if missing:
            raise KeyError(f"Missing columns: {sorted(missing)}")

        if "target_norm" not in self.df.columns:
            self.df["target_norm"] = self.df["target"].astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        return {
            "province": row["province"],
            "regency": row["regency"],
            "year": int(row["year"]),
            "sample_id": row["sample_id"],
            "target": np.float32(row["target"]),
            "target_norm": np.float32(row["target_norm"]),
            "target_raw": np.float32(row["target_raw"]),
            "padi_production_tons": np.float32(row["padi_production_tons"]),
            "beras_production_tons": np.float32(row["beras_production_tons"]),
            "harvested_area_ha": np.float32(row["harvested_area_ha"]),
        }


load_processed_bps_dataframe = load_data
TargetScaler = Scaler
fit_target_scaler = fit_scaler
apply_target_scaler = apply_scaler
BPS_Dataset = RiceDataset