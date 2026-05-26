from __future__ import annotations

import contextlib
import copy
import io
import inspect
import json
import os
import random
import re
import time
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as f
import xarray as xr
from torch.utils.data import DataLoader, Dataset

experiment_name = "standard_random_grouped"
split_mode = "standard"
use_regency_embedding = False
output_root = Path("/vol/home/s3881946/Downloads/MMST-ViT-main/multiseed_finetune_results_temporal")

experiment_name = "standard_random_grouped"
holdout_year = 2025
spatial_split_seed = 42
seeds = [0, 1, 2, 3, 4, 5, 42, 123, 777, 2025]
default_years = (2021, 2022, 2023, 2024, 2025)

def env_bool(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default).strip().lower()
    return value in {"1", "true", "yes", "y", "on"}

def quiet_call(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*args, **kwargs)

overwrite_existing = env_bool("mmst_overwrite_existing", "0")
run_rf_baseline = env_bool("mmst_run_rf_baseline", "1")
run_cnn_baseline = env_bool("mmst_run_cnn_baseline", "1")
baseline_cnn_image_size = int(os.environ.get("mmst_baseline_cnn_image_size", "64"))
baseline_cnn_epochs = int(os.environ.get("mmst_baseline_cnn_epochs", "35"))
baseline_cnn_batch_size = int(os.environ.get("mmst_baseline_cnn_batch_size", "8"))
baseline_cnn_patience = int(os.environ.get("mmst_baseline_cnn_patience", "8"))

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def normalize_text(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = " ".join(s.split())
    return s

def normalize_regency_name(name: object) -> str:
    s = normalize_text(name)
    s = s.replace("kabupaten ", "").replace("kota ", "")
    s = s.replace("/", " ").replace("-", " ")
    s = " ".join(s.split())
    return s

def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def find_year_csvs(root: str, year: int) -> list[str]:
    year_dir = os.path.join(root, str(year))
    if not os.path.isdir(year_dir):
        raise FileNotFoundError(f"Processed BPS year folder not found: {year_dir}")
    csvs = [os.path.join(year_dir, f) for f in os.listdir(year_dir) if f.lower().endswith(".csv")]
    csvs.sort()
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {year_dir}")
    return csvs

def clean_legacy_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    renamed = {}
    for c in df.columns:
        lc = normalize_text(c)
        if lc in {"unnamed: 1", "regency", "kabupaten/kota", "kabupaten", "kota"}:
            renamed[c] = "regency_name"
        elif lc in {"province", "provinsi"}:
            renamed[c] = "province"
        elif lc in {"year", "tahun"}:
            renamed[c] = "year"
        elif lc in {"padi", "padi_production", "padi_production_tons", "produksi padi", "produksi_padi"}:
            renamed[c] = "padi_production_tons"
        elif lc in {"beras", "beras_production", "beras_production_tons", "produksi beras", "produksi_beras"}:
            renamed[c] = "beras_production_tons"
        elif lc in {"luas panen", "luas_panen", "harvested_area", "harvested_area_ha", "luas panen (ha)"}:
            renamed[c] = "harvested_area_ha"
        elif lc in {"tl_pct", "tr_pct", "bl_pct", "br_pct", "total_tile_pct"}:
            renamed[c] = lc
    if renamed:
        df = df.rename(columns=renamed)
    if "regency_name" not in df.columns and "Unnamed: 1" in df.columns:
        df = df.rename(columns={"Unnamed: 1": "regency_name"})
    return df

def get_series(df: pd.DataFrame, col_name: str) -> pd.Series:
    obj = df[col_name]
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]
    return obj

def ensure_required_fields(df: pd.DataFrame, default_year: Optional[int] = None, source_path: Optional[str] = None) -> pd.DataFrame:
    df = df.copy()
    source_name = os.path.basename(source_path).lower() if source_path is not None else ""

    def lower_map(frame: pd.DataFrame) -> dict:
        return {str(c).strip().lower(): c for c in frame.columns}

    def find_first_column(frame: pd.DataFrame, candidates: list[str]) -> Optional[str]:
        lm = lower_map(frame)
        for cand in candidates:
            if cand in frame.columns:
                return cand
            if cand.lower() in lm:
                return lm[cand.lower()]
        return None

    def find_by_keywords(frame: pd.DataFrame, include_any: list[str], include_all: Optional[list[str]] = None, exclude_any: Optional[list[str]] = None) -> Optional[str]:
        include_all = include_all or []
        exclude_any = exclude_any or []
        for c in frame.columns:
            lc = str(c).strip().lower()
            if include_any and not any(tok in lc for tok in include_any):
                continue
            if include_all and not all(tok in lc for tok in include_all):
                continue
            if exclude_any and any(tok in lc for tok in exclude_any):
                continue
            return c
        return None

    def find_numeric_candidate(frame: pd.DataFrame, preferred_keywords: list[str]) -> Optional[str]:
        numeric_cols = []
        for c in frame.columns:
            try:
                s = pd.to_numeric(get_series(frame, c), errors="coerce")
                if s.notna().sum() > max(3, len(frame) // 4):
                    numeric_cols.append(c)
            except Exception:
                pass
        for c in numeric_cols:
            lc = str(c).strip().lower()
            if any(tok in lc for tok in preferred_keywords):
                return c
        return None

    province_col = find_first_column(df, ["province", "provinsi"])
    if province_col is not None:
        df["province"] = get_series(df, province_col)
    else:
        inferred_province = None
        if "yogyakarta" in source_name or "di yogyakarta" in source_name:
            inferred_province = "di yogyakarta"
        elif "jawa tengah" in source_name or "central java" in source_name:
            inferred_province = "jawa tengah"
        elif "jawa timur" in source_name or "east java" in source_name or "gkg" in source_name:
            inferred_province = "jawa timur"
        if inferred_province is None:
            raise KeyError(f"Missing required column 'province' and could not infer province from filename: {source_path}")
        df["province"] = inferred_province

    regency_col = find_first_column(
        df,
        [
            "regency_name", "kabupaten/kota", "kabupaten_kota", "kabupaten kota",
            "nama kabupaten/kota", "nama kabupaten kota", "kabupaten", "kota",
            "wilayah", "region", "Unnamed: 1", "unnamed: 1", "Kabupaten/Kota Se Jawa Timur",
        ],
    )
    if regency_col is None:
        regency_col = find_by_keywords(df, include_any=["kabupaten", "kota", "regency", "wilayah", "region"])
    if regency_col is None:
        raise KeyError(f"Missing required column 'regency_name' in {source_path}. Available columns: {list(df.columns)}")
    df["regency_name"] = get_series(df, regency_col)

    year_col = find_first_column(df, ["year", "tahun"])
    if year_col is not None:
        df["year"] = get_series(df, year_col)
    else:
        if default_year is None:
            raise KeyError(f"Missing required column 'year' in {source_path}")
        df["year"] = default_year

    production_col = find_first_column(
        df,
        [
            "padi_production_tons", "padi_production", "padi", "produksi padi", "produksi_padi",
            "produksi padi (gkg)", "produksi padi gkg", "produksi padi - produksi padi (ton) (ton)",
            "produksi (ton)", "produksi", "production", "gkg", "gkg (ton)", "padi (gkg)",
            "padi gkg", "produksi padi sawah",
        ],
    )
    if production_col is None:
        production_col = find_by_keywords(df, include_any=["produksi", "production", "gkg", "ton"], include_all=["padi"], exclude_any=["beras", "luas", "panen"])
    if production_col is None:
        production_col = find_by_keywords(df, include_any=["produksi", "production", "gkg", "ton"], exclude_any=["beras", "luas", "panen", "tile", "pct", "year", "tahun"])
    if production_col is None:
        production_col = find_numeric_candidate(df, ["produksi", "production", "gkg", "ton", "padi"])
    if production_col is None:
        raise KeyError(f"Missing required padi production column in {source_path}. Available columns: {list(df.columns)}")
    df["padi_production_tons"] = get_series(df, production_col)

    area_col = find_first_column(
        df,
        ["harvested_area_ha", "harvested_area", "luas panen", "luas_panen", "luas panen (ha)", "luas panen ha", "panen", "area harvested", "luas tanam"],
    )
    if area_col is None:
        area_col = find_by_keywords(df, include_any=["luas", "harvest", "area", "panen"], include_all=["panen"], exclude_any=["produksi", "beras", "tile", "pct"])
    if area_col is None:
        area_col = find_by_keywords(df, include_any=["luas", "harvest", "area"], exclude_any=["produksi", "beras", "tile", "pct"])
    df["harvested_area_ha"] = get_series(df, area_col) if area_col is not None else np.nan

    beras_col = find_first_column(
        df,
        ["beras_production_tons", "beras_production", "beras", "produksi beras", "produksi_beras", "produksi padi - produksi beras (ton) (ton)"],
    )
    if beras_col is None:
        beras_col = find_by_keywords(df, include_any=["beras"], exclude_any=["luas", "panen", "tile", "pct"])
    df["beras_production_tons"] = get_series(df, beras_col) if beras_col is not None else np.nan

    if "total_tile_pct" not in df.columns:
        total_tile_col = find_first_column(df, ["total_tile_pct"])
        df["total_tile_pct"] = get_series(df, total_tile_col) if total_tile_col is not None else 100.0

    for c in ["tl_pct", "tr_pct", "bl_pct", "br_pct"]:
        if c not in df.columns:
            found = find_first_column(df, [c])
            df[c] = get_series(df, found) if found is not None else 0.0

    for c in ["padi_production_tons", "beras_production_tons", "harvested_area_ha", "total_tile_pct", "tl_pct", "tr_pct", "bl_pct", "br_pct"]:
        df[c] = safe_numeric(get_series(df, c))

    df["year"] = pd.to_numeric(get_series(df, "year"), errors="coerce").astype("Int64")
    df["province"] = get_series(df, "province").map(normalize_text)
    df["regency"] = get_series(df, "regency_name").map(normalize_regency_name)
    return df

def load_processed_bps_tile_rows(processed_root: str, years: Iterable[int]) -> pd.DataFrame:
    def make_columns_unique(df: pd.DataFrame) -> pd.DataFrame:
        counts = {}
        new_cols = []
        for col in df.columns:
            col_str = str(col)
            if col_str not in counts:
                counts[col_str] = 0
                new_cols.append(col_str)
            else:
                counts[col_str] += 1
                new_cols.append(f"{col_str}__dup{counts[col_str]}")
        df = df.copy()
        df.columns = new_cols
        return df

    parts = []
    for year in years:
        csv_paths = find_year_csvs(processed_root, int(year))
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            df = make_columns_unique(df)
            df = clean_legacy_columns(df)
            df = make_columns_unique(df)
            df = ensure_required_fields(df, default_year=int(year), source_path=csv_path)
            df = make_columns_unique(df)
            parts.append(df)

    if not parts:
        raise ValueError("No processed BPS dataframes were loaded")

    full = pd.concat(parts, ignore_index=True, sort=False)
    full = full.dropna(subset=["year"]).copy()
    full["year"] = full["year"].astype(int)

    weights = full[["tl_pct", "tr_pct", "bl_pct", "br_pct"]].fillna(0.0).to_numpy(dtype=np.float64)
    sums = weights.sum(axis=1, keepdims=True)
    sums[sums <= 0] = 1.0
    weights = weights / sums

    full["tile_w_tl"] = weights[:, 0]
    full["tile_w_tr"] = weights[:, 1]
    full["tile_w_bl"] = weights[:, 2]
    full["tile_w_br"] = weights[:, 3]

    agg = (
        full.groupby(["province", "regency", "year"], as_index=False)
        .agg(
            padi_production_tons=("padi_production_tons", "mean"),
            beras_production_tons=("beras_production_tons", "mean"),
            harvested_area_ha=("harvested_area_ha", "mean"),
            total_tile_pct=("total_tile_pct", "mean"),
            tile_w_tl=("tile_w_tl", "mean"),
            tile_w_tr=("tile_w_tr", "mean"),
            tile_w_bl=("tile_w_bl", "mean"),
            tile_w_br=("tile_w_br", "mean"),
        )
        .copy()
    )
    agg = agg.replace([np.inf, -np.inf], np.nan)
    agg = agg.dropna(subset=["padi_production_tons"]).copy()

    has_area = agg["harvested_area_ha"].notna() & (agg["harvested_area_ha"] > 0)
    agg["target_raw"] = np.where(has_area, agg["padi_production_tons"] / agg["harvested_area_ha"], agg["padi_production_tons"])
    agg["target"] = np.log1p(agg["target_raw"])
    agg = agg.replace([np.inf, -np.inf], np.nan)
    agg = agg.dropna(subset=["target", "target_raw"]).reset_index(drop=True)
    agg["sample_id"] = agg["province"].astype(str) + "__" + agg["regency"].astype(str) + "__" + agg["year"].astype(str)
    agg["group_key"] = agg["regency"].astype(str) + "__" + agg["year"].astype(str)

    return agg

def grouped_train_val_split(df: pd.DataFrame, group_col: str = "group_key", val_ratio: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    groups = df[group_col].astype(str).unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_val = max(1, int(round(len(groups) * val_ratio)))
    if n_val >= len(groups):
        n_val = len(groups) - 1
    val_groups = set(groups[:n_val])
    train_df = df[~df[group_col].astype(str).isin(val_groups)].reset_index(drop=True)
    val_df = df[df[group_col].astype(str).isin(val_groups)].reset_index(drop=True)
    if train_df.empty or val_df.empty:
        raise ValueError("Grouped split produced an empty train or val set")
    return train_df, val_df

class TargetScaler:
    def __init__(self, mean: float, std: float):
        self.mean = float(mean)
        self.std = float(std) if float(std) >= 1e-8 else 1.0

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

def fit_target_scaler(train_df: pd.DataFrame, target_col: str = "target") -> TargetScaler:
    return TargetScaler(train_df[target_col].mean(), train_df[target_col].std())

def apply_target_scaler(df: pd.DataFrame, scaler: TargetScaler, target_col: str = "target") -> pd.DataFrame:
    df = df.copy()
    df["target_norm"] = scaler.transform(df[target_col].to_numpy(dtype=np.float32))
    return df

def to_tensor(x: Any) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.float()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return torch.tensor(x, dtype=torch.float32)

def fix_sentinel_layout(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 5:
        raise ValueError(f"Sentinel sample must be 5D, got shape {tuple(x.shape)}")
    if x.shape[2] <= 32 and x.shape[-1] > 32:
        return x.contiguous()
    return x.permute(0, 1, 4, 2, 3).contiguous()

def fix_weather_layout(x: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    if x.ndim == 5:
        return x
    if x.ndim == 4:
        return x.unsqueeze(2)
    if x.ndim == 3:
        return x.unsqueeze(2)
    if x.ndim == 2:
        return x.unsqueeze(1).unsqueeze(1)
    if x.ndim == 1:
        return x.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    raise ValueError(f"Unsupported weather tensor shape: {tuple(x.shape)}")

def build_sentinel_index(processed_bps_root: str, years: Iterable[int], sentinel_year_lookup: Dict[int, Any]) -> Dict[Tuple[str, int], torch.Tensor]:
    rows = load_processed_bps_tile_rows(processed_bps_root, years)
    if rows.empty:
        raise ValueError("load_processed_bps_tile_rows returned no rows")
    available_years = sorted(int(y) for y in sentinel_year_lookup.keys())
    out: Dict[Tuple[str, int], torch.Tensor] = {}
    missing_years = {}
    matched = 0
    for _, row in rows.iterrows():
        regency = normalize_regency_name(row["regency"])
        year = int(row["year"])
        key = (regency, year)
        if year not in sentinel_year_lookup:
            missing_years[year] = missing_years.get(year, 0) + 1
            continue
        seq = fix_sentinel_layout(to_tensor(sentinel_year_lookup[year])).clone()
        if seq.shape[1] != 4:
            raise ValueError(f"Expected 4 Sentinel tiles, got shape {tuple(seq.shape)}")
        weights = torch.tensor(
            [float(row["tile_w_tl"]), float(row["tile_w_tr"]), float(row["tile_w_bl"]), float(row["tile_w_br"])],
            dtype=seq.dtype,
            device=seq.device,
        ).view(1, 4, 1, 1, 1)
        out[key] = seq * weights
        matched += 1
    if missing_years:
        pass
    if not out:
        sample_years = sorted(rows["year"].dropna().astype(int).unique().tolist())
        sample_regencies = rows["regency"].astype(str).head(10).tolist()
        raise ValueError(f"Sentinel index is empty. BPS years={sample_years}, Sentinel years={available_years}, example regencies={sample_regencies}")
    return out

def build_weather_index(processed_bps_root: str, years: Iterable[int], weather_root: str) -> Dict[Tuple[str, int], Dict[str, Optional[torch.Tensor]]]:
    rows = load_processed_bps_tile_rows(processed_bps_root, years)

    def extract_t_id(name: str) -> Optional[int]:
        m = re.fullmatch(r"T(\d+)", name)
        return int(m.group(1)) if m else None

    def summarize_array(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        flat = arr.reshape(-1)
        return np.array([float(np.mean(flat)), float(np.std(flat)), float(np.min(flat)), float(np.max(flat))], dtype=np.float32)

    def load_timeframe_feature_vector(t_dir: str) -> tuple[np.ndarray, np.ndarray]:
        month_dirs = []
        for name in sorted(os.listdir(t_dir)):
            p = os.path.join(t_dir, name)
            if os.path.isdir(p):
                month_dirs.append(p)
        if not month_dirs:
            raise ValueError(f"No monthly folders found under {t_dir}")
        monthly_feature_vectors = []
        for month_dir in month_dirs:
            nc_files = [os.path.join(month_dir, f) for f in os.listdir(month_dir) if f.lower().endswith(".nc")]
            nc_files.sort()
            if not nc_files:
                continue
            file_feature_vectors = []
            for nc_path in nc_files:
                ds = xr.open_dataset(nc_path)
                try:
                    var_features = []
                    for var_name in ds.data_vars:
                        values = ds[var_name].values
                        var_features.append(summarize_array(values))
                    if not var_features:
                        continue
                    file_vec = np.concatenate(var_features, axis=0)
                    file_feature_vectors.append(file_vec)
                finally:
                    ds.close()
            if file_feature_vectors:
                month_vec = np.stack(file_feature_vectors, axis=0).mean(axis=0)
                monthly_feature_vectors.append(month_vec)
        if not monthly_feature_vectors:
            raise ValueError(f"No usable NetCDF data found under {t_dir}")
        monthly_feature_vectors = np.stack(monthly_feature_vectors, axis=0)
        ys = monthly_feature_vectors.astype(np.float32)
        yl = monthly_feature_vectors.mean(axis=0, keepdims=True).astype(np.float32)
        return ys, yl

    timeframe_entries = []
    for name in sorted(os.listdir(weather_root)):
        t_id = extract_t_id(name)
        if t_id is None:
            continue
        t_dir = os.path.join(weather_root, name)
        if not os.path.isdir(t_dir):
            continue
        ys, yl = load_timeframe_feature_vector(t_dir)
        timeframe_entries.append((t_id, torch.from_numpy(ys).float(), torch.from_numpy(yl).float()))
    timeframe_entries.sort(key=lambda x: x[0])
    if not timeframe_entries:
        raise ValueError(f"No weather timeframe folders parsed under {weather_root}")
    year_list = list(years)
    if len(timeframe_entries) < len(year_list):
        raise ValueError(f"Only {len(timeframe_entries)} weather timeframe samples were parsed, but need at least {len(year_list)}")
    chunks = np.array_split(np.arange(len(timeframe_entries)), len(year_list))
    weather_year_lookup = {}
    for year, idxs in zip(year_list, chunks):
        ys_list = [timeframe_entries[int(i)][1] for i in idxs]
        yl_list = [timeframe_entries[int(i)][2] for i in idxs]
        ys_shapes = [tuple(x.shape) for x in ys_list]
        yl_shapes = [tuple(x.shape) for x in yl_list]
        if len(set(ys_shapes)) != 1:
            raise ValueError(f"Inconsistent ys shapes for year {year}: {ys_shapes}")
        if len(set(yl_shapes)) != 1:
            raise ValueError(f"Inconsistent yl shapes for year {year}: {yl_shapes}")
        ys_year = torch.stack(ys_list, dim=0).mean(dim=0)
        yl_year = torch.stack(yl_list, dim=0).mean(dim=0)
        weather_year_lookup[int(year)] = {"ys": ys_year, "yl": yl_year}
    weather_index: Dict[Tuple[str, int], Dict[str, Optional[torch.Tensor]]] = {}
    for _, row in rows.iterrows():
        year = int(row["year"])
        regency = normalize_regency_name(row["regency"])
        key = (regency, year)
        if year not in weather_year_lookup:
            continue
        entry = weather_year_lookup[year]
        weather_index[key] = {"ys": entry["ys"].clone(), "yl": entry["yl"].clone()}
    if not weather_index:
        raise ValueError("Weather index is empty")
    return weather_index

class RiceFineTuneDataset(Dataset):
    def __init__(self, samples_df: pd.DataFrame, sentinel_index: Dict[Tuple[str, int], Any], weather_index: Dict[Tuple[str, int], Any]):
        self.df = samples_df.reset_index(drop=True).copy()
        self.sentinel_index = sentinel_index
        self.weather_index = weather_index

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        regency = normalize_regency_name(row["regency"])
        year = int(row["year"])
        key = (regency, year)
        if key not in self.sentinel_index:
            raise KeyError(f"Missing sentinel sample for key={key}")
        if key not in self.weather_index:
            raise KeyError(f"Missing weather sample for key={key}")
        sentinel = fix_sentinel_layout(to_tensor(self.sentinel_index[key]))
        weather_item = self.weather_index[key]
        ys = fix_weather_layout(weather_item["ys"])
        yl = fix_weather_layout(weather_item["yl"]) if weather_item["yl"] is not None else None
        aux_cont = torch.tensor(
            [
                float(row["tile_w_tl"]), float(row["tile_w_tr"]), float(row["tile_w_bl"]), float(row["tile_w_br"]),
                float(row["total_tile_pct"]), float(row["year_norm_aux"]),
            ],
            dtype=torch.float32,
        )
        return {
            "sample_id": row["sample_id"],
            "province": row["province"],
            "regency": regency,
            "regency_idx": torch.tensor(int(row["regency_idx"]), dtype=torch.long),
            "province_idx": torch.tensor(int(row["province_idx"]), dtype=torch.long),
            "year": year,
            "sentinel": sentinel,
            "ys": ys,
            "yl": yl,
            "aux_cont": aux_cont,
            "target": torch.tensor(float(row["target"]), dtype=torch.float32),
            "target_norm": torch.tensor(float(row["target_norm"]), dtype=torch.float32),
            "target_raw": torch.tensor(float(row["target_raw"]), dtype=torch.float32),
        }

def collate_mmst_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    sentinel = torch.stack([b["sentinel"] for b in batch], dim=0)
    ys = torch.stack([b["ys"] for b in batch], dim=0)
    has_yl = all(b["yl"] is not None for b in batch)
    yl = torch.stack([b["yl"] for b in batch], dim=0) if has_yl else None
    return {
        "sample_id": [b["sample_id"] for b in batch],
        "province": [b["province"] for b in batch],
        "regency": [b["regency"] for b in batch],
        "regency_idx": torch.stack([b["regency_idx"] for b in batch], dim=0),
        "province_idx": torch.stack([b["province_idx"] for b in batch], dim=0),
        "year": torch.tensor([b["year"] for b in batch], dtype=torch.long),
        "sentinel": sentinel,
        "ys": ys,
        "yl": yl,
        "aux_cont": torch.stack([b["aux_cont"] for b in batch], dim=0),
        "target": torch.stack([b["target"] for b in batch], dim=0),
        "target_norm": torch.stack([b["target_norm"] for b in batch], dim=0),
        "target_raw": torch.stack([b["target_raw"] for b in batch], dim=0),
    }

def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)

def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) <= 1e-12 or np.std(y_pred) <= 1e-12:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])

def regression_report(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
    return {f"{prefix}rmse": rmse(y_true, y_pred), f"{prefix}mae": mae(y_true, y_pred), f"{prefix}r2": r2_score_np(y_true, y_pred), f"{prefix}pcc": pearson_corr(y_true, y_pred)}

def extract_prediction_tensor(model_output: Any) -> torch.Tensor:
    if isinstance(model_output, dict):
        for key in ("pred", "preds", "logits", "output"):
            if key in model_output:
                model_output = model_output[key]
                break
        else:
            raise KeyError("Model output dict missing one of: pred, preds, logits, output")
    if not isinstance(model_output, torch.Tensor):
        raise TypeError(f"Unsupported model output type: {type(model_output)}")
    if model_output.ndim == 1:
        return model_output
    if model_output.ndim == 2 and model_output.shape[1] == 1:
        return model_output.squeeze(1)
    if model_output.ndim == 2 and model_output.shape[1] >= 2:
        return model_output[:, 0]
    raise ValueError(f"Unsupported prediction shape: {tuple(model_output.shape)}")

def forward_mmst_model(model: nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
    sig = inspect.signature(model.forward)
    param_names = [p.name for p in sig.parameters.values() if p.name != "self"]
    candidate_kwargs = {"x": batch["sentinel"], "ys": batch["ys"], "yl": batch["yl"], "regency_idx": batch.get("regency_idx"), "province_idx": batch.get("province_idx"), "aux_cont": batch.get("aux_cont")}
    filtered = {k: v for k, v in candidate_kwargs.items() if k in param_names}
    out = model(**filtered)
    return extract_prediction_tensor(out)

def tensor_stats(name: str, x: torch.Tensor) -> str:
    if x is None:
        return f"{name}: None"
    x_det = x.detach()
    finite = torch.isfinite(x_det)
    finite_ratio = float(finite.float().mean().item()) if x_det.numel() > 0 else 1.0
    if finite.any():
        vals = x_det[finite]
        return f"{name}: shape={tuple(x_det.shape)} mean={float(vals.mean().item()):.6f} std={float(vals.std().item()) if vals.numel() > 1 else 0.0:.6f} min={float(vals.min().item()):.6f} max={float(vals.max().item()):.6f} finite_ratio={finite_ratio:.6f}"
    return f"{name}: shape={tuple(x_det.shape)} finite_ratio=0.000000"

def assert_finite_tensor(name: str, x: torch.Tensor) -> None:
    if x is None:
        return
    if not torch.isfinite(x).all():
        raise ValueError(f"Non-finite tensor detected -> {tensor_stats(name, x)}")

def debug_batch_finiteness(batch: Dict[str, Any]) -> None:
    for k in ["sentinel", "ys", "yl", "aux_cont", "target", "target_norm", "target_raw"]:
        if k in batch and isinstance(batch[k], torch.Tensor):
            assert_finite_tensor(k, batch[k])

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, amp_enabled: bool = False, grad_clip: float = 1.0, scaler: Optional[torch.cuda.amp.GradScaler] = None, accumulation_steps: int = 4) -> float:
    model.train()
    criterion = nn.SmoothL1Loss(beta=0.5)
    total_loss = 0.0
    n = 0
    optimizer.zero_grad(set_to_none=True)
    optimizer_steps = 0
    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        debug_batch_finiteness(batch)
        if amp_enabled:
            with torch.cuda.amp.autocast():
                pred_norm = forward_mmst_model(model, batch)
                assert_finite_tensor("pred_norm", pred_norm)
                loss = criterion(pred_norm, batch["target_norm"])
                assert_finite_tensor("loss_before_div", loss)
                loss = loss / accumulation_steps
                assert_finite_tensor("loss_after_div", loss)
            scaler.scale(loss).backward()
            if step % accumulation_steps == 0 or step == len(loader):
                if grad_clip is not None and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                prev_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                new_scale = scaler.get_scale()
                if new_scale > prev_scale:
                    optimizer_steps += 1
                optimizer.zero_grad(set_to_none=True)
        else:
            pred_norm = forward_mmst_model(model, batch)
            assert_finite_tensor("pred_norm", pred_norm)
            loss = criterion(pred_norm, batch["target_norm"])
            assert_finite_tensor("loss_before_div", loss)
            loss = loss / accumulation_steps
            assert_finite_tensor("loss_after_div", loss)
            loss.backward()
            if step % accumulation_steps == 0 or step == len(loader):
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
        bs = batch["target_norm"].shape[0]
        total_loss += float(loss.item()) * bs * accumulation_steps
        n += bs
    epoch_loss = total_loss / max(n, 1)
    if not np.isfinite(epoch_loss):
        raise ValueError(f"Epoch loss is non-finite: {epoch_loss}")
    return epoch_loss

@torch.no_grad()
def predict_table(model: nn.Module, loader: DataLoader, device: torch.device, target_mean: float, target_std: float) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        source_batch = batch
        batch = move_batch_to_device(batch, device)
        pred_norm = forward_mmst_model(model, batch)
        pred_log = pred_norm.detach().cpu().numpy() * target_std + target_mean
        pred_raw = np.expm1(pred_log).astype(np.float64)
        true_raw = batch["target_raw"].detach().cpu().numpy().astype(np.float64)
        for i, sample_id in enumerate(source_batch["sample_id"]):
            rows.append({
                "sample_id": str(sample_id),
                "province": str(source_batch["province"][i]),
                "regency": str(source_batch["regency"][i]),
                "year": int(source_batch["year"][i].item()),
                "true_raw": float(true_raw[i]),
                "pred_raw": float(pred_raw[i]),
                "signed_error": float(pred_raw[i] - true_raw[i]),
                "abs_error": float(abs(pred_raw[i] - true_raw[i])),
            })
    return pd.DataFrame(rows)

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, target_mean: float, target_std: float) -> Dict[str, float]:
    table = predict_table(model, loader, device, target_mean, target_std)
    return regression_report(table["true_raw"].to_numpy(dtype=np.float64), table["pred_raw"].to_numpy(dtype=np.float64), prefix="raw_")

def mean_baseline_metrics(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, float]:
    mean_log = float(train_df["target"].mean())
    y_raw = val_df["target_raw"].to_numpy(dtype=np.float64)
    pred_raw = np.full_like(y_raw, float(np.expm1(mean_log)), dtype=np.float64)
    return regression_report(y_raw, pred_raw, prefix="raw_")

def baseline_key(row: pd.Series) -> Tuple[str, int]:
    return normalize_regency_name(row["regency"]), int(row["year"])

def tensor_feature_stats(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float().cpu()
    if x.ndim == 5:
        values = x.permute(2, 0, 1, 3, 4).reshape(x.shape[2], -1)
        stats = torch.cat(
            [
                values.mean(dim=1),
                values.std(dim=1),
                values.min(dim=1).values,
                values.max(dim=1).values,
            ],
            dim=0,
        )
        return stats.numpy().astype(np.float32)
    flat = x.reshape(-1)
    return torch.tensor(
        [float(flat.mean()), float(flat.std()), float(flat.min()), float(flat.max())],
        dtype=torch.float32,
    ).numpy().astype(np.float32)

def flat_weather_features(x: Optional[torch.Tensor]) -> np.ndarray:
    if x is None:
        return np.zeros(4, dtype=np.float32)
    x = fix_weather_layout(x).detach().float().cpu()
    flat = x.reshape(-1)
    stats = torch.tensor(
        [float(flat.mean()), float(flat.std()), float(flat.min()), float(flat.max())],
        dtype=torch.float32,
    )
    return np.concatenate([flat.numpy().astype(np.float32), stats.numpy().astype(np.float32)], axis=0)

def baseline_feature_vector(row: pd.Series, sentinel_index: Dict[Tuple[str, int], Any], weather_index: Dict[Tuple[str, int], Any]) -> np.ndarray:
    key = baseline_key(row)
    sentinel = fix_sentinel_layout(to_tensor(sentinel_index[key]))
    weather = weather_index[key]
    aux = np.array(
        [
            float(row["tile_w_tl"]),
            float(row["tile_w_tr"]),
            float(row["tile_w_bl"]),
            float(row["tile_w_br"]),
            float(row["total_tile_pct"]),
            float(row["year_norm_aux"]),
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            aux,
            tensor_feature_stats(sentinel),
            flat_weather_features(weather.get("ys")),
            flat_weather_features(weather.get("yl")),
        ],
        axis=0,
    ).astype(np.float32)

def build_baseline_features(df: pd.DataFrame, sentinel_index: Dict[Tuple[str, int], Any], weather_index: Dict[Tuple[str, int], Any]) -> np.ndarray:
    return np.stack([baseline_feature_vector(row, sentinel_index, weather_index) for _, row in df.iterrows()], axis=0).astype(np.float32)

def random_forest_baseline_metrics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    sentinel_index: Dict[Tuple[str, int], Any],
    weather_index: Dict[Tuple[str, int], Any],
    seed: int,
) -> Dict[str, float]:
    from sklearn.ensemble import RandomForestRegressor

    x_train = build_baseline_features(train_df, sentinel_index, weather_index)
    x_val = build_baseline_features(val_df, sentinel_index, weather_index)
    y_train = train_df["target"].to_numpy(dtype=np.float64)
    y_val_raw = val_df["target_raw"].to_numpy(dtype=np.float64)
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=int(seed),
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    pred_log = model.predict(x_val).astype(np.float64)
    pred_raw = np.expm1(pred_log)
    return regression_report(y_val_raw, pred_raw, prefix="raw_")

class BaselineSentinelDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        sentinel_index: Dict[Tuple[str, int], Any],
        target_mean: float,
        target_std: float,
        image_size: int,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.sentinel_index = sentinel_index
        self.target_mean = float(target_mean)
        self.target_std = float(target_std) if float(target_std) >= 1e-8 else 1.0
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        x = fix_sentinel_layout(to_tensor(self.sentinel_index[baseline_key(row)]))
        t, g, c, h, w = x.shape
        x = x.reshape(t * g * c, h, w).float() / 255.0
        if self.image_size > 0 and (h != self.image_size or w != self.image_size):
            x = f.interpolate(x.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False).squeeze(0)
        y_log = float(row["target"])
        y_norm = (y_log - self.target_mean) / self.target_std
        y_raw = float(row["target_raw"])
        return x, torch.tensor(y_norm, dtype=torch.float32), torch.tensor(y_raw, dtype=torch.float32)

class BaselineCNNRegressor(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)

@torch.no_grad()
def evaluate_cnn_baseline(model: nn.Module, loader: DataLoader, device: torch.device, target_mean: float, target_std: float) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    y_all = []
    pred_all = []
    for x, y_norm, y_raw in loader:
        x = x.to(device, non_blocking=True)
        y_norm = y_norm.to(device, non_blocking=True)
        pred_norm = model(x)
        losses.append(float(f.smooth_l1_loss(pred_norm, y_norm).detach().cpu().item()))
        pred_log = pred_norm.detach().cpu().numpy().astype(np.float64) * float(target_std) + float(target_mean)
        pred_all.append(np.expm1(pred_log))
        y_all.append(y_raw.detach().cpu().numpy().astype(np.float64))
    return float(np.mean(losses)), np.concatenate(y_all), np.concatenate(pred_all)

def cnn_sentinel_baseline_metrics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    sentinel_index: Dict[Tuple[str, int], Any],
    cfg: FineTuneConfig,
) -> Dict[str, float]:
    device = torch.device(cfg.device)
    target_mean = float(train_df["target"].mean())
    target_std = float(train_df["target"].std())
    if target_std < 1e-8:
        target_std = 1.0
    train_ds = BaselineSentinelDataset(train_df, sentinel_index, target_mean, target_std, baseline_cnn_image_size)
    val_ds = BaselineSentinelDataset(val_df, sentinel_index, target_mean, target_std, baseline_cnn_image_size)
    first_x, _, _ = train_ds[0]
    model = BaselineCNNRegressor(int(first_x.shape[0])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_loader = DataLoader(train_ds, batch_size=baseline_cnn_batch_size, shuffle=True, num_workers=0, pin_memory=cfg.device.startswith("cuda"), drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=baseline_cnn_batch_size, shuffle=False, num_workers=0, pin_memory=cfg.device.startswith("cuda"), drop_last=False)
    best_state = None
    best_loss = np.inf
    bad_epochs = 0
    for _ in range(baseline_cnn_epochs):
        model.train()
        for x, y_norm, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y_norm = y_norm.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = f.smooth_l1_loss(model(x), y_norm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_loss, _, _ = evaluate_cnn_baseline(model, val_loader, device, target_mean, target_std)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= baseline_cnn_patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    _, y_raw, pred_raw = evaluate_cnn_baseline(model, val_loader, device, target_mean, target_std)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return regression_report(y_raw, pred_raw, prefix="raw_")

def compute_all_baselines(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    sentinel_index: Dict[Tuple[str, int], Any],
    weather_index: Dict[Tuple[str, int], Any],
    cfg: FineTuneConfig,
) -> Dict[str, Dict[str, Any]]:
    baselines: Dict[str, Dict[str, Any]] = {}
    baselines["mean_baseline"] = mean_baseline_metrics(train_df, val_df)
    if run_rf_baseline:
        try:
            baselines["random_forest_baseline"] = random_forest_baseline_metrics(train_df, val_df, sentinel_index, weather_index, cfg.seed)
        except Exception as e:
            baselines["random_forest_baseline"] = {"error": repr(e)}
    if run_cnn_baseline:
        try:
            baselines["cnn_baseline"] = cnn_sentinel_baseline_metrics(train_df, val_df, sentinel_index, cfg)
        except Exception as e:
            baselines["cnn_baseline"] = {"error": repr(e)}
    return baselines

def import_module_by_path(file_path: str):
    import importlib.util
    file_path = os.path.abspath(file_path)
    module_name = Path(file_path).stem + "_dynamic"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        spec.loader.exec_module(module)
    return module

def build_model(repo_root: str, num_regencies: int, use_regency_embedding: bool):
    mmst_module = import_module_by_path(os.path.join(repo_root, "models_mmst_vit.py"))
    pvt_module = import_module_by_path(os.path.join(repo_root, "models_pvt_simclr.py"))
    backbone = pvt_module.PVTSimCLR(base_model="pvt_tiny", out_dim=512, context_dim=40, pretrained=True)
    cls = mmst_module.MMST_ViT
    sig = inspect.signature(cls)
    candidate_kwargs = {
        "out_dim": 1,
        "num_grid": 4,
        "num_short_term_seq": 6,
        "num_long_term_seq": 12,
        "num_year": 5,
        "pvt_backbone": backbone,
        "context_dim": 40,
        "dim": None,
        "batch_size": 4,
        "depth": 2,
        "heads": 4,
        "dim_head": 64,
        "dropout": 0.2,
        "emb_dropout": 0.1,
        "scale_dim": 4,
        "pool": "cls",
        "num_regencies": num_regencies,
        "regency_emb_dim": 64,
        "aux_cont_dim": 6,
        "use_regency_embedding": use_regency_embedding,
    }
    filtered = {k: v for k, v in candidate_kwargs.items() if k in sig.parameters}
    model = cls(**filtered)
    return model

def build_sentinel_weather_indices(repo_root: str, processed_bps_root: str, years: Iterable[int], sentinel_root: str, weather_root: str):
    def build_sentinel_year_lookup_from_h5(root):
        h5_files = []
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if fn.lower().endswith((".h5", ".hdf5")):
                    h5_files.append(os.path.join(dp, fn))
        h5_files.sort()
        if not h5_files:
            raise FileNotFoundError(f"No H5 Sentinel files found under {root}")
        timeframe_tensors = []
        for h5_path in h5_files:
            with h5py.File(h5_path, "r") as f:
                tf_keys = []
                for key in f.keys():
                    if re.fullmatch(r"TF_\d+", key):
                        if "image" in f[key]:
                            tf_keys.append(key)
                tf_keys.sort(key=lambda x: int(x.split("_")[1]))
                if not tf_keys:
                    raise ValueError(f"No TF_##/image groups found in {h5_path}")
                for tf_key in tf_keys:
                    arr = np.asarray(f[tf_key]["image"][()])
                    if arr.ndim != 4:
                        raise ValueError(f"{tf_key}/image has unsupported shape {arr.shape}")
                    if arr.shape[0] != 4:
                        raise ValueError(f"{tf_key}/image expected first dim=4 tiles, got shape {arr.shape}")
                    arr = np.repeat(arr[None, ...], 6, axis=0)
                    tensor = torch.from_numpy(arr).float()
                    tensor = fix_sentinel_layout(tensor)
                    timeframe_tensors.append((tf_key, tensor))
        if not timeframe_tensors:
            raise ValueError("No Sentinel timeframe tensors parsed from H5 files")
        ordered_tensors = [x[1] for x in timeframe_tensors]
        year_list = list(years)
        if len(ordered_tensors) < len(year_list):
            raise ValueError(f"Only {len(ordered_tensors)} Sentinel timeframes were parsed, but need at least {len(year_list)}")
        chunks = np.array_split(np.arange(len(ordered_tensors)), len(year_list))
        year_lookup = {}
        for year, idxs in zip(year_list, chunks):
            seqs = [ordered_tensors[int(i)] for i in idxs]
            shapes = [tuple(s.shape) for s in seqs]
            if len(set(shapes)) != 1:
                raise ValueError(f"Inconsistent Sentinel sequence shapes for year {year}: {shapes}")
            stacked = torch.stack(seqs, dim=0)
            year_lookup[int(year)] = stacked.mean(dim=0)
        return year_lookup

    sentinel_year_lookup = build_sentinel_year_lookup_from_h5(sentinel_root)
    sentinel_index = build_sentinel_index(processed_bps_root, years, sentinel_year_lookup)
    weather_index = build_weather_index(processed_bps_root, years, weather_root)
    return sentinel_index, weather_index

@dataclass
class FineTuneConfig:
    repo_root: str = os.environ.get("mmst_repo_root", "/vol/home/s3881946/Downloads/MMST-ViT-main")
    processed_bps_root: str = os.environ.get("mmst_processed_bps_root", "/vol/home/s3881946/Downloads/Processed_BPS_Data")
    sentinel_root: str = os.environ.get("mmst_sentinel_root", "/vol/home/s3881946/Downloads/H5_Loader_Input")
    weather_root: str = os.environ.get("mmst_weather_root", "/vol/home/s3881946/Downloads/ERA5_Data")
    years: Tuple[int, ...] = default_years
    batch_size: int = int(os.environ.get("mmst_batch_size", "1"))
    num_workers: int = int(os.environ.get("mmst_num_workers", "0"))
    epochs: int = int(os.environ.get("mmst_epochs", "100"))
    lr_head: float = float(os.environ.get("mmst_lr_head", "1e-3"))
    lr_backbone: float = float(os.environ.get("mmst_lr_backbone", "1e-5"))
    weight_decay: float = float(os.environ.get("mmst_weight_decay", "1e-4"))
    val_ratio: float = float(os.environ.get("mmst_val_ratio", "0.2"))
    seed: int = 42
    amp_enabled: bool = env_bool("mmst_amp_enabled", "0")
    grad_clip: float = float(os.environ.get("mmst_grad_clip", "1.0"))
    min_lr: float = float(os.environ.get("mmst_min_lr", "1e-6"))
    monitor_metric: str = os.environ.get("mmst_monitor_metric", "raw_r2")
    device: str = os.environ.get("mmst_device", "cuda" if torch.cuda.is_available() else "cpu")
    freeze_backbone_epochs: int = int(os.environ.get("mmst_freeze_backbone_epochs", "2"))
    accumulation_steps: int = int(os.environ.get("mmst_accumulation_steps", "4"))
    early_stopping_patience: int = int(os.environ.get("mmst_early_stopping_patience", "20"))

def initialize_lazy_modules(model: nn.Module, train_loader: DataLoader, device: torch.device) -> None:
    model = model.to(device)
    model.eval()
    first_batch = next(iter(train_loader))
    first_batch = move_batch_to_device(first_batch, device)
    with torch.no_grad():
        _ = forward_mmst_model(model, first_batch)

def safe_numel(p: torch.nn.Parameter) -> int:
    try:
        return p.numel()
    except ValueError:
        return -1

def train_and_evaluate(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, scaler_obj: TargetScaler, cfg: FineTuneConfig):
    device = torch.device(cfg.device)
    model = model.to(device)
    initialize_lazy_modules(model, train_loader, device)
    backbone_params = []
    head_params = []
    for name, p in model.named_parameters():
        if "pvt_backbone" in name:
            backbone_params.append(p)
        else:
            head_params.append(p)
    for p in backbone_params:
        p.requires_grad = False
    optimizer = torch.optim.AdamW([{"params": head_params, "lr": cfg.lr_head}, {"params": backbone_params, "lr": cfg.lr_backbone}], weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr)
    amp_scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp_enabled)
    trainable = [(n, safe_numel(p)) for n, p in model.named_parameters() if p.requires_grad]
    best_score = -np.inf
    best_epoch = -1
    best_state = None
    bad_epochs = 0
    history = []
    with torch.no_grad():
        head_before = None
        for n, p in model.named_parameters():
            if "mlp_head" in n and "weight" in n:
                try:
                    head_before = p.detach().clone()
                except Exception:
                    head_before = None
                break
    for epoch in range(1, cfg.epochs + 1):
        start = time.time()
        if epoch == cfg.freeze_backbone_epochs + 1:
            for p in backbone_params:
                p.requires_grad = True
        train_loss = train_one_epoch(model=model, loader=train_loader, optimizer=optimizer, device=device, amp_enabled=cfg.amp_enabled, grad_clip=cfg.grad_clip, scaler=amp_scaler, accumulation_steps=cfg.accumulation_steps)
        val_metrics = evaluate(model, val_loader, device, scaler_obj.mean, scaler_obj.std)
        score = val_metrics[cfg.monitor_metric]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        scheduler.step()
        current_lrs = [pg["lr"] for pg in optimizer.param_groups]
        elapsed = time.time() - start
        head_delta = None
        if head_before is not None:
            for n, p in model.named_parameters():
                if "mlp_head" in n and "weight" in n:
                    try:
                        head_delta = (p.detach() - head_before).abs().mean().item()
                    except Exception:
                        head_delta = None
                    break
        row = {"epoch": epoch, "train_loss": train_loss, "lr_head": current_lrs[0], "lr_backbone": current_lrs[1], "head_weight_delta": head_delta, **val_metrics}
        history.append(row)
        print(f"Epoch {epoch}/{cfg.epochs} - rmse: {val_metrics['raw_rmse']:.4f} | mae: {val_metrics['raw_mae']:.4f} | r^2: {val_metrics['raw_r2']:.4f} | pcc: {val_metrics['raw_pcc']:.4f} |")
        if bad_epochs >= cfg.early_stopping_patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    final_prediction_table = predict_table(model, val_loader, device, scaler_obj.mean, scaler_obj.std)
    final_metrics = regression_report(
        final_prediction_table["true_raw"].to_numpy(dtype=np.float64),
        final_prediction_table["pred_raw"].to_numpy(dtype=np.float64),
        prefix="raw_",
    )
    return model, history, best_epoch, best_score, final_metrics, final_prediction_table

def make_split(samples: pd.DataFrame, cfg: FineTuneConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if split_mode == "standard":
        return grouped_train_val_split(samples, group_col="group_key", val_ratio=cfg.val_ratio, seed=cfg.seed)
    if split_mode == "temporal":
        train_df = samples[samples["year"].astype(int) != int(holdout_year)].reset_index(drop=True)
        val_df = samples[samples["year"].astype(int) == int(holdout_year)].reset_index(drop=True)
        if train_df.empty or val_df.empty:
            raise ValueError("Temporal split produced empty train or validation set.")
        return train_df, val_df
    if split_mode == "spatial":
        df = samples.copy().reset_index(drop=True)
        df["_spatial_group"] = df["province"].astype(str) + "__" + df["regency"].astype(str)
        groups = sorted(df["_spatial_group"].astype(str).unique().tolist())
        rng = np.random.default_rng(spatial_split_seed)
        rng.shuffle(groups)
        n_val = max(1, int(round(len(groups) * cfg.val_ratio)))
        if n_val >= len(groups):
            n_val = len(groups) - 1
        val_groups = set(groups[:n_val])
        train_df = df[~df["_spatial_group"].astype(str).isin(val_groups)].drop(columns=["_spatial_group"]).reset_index(drop=True)
        val_df = df[df["_spatial_group"].astype(str).isin(val_groups)].drop(columns=["_spatial_group"]).reset_index(drop=True)
        if train_df.empty or val_df.empty:
            raise ValueError("Spatial split produced empty train or validation set.")
        return train_df, val_df
    raise ValueError(f"Unknown split_mode={split_mode}")

def prepare_data_and_model(cfg: FineTuneConfig):
    sentinel_index, weather_index = quiet_call(build_sentinel_weather_indices, repo_root=cfg.repo_root, processed_bps_root=cfg.processed_bps_root, years=cfg.years, sentinel_root=cfg.sentinel_root, weather_root=cfg.weather_root)
    samples = quiet_call(load_processed_bps_tile_rows, cfg.processed_bps_root, cfg.years)
    valid_keys = set(sentinel_index.keys()) & set(weather_index.keys())
    samples = samples[samples.apply(lambda r: (normalize_regency_name(r["regency"]), int(r["year"])) in valid_keys, axis=1)].reset_index(drop=True)
    if samples.empty:
        raise ValueError("No regency-year samples matched both Sentinel and weather indices.")
    unique_regencies = sorted(samples["regency"].astype(str).unique().tolist())
    regency_to_idx = {r: i for i, r in enumerate(unique_regencies)}
    samples["regency_idx"] = samples["regency"].astype(str).map(regency_to_idx).astype(int)
    unique_provinces = sorted(samples["province"].astype(str).unique().tolist())
    province_to_idx = {p: i for i, p in enumerate(unique_provinces)}
    samples["province_idx"] = samples["province"].astype(str).map(province_to_idx).astype(int)
    year_min = min(cfg.years)
    year_max = max(cfg.years)
    samples["year_norm_aux"] = (samples["year"].astype(float) - year_min) / max(year_max - year_min, 1)
    model = quiet_call(build_model, cfg.repo_root, num_regencies=len(unique_regencies), use_regency_embedding=use_regency_embedding)
    train_df, val_df = quiet_call(make_split, samples, cfg)
    scaler_obj = fit_target_scaler(train_df, target_col="target")
    train_df = apply_target_scaler(train_df, scaler_obj, target_col="target")
    val_df = apply_target_scaler(val_df, scaler_obj, target_col="target")
    baseline = quiet_call(compute_all_baselines, train_df, val_df, sentinel_index, weather_index, cfg)
    train_ds = RiceFineTuneDataset(train_df, sentinel_index, weather_index)
    val_ds = RiceFineTuneDataset(val_df, sentinel_index, weather_index)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=cfg.device.startswith("cuda"), collate_fn=collate_mmst_batch, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=cfg.device.startswith("cuda"), collate_fn=collate_mmst_batch, drop_last=False)
    return model, train_loader, val_loader, scaler_obj, baseline

def run_one_seed(seed: int) -> dict:
    seed_dir = output_root / f"seed_{seed}"
    result_path = seed_dir / "finetune_result.json"
    if result_path.exists() and not overwrite_existing:
        with open(result_path, "r", encoding="utf-8") as f:
            return json.load(f)
    seed_dir.mkdir(parents=True, exist_ok=True)
    cfg = FineTuneConfig(seed=seed)
    seed_everything(seed)
    model, train_loader, val_loader, scaler_obj, baseline = prepare_data_and_model(cfg)
    model, history, best_epoch, best_score, final_metrics, prediction_table = train_and_evaluate(model=model, train_loader=train_loader, val_loader=val_loader, scaler_obj=scaler_obj, cfg=cfg)
    prediction_csv = seed_dir / "validation_predictions.csv"
    prediction_table.to_csv(prediction_csv, index=False)
    result = {
        "config": {
            "experiment": experiment_name,
            "split_mode": split_mode,
            "holdout_year": holdout_year if split_mode == "temporal" else None,
            "spatial_split_seed": spatial_split_seed if split_mode == "spatial" else None,
            "years": list(cfg.years),
            "seed": seed,
            "use_regency_embedding": use_regency_embedding,
            "monitor_metric": cfg.monitor_metric,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr_head": cfg.lr_head,
            "lr_backbone": cfg.lr_backbone,
            "weight_decay": cfg.weight_decay,
            "val_ratio": cfg.val_ratio,
        },
        "best_epoch": best_epoch,
        "best_score": best_score,
        "final_val_metrics": final_metrics,
        "mean_baseline": baseline.get("mean_baseline", {}),
        "baseline_mean": baseline.get("mean_baseline", {}),
        "random_forest_baseline": baseline.get("random_forest_baseline", {}),
        "cnn_baseline": baseline.get("cnn_baseline", {}),
        "baselines": baseline,
        "prediction_table": str(prediction_csv),
        "predictions": prediction_table.to_dict(orient="records"),
    }
    with open(seed_dir / "finetune_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    torch.save(model.state_dict(), seed_dir / "best_finetune_checkpoint.pt")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result

raw_summary_metrics = ["raw_rmse", "raw_mae", "raw_r2", "raw_pcc"]

def metric_mean(values: List[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())

def metric_std(values: List[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size <= 1:
        return 0.0 if arr.size == 1 else float("nan")
    return float(arr.std(ddof=1))

def metric_min(values: List[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(arr.min())

def metric_max(values: List[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(arr.max())

def collect_metric(results: List[dict], section: str, metric: str) -> List[float]:
    values = []
    for result in results:
        metrics = result.get(section, {})
        value = metrics.get(metric)
        if isinstance(value, (int, float, np.integer, np.floating)):
            values.append(float(value))
    return values

def summarize_results(results: List[dict]) -> dict:
    summary = {}
    for metric in raw_summary_metrics:
        model_values = collect_metric(results, "final_val_metrics", metric)
        mean_values = collect_metric(results, "mean_baseline", metric)
        rf_values = collect_metric(results, "random_forest_baseline", metric)
        cnn_values = collect_metric(results, "cnn_baseline", metric)
        summary[metric] = {
            "model_mean": metric_mean(model_values),
            "model_std": metric_std(model_values),
            "model_min": metric_min(model_values),
            "model_max": metric_max(model_values),
            "baseline_mean": metric_mean(mean_values),
            "baseline_rf": metric_mean(rf_values),
            "baseline_cnn": metric_mean(cnn_values),
        }
    return summary

def print_summary(summary: dict) -> None:
    print("\nFinal raw multi-seed summary")
    header = f"{'metric':14s} {'model_mean':>14s} {'model_std':>14s} {'model_min':>14s} {'model_max':>14s} {'baseline_mean':>14s} {'baseline_rf':>14s} {'baseline_cnn':>14s}"
    print(header)
    print("-" * len(header))
    for metric in raw_summary_metrics:
        row = summary.get(metric, {})
        print(
            f"{metric:14s} "
            f"{row.get('model_mean', float('nan')):14.6f} "
            f"{row.get('model_std', float('nan')):14.6f} "
            f"{row.get('model_min', float('nan')):14.6f} "
            f"{row.get('model_max', float('nan')):14.6f} "
            f"{row.get('baseline_mean', float('nan')):14.6f} "
            f"{row.get('baseline_rf', float('nan')):14.6f} "
            f"{row.get('baseline_cnn', float('nan')):14.6f}"
        )

def print_seed_table(seed_index: int, seed: int, result: dict) -> None:
    metrics = result.get("final_val_metrics", {})
    print(f"Seed {seed_index} done")
    print(f"{'metric':14s} {'model':>14s}")
    print("-" * 29)
    for metric in raw_summary_metrics:
        print(f"{metric:14s} {metrics.get(metric, float('nan')):14.6f}")


def save_combined_prediction_table(results: List[dict], path: Path) -> None:
    rows = []
    for result in results:
        seed = result.get("config", {}).get("seed")
        for row in result.get("predictions", []):
            item = dict(row)
            item["seed"] = seed
            rows.append(item)
    if not rows:
        pd.DataFrame().to_csv(path, index=False)
        return
    df = pd.DataFrame(rows)
    group_cols = ["sample_id", "province", "regency", "year"]
    combined = (
        df.groupby(group_cols, as_index=False)
        .agg(
            true_raw=("true_raw", "mean"),
            pred_raw_mean=("pred_raw", "mean"),
            pred_raw_std=("pred_raw", "std"),
            pred_raw_min=("pred_raw", "min"),
            pred_raw_max=("pred_raw", "max"),
            signed_error_mean=("signed_error", "mean"),
            abs_error_mean=("abs_error", "mean"),
            n_seed_predictions=("pred_raw", "count"),
        )
        .sort_values(["year", "province", "regency"])
        .reset_index(drop=True)
    )
    combined["pred_raw_std"] = combined["pred_raw_std"].fillna(0.0)
    combined.to_csv(path, index=False)

def save_raw_summary_csv(summary: dict, path: Path) -> None:
    rows = []
    for metric in raw_summary_metrics:
        row = {"metric": metric}
        row.update(summary.get(metric, {}))
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)

def main() -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for seed_index, seed in enumerate(seeds, start=1):
        result = run_one_seed(seed)
        results.append(result)
        print_seed_table(seed_index, seed, result)
        with open(output_root / "partial_seed_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    summary = summarize_results(results)
    with open(output_root / "all_seed_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(output_root / "summary_raw_mean_std.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    save_raw_summary_csv(summary, output_root / "summary_table_spatial.csv")
    save_combined_prediction_table(results, output_root / "validation_predictions_spatial.csv")
    print_summary(summary)

if __name__ == "__main__":
    main()