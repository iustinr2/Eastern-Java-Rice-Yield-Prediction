from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from dataset.bps_loader import find_csv, clean_columns, check_columns, clean_text, clean_regency


def _to_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.float()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return torch.tensor(x, dtype=torch.float32)


def _extract_year_from_string(s: str) -> Optional[int]:
    if not s:
        return None
    matches = re.findall(r"(20\d{2})", str(s))
    if not matches:
        return None
    return int(matches[0])


def _extract_year(item: Any) -> Optional[int]:
    if isinstance(item, dict):
        for key in ("year", "target_year", "sequence_year", "label_year"):
            if key in item and item[key] is not None:
                try:
                    return int(item[key])
                except Exception:
                    pass
        for key in ("sample_id", "id", "name", "path", "file", "filename"):
            if key in item and item[key] is not None:
                y = _extract_year_from_string(str(item[key]))
                if y is not None:
                    return y
    if isinstance(item, (tuple, list)) and len(item) >= 2:
        try:
            return int(item[1])
        except Exception:
            pass
    y = _extract_year_from_string(str(item))
    return y


def _extract_value_from_item(item: Any, candidate_keys: Iterable[str]) -> Any:
    if isinstance(item, dict):
        for k in candidate_keys:
            if k in item and item[k] is not None:
                return item[k]
    return None


def _fix_sentinel_layout(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 5:
        raise ValueError(f"Sentinel sample must be 5D, got shape {tuple(x.shape)}")
    if x.shape[2] <= 32 and x.shape[-1] > 32:
        return x.contiguous()
    return x.permute(0, 1, 4, 2, 3).contiguous()


def _normalize_tile_weights(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["tl_pct", "tr_pct", "bl_pct", "br_pct"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    weights = df[["tl_pct", "tr_pct", "bl_pct", "br_pct"]].to_numpy(dtype=np.float64)
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    weights = weights / row_sums

    df = df.copy()
    df["tile_w_tl"] = weights[:, 0]
    df["tile_w_tr"] = weights[:, 1]
    df["tile_w_bl"] = weights[:, 2]
    df["tile_w_br"] = weights[:, 3]
    return df


def load_processed_bps_tile_rows(processed_bps_root: str, years: Iterable[int]) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []

    for year in years:
        csv_path = find_csv(processed_bps_root, int(year))
        df = pd.read_csv(csv_path)
        df = clean_columns(df)
        df = check_columns(df)

        if "province" not in df.columns:
            raise KeyError("Missing required column 'province'")
        if "regency_name" not in df.columns:
            raise KeyError("Missing required column 'regency_name'")
        if "year" not in df.columns:
            df["year"] = int(year)

        df["province"] = df["province"].map(clean_text())
        df["regency"] = df["regency_name"].map(clean_regency())
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

        for col in ["tl_pct", "tr_pct", "bl_pct", "br_pct"]:
            if col not in df.columns:
                df[col] = 0.0

        parts.append(df)

    if not parts:
        raise ValueError("No processed BPS tile rows loaded")

    df = pd.concat(parts, ignore_index=True)
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    df = _normalize_tile_weights(df)

    grouped = (
        df.groupby(["province", "regency", "year"], as_index=False)
        .agg(
            tl_pct=("tl_pct", "mean"),
            tr_pct=("tr_pct", "mean"),
            bl_pct=("bl_pct", "mean"),
            br_pct=("br_pct", "mean"),
            tile_w_tl=("tile_w_tl", "mean"),
            tile_w_tr=("tile_w_tr", "mean"),
            tile_w_bl=("tile_w_bl", "mean"),
            tile_w_br=("tile_w_br", "mean"),
        )
        .copy()
    )

    grouped["sample_id"] = (
        grouped["province"].astype(str)
        + "__"
        + grouped["regency"].astype(str)
        + "__"
        + grouped["year"].astype(str)
    )
    return grouped


def collect_sentinel_year_lookup(samples: Iterable[Any]) -> Dict[int, torch.Tensor]:
    lookup: Dict[int, torch.Tensor] = {}

    for item in samples:
        year = _extract_year(item)
        if year is None:
            continue

        value = _extract_value_from_item(item, ("sentinel", "x", "image", "images", "sequence", "tensor"))
        if value is None and isinstance(item, (tuple, list)) and len(item) >= 1:
            value = item[0]
        if value is None:
            continue

        tensor = _fix_sentinel_layout(_to_tensor(value))
        lookup[int(year)] = tensor

    if not lookup:
        raise ValueError("No Sentinel year lookup could be built from the provided iterable")

    return lookup


def collect_weather_year_lookup(samples: Iterable[Any]) -> Dict[int, Dict[str, Optional[torch.Tensor]]]:
    lookup: Dict[int, Dict[str, Optional[torch.Tensor]]] = {}

    for item in samples:
        year = _extract_year(item)
        if year is None:
            continue

        ys = None
        yl = None

        if isinstance(item, dict):
            if "ys" in item:
                ys = item["ys"]
            if "yl" in item:
                yl = item["yl"]
            if ys is None:
                for k in ("weather", "x_weather", "context", "tensor"):
                    if k in item and item[k] is not None:
                        ys = item[k]
                        break
        elif isinstance(item, (tuple, list)):
            if len(item) == 2:
                ys, yl = item
            elif len(item) >= 1:
                ys = item[0]

        if ys is None:
            continue

        ys = _to_tensor(ys)
        yl = _to_tensor(yl) if yl is not None else None
        lookup[int(year)] = {"ys": ys, "yl": yl}

    if not lookup:
        raise ValueError("No weather year lookup could be built from the provided iterable")

    return lookup


def build_sentinel_index(
    processed_bps_root: str,
    years: Iterable[int],
    sentinel_year_lookup: Dict[int, Any],
) -> Dict[Tuple[str, int], torch.Tensor]:
    tile_rows = load_processed_bps_tile_rows(processed_bps_root, years)
    sentinel_index: Dict[Tuple[str, int], torch.Tensor] = {}

    for _, row in tile_rows.iterrows():
        year = int(row["year"])
        regency = str(row["regency"])
        key = (regency, year)

        if year not in sentinel_year_lookup:
            continue

        seq = _fix_sentinel_layout(_to_tensor(sentinel_year_lookup[year])).clone()

        if seq.shape[1] != 4:
            raise ValueError(
                f"Sentinel sequence for year={year} must have 4 tiles in dim=1, got shape {tuple(seq.shape)}"
            )

        weights = torch.tensor(
            [row["tile_w_tl"], row["tile_w_tr"], row["tile_w_bl"], row["tile_w_br"]],
            dtype=seq.dtype,
            device=seq.device,
        ).view(1, 4, 1, 1, 1)

        weighted_seq = seq * weights
        sentinel_index[key] = weighted_seq

    if not sentinel_index:
        raise ValueError("Sentinel index is empty after matching regency-year rows to Sentinel year lookup")

    return sentinel_index


def build_weather_index(
    processed_bps_root: str,
    years: Iterable[int],
    weather_year_lookup: Dict[int, Any],
) -> Dict[Tuple[str, int], Dict[str, Optional[torch.Tensor]]]:
    tile_rows = load_processed_bps_tile_rows(processed_bps_root, years)
    weather_index: Dict[Tuple[str, int], Dict[str, Optional[torch.Tensor]]] = {}

    for _, row in tile_rows.iterrows():
        year = int(row["year"])
        regency = str(row["regency"])
        key = (regency, year)

        if year not in weather_year_lookup:
            continue

        item = weather_year_lookup[year]

        if isinstance(item, dict):
            ys = _to_tensor(item.get("ys"))
            yl = _to_tensor(item.get("yl")) if item.get("yl") is not None else None
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            ys = _to_tensor(item[0])
            yl = _to_tensor(item[1]) if item[1] is not None else None
        else:
            ys = _to_tensor(item)
            yl = None

        weather_index[key] = {"ys": ys, "yl": yl}

    if not weather_index:
        raise ValueError("Weather index is empty after matching regency-year rows to weather year lookup")

    return weather_index


def build_indices_from_year_lookups(
    processed_bps_root: str,
    years: Iterable[int],
    sentinel_year_lookup: Dict[int, Any],
    weather_year_lookup: Dict[int, Any],
) -> Tuple[Dict[Tuple[str, int], torch.Tensor], Dict[Tuple[str, int], Dict[str, Optional[torch.Tensor]]]]:
    sentinel_index = build_sentinel_index(
        processed_bps_root=processed_bps_root,
        years=years,
        sentinel_year_lookup=sentinel_year_lookup,
    )
    weather_index = build_weather_index(
        processed_bps_root=processed_bps_root,
        years=years,
        weather_year_lookup=weather_year_lookup,
    )
    return sentinel_index, weather_index