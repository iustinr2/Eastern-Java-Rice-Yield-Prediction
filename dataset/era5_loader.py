from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import xarray as xr
except ImportError:
    raise ImportError()


input_dir = "/vol/home/s3881946/Downloads/ERA5_Data"

weather_features = [
    "dewpoint_temperature",
    "temperature",
    "soil_temperature",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "surface_solar_radiation_downwards",
    "total_evaporation",
    "u10",
    "v10",
    "total_precipitation",
]

feature_abbrevs = {
    "dewpoint_temperature": ["dewpoint_temperature", "d2m", "2m_dewpoint_temperature"],
    "temperature": ["temperature", "t2m", "2m_temperature"],
    "soil_temperature": ["soil_temperature", "stl1"],
    "volumetric_soil_water_layer_1": ["volumetric_soil_water_layer_1", "swvl1"],
    "volumetric_soil_water_layer_2": ["volumetric_soil_water_layer_2", "swvl2"],
    "surface_solar_radiation_downwards": ["surface_solar_radiation_downwards", "ssrd"],
    "total_evaporation": ["total_evaporation", "e"],
    "u10": ["u10", "10m_u_component_of_wind"],
    "v10": ["v10", "10m_v_component_of_wind"],
    "total_precipitation": ["total_precipitation", "tp"],
}


def get_number(name: str):
    text = name.strip().lower()

    if text.startswith("t"):
        text = text[1:]

    if text.isdigit():
        return int(text)

    return None


def get_dirs(root: Path):
    folders = []

    for path in root.iterdir():
        if not path.is_dir():
            continue

        num = get_number(path.name)

        if num is not None:
            folders.append((num, path))

    folders = sorted(folders, key=lambda x: x[0])

    if len(folders) != 24:
        print(f"Expected 24 folders, found {len(folders)}")

    return folders


def get_months(folder: Path):
    months = [path for path in folder.iterdir() if path.is_dir()]

    def sort_key(path: Path):
        try:
            return int(path.name)
        except Exception:
            return path.name

    months = sorted(months, key=sort_key)

    if len(months) != 2:
        raise FileNotFoundError(f"Expected 2 month long folders, found {len(months)}: {folder}")

    return months


def get_nc(folder: Path):
    files = [path for path in folder.iterdir() if path.is_file()]
    nc_files = [path for path in files if path.suffix.lower() == ".nc"]

    if len(nc_files) != 1:
        raise FileNotFoundError(f"Expected 1 nc file, found {len(nc_files)}: {folder}")

    return nc_files[0]


def find_name(ds, names):
    found = {name.lower(): name for name in ds.data_vars.keys()}

    for name in names:
        if name.lower() in found:
            return found[name.lower()]

    return None


def get_series(data):
    dimensions = list(data.dims)

    if "time" in dimensions:
        other_dimensions = [dimension for dimension in dimensions if dimension != "time"]

        if other_dimensions:
            data = data.mean(dim=other_dimensions, skipna=True)

        return np.asarray(data.values, dtype=np.float32).reshape(-1)

    values = np.asarray(data.values, dtype=np.float32)

    if values.ndim == 0:
        return np.array([float(values)], dtype=np.float32)

    return np.array([float(np.nanmean(values))], dtype=np.float32)


def read_nc(path: Path):
    ds = xr.open_dataset(path)

    try:
        series_list = []

        for name in weather_features:
            var = find_name(ds, feature_abbrevs[name])

            if var is None:
                raise KeyError(f"Missing weather variable {name}: {path}")

            series_list.append(get_series(ds[var]))

    finally:
        ds.close()

    size = max(len(series) for series in series_list)
    size = max(2, size)

    fixed_series = []

    for series in series_list:
        if len(series) == 1:
            series = np.repeat(series, repeats=size)
        elif len(series) < size:
            series = np.pad(series, (0, size - len(series)), mode="edge")

        fixed_series.append(series)

    arr = np.stack(fixed_series, axis=1)

    if arr.shape[0] == 1:
        arr = np.repeat(arr, repeats=2, axis=0)

    arr = arr[:2, :].astype(np.float32)

    if arr.shape != (2, len(weather_features)):
        raise ValueError(f"Incorrect weather shape {arr.shape}: {path}")

    return arr


class ERA5_Dataset(Dataset):
    def __init__(self, root_dir=input_dir, data_file=None):
        self.root_dir = Path(root_dir)

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Weather folder not found: {self.root_dir}")

        self.timeframe_dirs = get_dirs(self.root_dir)
        self.samples = []

        for num, folder in self.timeframe_dirs:
            months = get_months(folder)
            blocks = []

            for month in months:
                nc_file = get_nc(month)
                blocks.append(read_nc(nc_file))

            arr = np.concatenate(blocks, axis=0)

            if arr.shape != (4, len(weather_features)):
                raise ValueError(f"Incorrect weather shape {arr.shape}: timeframe {num}")

            self.samples.append((num, arr))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        num, arr = self.samples[idx]
        y_short = np.repeat(arr[None, :, :], repeats=4, axis=0)
        y_short = torch.tensor(y_short, dtype=torch.float32).unsqueeze(0)

        return y_short, f"TF_{num:02d}"


class ERA5_SequenceDataset(Dataset):
    def __init__(self, root_dir=input_dir, data_file=None):
        self.base = ERA5_Dataset(root_dir, data_file)

        if len(self.base) != 24:
            raise ValueError(f"Expected 24 samples, found {len(self.base)}")

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        timeframes = []

        for i in range(24):
            y_short, _ = self.base[i]
            timeframes.append(y_short.squeeze(0))

        timeframes = torch.stack(timeframes, dim=0)

        ys = timeframes.reshape(6, 4, *timeframes.shape[1:]).mean(dim=1)
        ys = ys.unsqueeze(0)

        mean = timeframes.mean(dim=(0, 1))
        mean = mean.mean(dim=0)
        yl = mean.view(1, 1, 1, -1).repeat(1, 5, 12, 1)

        return ys, yl, "SEQ_001"


if __name__ == "__main__":
    ds = ERA5_Dataset()
    print(f"Dataset length: {len(ds)}")

    if len(ds) == 0:
        raise ValueError(f"No weather samples found: {input_dir}")

    y, name = ds[0]
    print(name, y.shape)

    seq_ds = ERA5_SequenceDataset()
    print(f"Squence length: {len(seq_ds)}")

    ys, yl, seq_name = seq_ds[0]
    print(seq_name, ys.shape, yl.shape)