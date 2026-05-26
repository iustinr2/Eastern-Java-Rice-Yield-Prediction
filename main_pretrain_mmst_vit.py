import argparse
import contextlib
import io
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm.optim.optim_factory as optim_factory

import util.lr_sched as lr_sched
import util.misc as misc
from dataset import sentinel_wrapper
from dataset.era5_loader import ERA5_Dataset
from dataset.sentinel_loader import Sentinel_Dataset
from loss.contrastive_loss import ContrastiveLoss
from models_pvt_simclr import PVTSimCLR
from util.misc import NativeScalerWithGradNormCount as NativeScaler


context_dim = 10

base_sentinel_root_dir = "/vol/home/s3881946/Downloads/H5_Loader_Input"
base_sentinel_h5 = "/vol/home/s3881946/Downloads/H5_Loader_Input/sentinel_input.h5"
base_sentinel_manifest = "/vol/home/s3881946/Downloads/JSON_Loader_Input/sentinel_manifest.json"
base_weather_root_dir = "/vol/home/s3881946/Downloads/ERA5_Data"

extra_sentinel_root_dir = "/vol/home/s3881946/Downloads/H5_Loader_Input"
extra_sentinel_h5 = "/vol/home/s3881946/Downloads/H5_Loader_Input/sentinel_pretrain.h5"
extra_sentinel_manifest = "/vol/home/s3881946/Downloads/JSON_Loader_Input/sentinel_pretrain_manifest.json"
extra_weather_root_dir = "/vol/home/s3881946/Downloads/Pretrain_ERA5_Data"
extra_era5_timeframe_cache = "/vol/home/s3881946/Downloads/MMST-ViT-main/output_dir/pretrain_era5_timeframe_cache"

default_output_dir = "./output_dir/pvt_simclr_combined_pretrain"
default_log_dir = "./output_dir/pvt_simclr_combined_pretrain"

timeframe_to_months = {
    1: [1, 2],
    2: [3, 4],
    3: [5, 6],
    4: [7, 8],
    5: [9, 10],
    6: [11, 12],
}

extra_pretrain_start_year = 2017
extra_pretrain_timeframes_per_year = 6

month_name_to_num = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


class PairedSentinelWeatherDataset(torch.utils.data.Dataset):
    def __init__(self, sentinel_dataset, weather_dataset, name: str):
        self.sentinel_dataset = sentinel_dataset
        self.weather_dataset = weather_dataset
        self.name = name

        if len(self.sentinel_dataset) != len(self.weather_dataset):
            raise ValueError(
                f"[{self.name}] Sentinel / weather count mismatch: "
                f"{len(self.sentinel_dataset)} vs {len(self.weather_dataset)}"
            )

        if len(self.sentinel_dataset) == 0:
            raise ValueError(f"[{self.name}] Dataset is empty.")

    def __len__(self):
        return len(self.sentinel_dataset)

    def __getitem__(self, idx):
        x = self.sentinel_dataset[idx]
        y = self.weather_dataset[idx]
        return x, y


def get_args_parser():
    parser = argparse.ArgumentParser("PVT SimCLR pre-training", add_help=False)

    parser.add_argument("--batch_size", default=16, type=int, help="Batch size for contrastive pairs")
    parser.add_argument("--embed_dim", default=512, type=int, help="Embedding dimension")
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--model", default="pvt_tiny", type=str, metavar="MODEL", help="Backbone model")
    parser.add_argument("--accum_iter", default=1, type=int, help="Gradient accumulation steps")
    parser.add_argument("--input_size", default=224, type=int, help="Input image size")

    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--layer_decay", type=float, default=0.75)
    parser.add_argument("--lr", type=float, default=None, metavar="LR")
    parser.add_argument("--blr", type=float, default=1e-3, metavar="LR")
    parser.add_argument("--min_lr", type=float, default=0.0, metavar="LR")
    parser.add_argument("--warmup_epochs", type=int, default=20, metavar="N")

    parser.add_argument("--output_dir", default=default_output_dir)
    parser.add_argument("--log_dir", default=default_log_dir)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N")
    parser.add_argument("--num_workers", default=4, type=int)

    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")

    parser.add_argument("--sentinel_root_dir", type=str, default=base_sentinel_root_dir)
    parser.add_argument("--sentinel_h5_file", type=str, default=base_sentinel_h5)
    parser.add_argument("--sentinel_data_file", type=str, default=base_sentinel_manifest)
    parser.add_argument("--weather_root_dir", type=str, default=base_weather_root_dir)

    parser.add_argument(
        "--use_extra_pretrain_data",
        dest="use_extra_pretrain_data",
        action="store_true",
        default=True,
        help="Use additional 2017-2021 pretraining-only Sentinel/ERA5 data",
    )
    parser.add_argument(
        "--no_extra_pretrain_data",
        dest="use_extra_pretrain_data",
        action="store_false",
        help="Disable the additional 2017-2021 pretraining-only data",
    )

    parser.add_argument("--extra_sentinel_root_dir", type=str, default=extra_sentinel_root_dir)
    parser.add_argument("--extra_sentinel_h5_file", type=str, default=extra_sentinel_h5)
    parser.add_argument("--extra_sentinel_data_file", type=str, default=extra_sentinel_manifest)
    parser.add_argument("--extra_weather_root_dir", type=str, default=extra_weather_root_dir)
    parser.add_argument("--extra_era5_timeframe_cache", type=str, default=extra_era5_timeframe_cache)
    parser.add_argument("--save_freq", type=int, default=5)

    return parser


def require_file(path: str, name: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{name} not found: {path}")


def require_dir(path: str, name: str):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{name} not found: {path}")


def resolve_device(device_str: str) -> str:
    if not device_str.startswith("cuda"):
        return device_str

    if not torch.cuda.is_available():
        return "cpu"

    if device_str == "cuda" or ":" not in device_str:
        return "cuda:0"

    try:
        gpu_index = int(device_str.split(":")[1])
    except ValueError:
        return "cuda:0"

    if gpu_index >= torch.cuda.device_count():
        return "cuda:0"

    return device_str


def parse_year_tf_from_manifest_item(item: dict):
    year = item.get("year", None)
    tf = item.get("timeframe_index", None)

    if year is not None and tf is not None:
        year = int(year)
        tf = int(tf)

        if tf in timeframe_to_months:
            return year, tf

    text = " ".join(
        [
            str(item.get("group", "")),
            str(item.get("id", "")),
            str(item.get("source_dir", "")),
        ]
    )

    m = re.search(r"Y(\d{4})_TF_?(\d+)", text, flags=re.IGNORECASE)

    if m is not None:
        year = int(m.group(1))
        tf = int(m.group(2))

        if tf not in timeframe_to_months:
            raise ValueError(f"Parsed unsupported timeframe_index={tf} from manifest item: {item}")

        return year, tf

    m = re.search(
        r"(?:^|[^A-Za-z0-9])TF_?(\d+)(?:$|[^A-Za-z0-9])",
        text,
        flags=re.IGNORECASE,
    )

    if m is None:
        m = re.search(
            r"(?:^|[/\\_\-\s])T_?(\d+)(?:$|[/\\_\-\s])",
            text,
            flags=re.IGNORECASE,
        )

    if m is None:
        raise ValueError(f"Could not infer year/timeframe from manifest item: {item}")

    global_tf_idx = int(m.group(1))

    if global_tf_idx <= 0:
        raise ValueError(f"Invalid global timeframe index {global_tf_idx} from manifest item: {item}")

    inferred_year = extra_pretrain_start_year + (
        (global_tf_idx - 1) // extra_pretrain_timeframes_per_year
    )
    inferred_tf = ((global_tf_idx - 1) % extra_pretrain_timeframes_per_year) + 1

    if inferred_tf not in timeframe_to_months:
        raise ValueError(f"Inferred unsupported local timeframe {inferred_tf} from global index {global_tf_idx}")

    return inferred_year, inferred_tf


def infer_month_from_folder_name(folder_name: str, year: int):
    s = folder_name.strip().lower()

    if s in month_name_to_num:
        return month_name_to_num[s]

    if s.isdigit():
        value = int(s)

        if 1 <= value <= 12:
            return value

        if len(s) == 6 and s.startswith(str(year)):
            maybe_month = int(s[-2:])

            if 1 <= maybe_month <= 12:
                return maybe_month

    numbers = re.findall(r"\d+", s)

    if numbers:
        for i, num in enumerate(numbers):
            if num == str(year) and i + 1 < len(numbers):
                maybe_month = int(numbers[i + 1])

                if 1 <= maybe_month <= 12:
                    return maybe_month

        for num in reversed(numbers):
            maybe_month = int(num[-2:]) if len(num) > 2 else int(num)

            if 1 <= maybe_month <= 12:
                return maybe_month

    for key, value in month_name_to_num.items():
        if key in s:
            return value

    return None


def find_calendar_month_dir(era5_root: str, year: int, month: int) -> str:
    year_dir = os.path.join(era5_root, str(year))

    if not os.path.isdir(year_dir):
        raise FileNotFoundError(f"Could not find ERA5-Land data year folder: {year_dir}")

    direct_candidates = [
        os.path.join(year_dir, str(month)),
        os.path.join(year_dir, f"{month:02d}"),
        os.path.join(year_dir, f"M{month:02d}"),
        os.path.join(year_dir, f"month_{month:02d}"),
        os.path.join(year_dir, f"{year}_{month:02d}"),
        os.path.join(year_dir, f"{year}{month:02d}"),
    ]

    for p in direct_candidates:
        if os.path.isdir(p):
            return p

    subdirs = sorted([p for p in Path(year_dir).iterdir() if p.is_dir()], key=lambda p: p.name)

    for p in subdirs:
        if infer_month_from_folder_name(p.name, year) == month:
            return str(p)

    available = [p.name for p in subdirs]

    raise FileNotFoundError(
        f"Could not find ERA5-Land month folder for year={year}, month={month}. "
        f"Checked: {year_dir}. Available: {available}"
    )


def link_or_copy_month_dir(src_dir: str, dst_dir: str):
    dst_path = Path(dst_dir)

    if dst_path.exists() or dst_path.is_symlink():
        if dst_path.is_symlink() or dst_path.is_file():
            dst_path.unlink()
        elif dst_path.is_dir():
            shutil.rmtree(dst_path)

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        os.symlink(src_dir, dst_dir, target_is_directory=True)
    except Exception:
        shutil.copytree(src_dir, dst_dir)


def prepare_extra_era5_year_month_as_timeframes(
    era5_root: str,
    sentinel_manifest_path: str,
    cache_root: str,
) -> str:
    era5_root_path = Path(era5_root)
    sentinel_manifest = Path(sentinel_manifest_path)
    cache_root_path = Path(cache_root)

    if not era5_root_path.is_dir():
        raise FileNotFoundError(f"Extra ERA5-Land root does not exist: {era5_root}")

    if not sentinel_manifest.is_file():
        raise FileNotFoundError(f"Extra Sentinel manifest does not exist: {sentinel_manifest_path}")

    with open(sentinel_manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if not isinstance(manifest, list) or len(manifest) == 0:
        raise ValueError(f"Invalid or empty Sentinel manifest: {sentinel_manifest_path}")

    if cache_root_path.exists():
        shutil.rmtree(cache_root_path)

    cache_root_path.mkdir(parents=True, exist_ok=True)

    mapping = []

    for idx, item in enumerate(manifest, start=1):
        year, tf = parse_year_tf_from_manifest_item(item)

        if tf not in timeframe_to_months:
            raise ValueError(f"Unsupported timeframe_index={tf} for item {item}")

        months = timeframe_to_months[tf]
        tf_cache_dir = cache_root_path / f"T{idx:03d}"
        tf_cache_dir.mkdir(parents=True, exist_ok=True)

        for local_month_idx, calendar_month in enumerate(months, start=1):
            src_month_dir = find_calendar_month_dir(
                era5_root=era5_root,
                year=year,
                month=calendar_month,
            )
            dst_month_dir = tf_cache_dir / str(local_month_idx)

            link_or_copy_month_dir(
                src_dir=src_month_dir,
                dst_dir=str(dst_month_dir),
            )

        mapping.append(
            {
                "cache_timeframe": f"T{idx:03d}",
                "sentinel_group": item.get("group", item.get("id", "")),
                "year": year,
                "timeframe_index": tf,
                "calendar_months": months,
                "cache_dir": str(tf_cache_dir),
            }
        )

    mapping_path = cache_root_path / "mapping.json"

    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    return str(cache_root_path)


def build_pretrain_dataset(args):
    paired_datasets = []

    require_dir(args.sentinel_root_dir, "Base Sentinel root directory")
    require_file(args.sentinel_h5_file, "Base Sentinel H5")
    require_file(args.sentinel_data_file, "Base Sentinel manifest")
    require_dir(args.weather_root_dir, "Base ERA5-Land root directory")

    base_sentinel = Sentinel_Dataset(args.sentinel_root_dir, args.sentinel_data_file)
    base_weather = ERA5_Dataset(args.weather_root_dir)
    base_pair = PairedSentinelWeatherDataset(base_sentinel, base_weather, name="base_2021_2025")
    paired_datasets.append(base_pair)

    if args.use_extra_pretrain_data:
        require_dir(args.extra_sentinel_root_dir, "Extra Sentinel root directory")
        require_file(args.extra_sentinel_h5_file, "Extra Sentinel H5")
        require_file(args.extra_sentinel_data_file, "Extra Sentinel manifest")
        require_dir(args.extra_weather_root_dir, "Extra ERA5-Land root directory")

        extra_sentinel = Sentinel_Dataset(args.extra_sentinel_root_dir, args.extra_sentinel_data_file)

        extra_weather_root_for_loader = prepare_extra_era5_year_month_as_timeframes(
            era5_root=args.extra_weather_root_dir,
            sentinel_manifest_path=args.extra_sentinel_data_file,
            cache_root=args.extra_era5_timeframe_cache,
        )

        extra_weather = ERA5_Dataset(extra_weather_root_for_loader)
        extra_pair = PairedSentinelWeatherDataset(extra_sentinel, extra_weather, name="extra_2017_2021")
        paired_datasets.append(extra_pair)

    combined = paired_datasets[0] if len(paired_datasets) == 1 else torch.utils.data.ConcatDataset(paired_datasets)

    if len(combined) == 0:
        raise ValueError("Combined pretraining dataset is empty.")

    return combined


def main(args):
    with quiet():
        misc.init_distributed_mode(args)

    args.device = resolve_device(args.device)
    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    with quiet():
        dataset_pretrain = build_pretrain_dataset(args)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    sampler_pretrain = torch.utils.data.DistributedSampler(
        dataset_pretrain,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=True,
    )

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_pretrain = torch.utils.data.DataLoader(
        dataset_pretrain,
        sampler=sampler_pretrain,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        shuffle=False,
        drop_last=True,
    )

    model = PVTSimCLR(
        args.model,
        out_dim=args.embed_dim,
        context_dim=context_dim,
        pretrained=True,
    )
    model.to(device)

    model_without_ddp = model
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    if args.distributed:
        gpu_index = int(args.device.split(":")[1]) if args.device.startswith("cuda:") else 0

        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[gpu_index],
            find_unused_parameters=True,
        )
        model_without_ddp = model.module

    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    with quiet():
        misc.load_model(
            args=args,
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
        )

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_pretrain.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            data_loader_pretrain,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
        )

        if args.output_dir and (epoch % args.save_freq == 0 or epoch + 1 == args.epochs):
            with quiet():
                misc.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()

            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if misc.is_main_process():
            print(
                f"Epoch {epoch + 1}/{args.epochs} - "
                f"lr: {train_stats.get('lr', 0.0):.8f} | "
                f"loss: {train_stats.get('loss', 0.0):.6f}",
                flush=True,
            )


def train_one_epoch(
    model: torch.nn.Module,
    data_loader_pretrain: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):
    model.train(True)

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    batch_size = args.batch_size
    accum_iter = args.accum_iter

    optimizer.zero_grad()
    global_inner_step = 0

    for data_iter_step, (x, y) in enumerate(data_loader_pretrain):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer,
                data_iter_step / len(data_loader_pretrain) + epoch,
                args,
            )

        x_img = x[0].squeeze(1)
        y_short = y[0].squeeze(1)

        train_loader = sentinel_wrapper.get_data_loader(
            x_img,
            y_short,
            batch_size=batch_size,
        )

        for xi, xj, ys in train_loader:
            xi = xi.to(device, non_blocking=True)
            xj = xj.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)

            zi = model(xi, ys)
            zj = model(xj, ys)

            criterion = ContrastiveLoss(zi.shape[0], device)
            loss = criterion(zi, zj)
            loss_value = loss.item()

            if not math.isfinite(loss_value):
                raise ValueError(f"Loss is {loss_value}, stopping training")

            loss = loss / accum_iter
            update_grad = (global_inner_step + 1) % accum_iter == 0

            loss_scaler(
                loss,
                optimizer,
                parameters=model.parameters(),
                update_grad=update_grad,
            )

            if update_grad:
                optimizer.zero_grad()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            loss_value_reduce = misc.all_reduce_mean(loss_value)

            if log_writer is not None and update_grad:
                epoch_1000x = int((data_iter_step / len(data_loader_pretrain) + epoch) * 1000)
                log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
                log_writer.add_scalar("lr", lr, epoch_1000x)

            global_inner_step += 1

    optimizer.zero_grad()

    with quiet():
        metric_logger.synchronize_between_processes()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
