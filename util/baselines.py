from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader, TensorDataset

from util.metrics import regression_report


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def as_1d(x) -> np.ndarray:
    x = to_numpy(x).astype(np.float64).reshape(-1)
    if not np.isfinite(x).all():
        raise ValueError("target contains non-finite values")
    return x


def as_2d(x) -> np.ndarray:
    x = to_numpy(x).astype(np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    if not np.isfinite(x).all():
        raise ValueError("features contain non-finite values")
    return x


def mean_baseline(train_targets, val_targets) -> Dict[str, float]:
    train_targets = as_1d(train_targets)
    val_targets = as_1d(val_targets)
    pred = np.full_like(val_targets, float(train_targets.mean()), dtype=np.float64)
    return regression_report(val_targets, pred)


def standardize_features(train_x, val_x) -> Tuple[np.ndarray, np.ndarray]:
    train_x = as_2d(train_x)
    val_x = as_2d(val_x)
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (train_x - mean) / std, (val_x - mean) / std


def linear_weather_baseline(
    train_weather,
    train_target,
    val_weather,
    val_target,
    ridge_lambda: float = 1.0,
) -> Dict[str, float]:
    x_train, x_val = standardize_features(train_weather, val_weather)
    y_train = as_1d(train_target)
    y_val = as_1d(val_target)

    x_train = np.concatenate([np.ones((x_train.shape[0], 1)), x_train], axis=1)
    x_val = np.concatenate([np.ones((x_val.shape[0], 1)), x_val], axis=1)

    eye = np.eye(x_train.shape[1], dtype=np.float64)
    eye[0, 0] = 0.0

    beta = np.linalg.solve(
        x_train.T @ x_train + float(ridge_lambda) * eye,
        x_train.T @ y_train,
    )

    pred = x_val @ beta
    return regression_report(y_val, pred)


def pooled_features(*arrays) -> np.ndarray:
    out = []

    for arr in arrays:
        x = to_numpy(arr).astype(np.float64)

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        if x.ndim == 2:
            out.append(x)
        else:
            flat = x.reshape(x.shape[0], -1)
            out.append(
                np.stack(
                    [
                        flat.mean(axis=1),
                        flat.std(axis=1),
                        flat.min(axis=1),
                        flat.max(axis=1),
                    ],
                    axis=1,
                )
            )

    if not out:
        raise ValueError("no feature arrays provided")

    features = np.concatenate(out, axis=1)

    if not np.isfinite(features).all():
        raise ValueError("pooled features contain non-finite values")

    return features


def random_forest_baseline(
    train_features,
    train_target,
    val_features,
    val_target,
    seed: int = 42,
    n_estimators: int = 500,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 2,
    max_features: str | float | int | None = "sqrt",
    n_jobs: int = -1,
) -> Dict[str, float]:
    from sklearn.ensemble import RandomForestRegressor

    x_train = as_2d(train_features)
    x_val = as_2d(val_features)
    y_train = as_1d(train_target)
    y_val = as_1d(val_target)

    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=max_depth,
        min_samples_leaf=int(min_samples_leaf),
        max_features=max_features,
        random_state=int(seed),
        n_jobs=int(n_jobs),
    )

    model.fit(x_train, y_train)
    pred = model.predict(x_val).astype(np.float64)

    return regression_report(y_val, pred)


def sentinel_to_nchw(x) -> np.ndarray:
    x = to_numpy(x).astype(np.float32)

    if x.ndim == 6:
        if x.shape[3] <= 16 and x.shape[-1] > 16:
            n, t, g, c, h, w = x.shape
            return x.reshape(n, t * g * c, h, w)

        if x.shape[-1] <= 16 and x.shape[3] > 16:
            x = np.transpose(x, (0, 1, 2, 5, 3, 4))
            n, t, g, c, h, w = x.shape
            return x.reshape(n, t * g * c, h, w)

    if x.ndim == 5:
        if x.shape[2] <= 16 and x.shape[-1] > 16:
            n, g, c, h, w = x.shape
            return x.reshape(n, g * c, h, w)

        if x.shape[-1] <= 16 and x.shape[2] > 16:
            x = np.transpose(x, (0, 1, 4, 2, 3))
            n, g, c, h, w = x.shape
            return x.reshape(n, g * c, h, w)

    if x.ndim == 4:
        if x.shape[1] <= 256 and x.shape[-1] > 16:
            return x

        if x.shape[-1] <= 256 and x.shape[1] > 16:
            return np.transpose(x, (0, 3, 1, 2))

    raise ValueError(f"unsupported sentinel shape: {x.shape}")


def standardize_images(train_x, val_x) -> Tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=(0, 2, 3), keepdims=True)
    std = train_x.std(axis=(0, 2, 3), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_x - mean) / std, (val_x - mean) / std


class SmallCNNRegressor(nn.Module):
    def __init__(self, in_channels: int, dropout: float = 0.2):
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
            nn.Dropout(float(dropout)),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def cnn_sentinel_baseline(
    train_sentinel,
    train_target,
    val_sentinel,
    val_target,
    seed: int = 42,
    epochs: int = 60,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.2,
    patience: int = 12,
    device: Optional[str] = None,
) -> Dict[str, float]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    x_train = sentinel_to_nchw(train_sentinel)
    x_val = sentinel_to_nchw(val_sentinel)
    x_train, x_val = standardize_images(x_train, x_val)

    y_train = as_1d(train_target)
    y_val = as_1d(val_target)

    target_mean = float(y_train.mean())
    target_std = float(y_train.std())

    if target_std < 1e-8:
        target_std = 1.0

    y_train_norm = ((y_train - target_mean) / target_std).astype(np.float32)
    y_val_norm = ((y_val - target_mean) / target_std).astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train.astype(np.float32)),
            torch.from_numpy(y_train_norm),
        ),
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device_obj = torch.device(device)

    model = SmallCNNRegressor(
        in_channels=int(x_train.shape[1]),
        dropout=float(dropout),
    ).to(device_obj)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )

    val_x = torch.from_numpy(x_val.astype(np.float32)).to(device_obj)
    val_y = torch.from_numpy(y_val_norm.astype(np.float32)).to(device_obj)

    best_state = None
    best_loss = float("inf")
    bad_epochs = 0

    for _ in range(int(epochs)):
        model.train()

        for xb, yb in train_loader:
            xb = xb.to(device_obj)
            yb = yb.to(device_obj)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = f.smooth_l1_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()

        with torch.no_grad():
            val_pred = model(val_x)
            val_loss = float(f.smooth_l1_loss(val_pred, val_y).detach().cpu().item())

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= int(patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    with torch.no_grad():
        pred_norm = model(val_x).detach().cpu().numpy().astype(np.float64)

    pred = pred_norm * target_std + target_mean

    return regression_report(y_val, pred)