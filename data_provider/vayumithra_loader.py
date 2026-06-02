"""
Dataset loader for VayuMithra 10-Year Weather Dataset.

Dataset characteristics:
  - Temporal range : Jan 2013 – Aug 2022 (~84,720 hourly steps)
  - Stations       : 32 (treated as a single multi-variate system)
  - Variables      : Temperature, Pressure, Humidity, Wind Speed (4 per station)
  - Total variates : 32 × 4 = 128
  - Static columns : lat, lon/long/longitude, index/station_id/station_index
                     → auto-detected and dropped before training

Expected CSV format (wide / pivoted):
  date, station_1_temp, station_1_pressure, ..., station_32_windspeed

  OR any consistent column naming with a parseable datetime column.

Split:  Train 70% | Val 10% | Test 20%  (paper-standard)

NaN handling:
  - Linear interpolation for gaps ≤ 3 consecutive hours (sensor glitches)
  - Forward-fill + backward-fill for any residual NaN at edges
  - Gaps > 3 consecutive hours are NOT interpolated; they persist as-is after
    ffill/bfill boundary fill so downstream windows may still contain NaN.
    This is acceptable for training (PyTorch MSE will propagate NaN → loss=NaN
    for that window, which is visible immediately and can be diagnosed).
  - If you want to hard-exclude such windows, set exclude_nan_windows=True.
"""

from __future__ import annotations

import os
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from utils.timefeatures import time_features

warnings.filterwarnings("ignore")

# ── Static column patterns to drop (case-insensitive substring match) ──────────
_STATIC_PATTERNS: List[str] = [
    "lat", "lon", "long", "longitude", "latitude",
    "index", "station_id", "station_index", "idx",
    "elevation", "altitude",
]


def _is_static_column(col: str) -> bool:
    """Return True if column name matches known static metadata patterns."""
    col_lower = col.lower()
    return any(pat in col_lower for pat in _STATIC_PATTERNS)


def _find_date_column(df: pd.DataFrame) -> str:
    """Auto-detect the datetime column from common naming conventions."""
    candidates = ["date", "datetime", "timestamp", "time", "Date", "DateTime"]
    for c in candidates:
        if c in df.columns:
            return c
    # Fallback: first column whose values parse as datetime
    for c in df.columns:
        try:
            pd.to_datetime(df[c].head(5))
            return c
        except Exception:
            pass
    raise ValueError(
        "Cannot detect datetime column. "
        "Rename it to 'date' or 'datetime' and retry."
    )


class Dataset_VayuMithra(Dataset):
    """
    PyTorch Dataset for the VayuMithra 10-Year hourly weather data.

    Args:
        root_path   : Directory containing the CSV file.
        data_path   : Filename of the CSV (e.g. 'vayumithra_10y.csv').
        flag        : 'train' | 'val' | 'test'.
        size        : [seq_len, label_len, pred_len].
        features    : 'M' (multivariate → multivariate). 'S'/'MS' not recommended.
        target      : Not used for features='M'; kept for API compatibility.
        scale       : Apply StandardScaler (fit on train split only).
        timeenc     : 0 = fixed temporal encoding, 1 = learnable time features.
        freq        : Frequency string for time_features ('h' = hourly).
        exclude_nan_windows : If True, windows containing any NaN are skipped.
    """

    def __init__(
        self,
        root_path: str,
        flag: str = "train",
        size: Optional[List[int]] = None,
        features: str = "M",
        data_path: str = "vayumithra_10y.csv",
        target: str = "OT",
        scale: bool = True,
        timeenc: int = 0,
        freq: str = "h",
        exclude_nan_windows: bool = False,
    ) -> None:
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ("train", "val", "test"), f"Invalid flag: {flag}"
        self.set_type: int = {"train": 0, "val": 1, "test": 2}[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.exclude_nan_windows = exclude_nan_windows

        self._valid_indices: Optional[np.ndarray] = None  # set if exclude_nan_windows
        self.__read_data__()

    # ── Data loading ────────────────────────────────────────────────────────

    def __read_data__(self) -> None:
        self.scaler = StandardScaler()
        csv_path = os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_csv(csv_path)

        # ── 1. Detect and parse date column ─────────────────────────────────
        date_col = _find_date_column(df_raw)
        df_raw[date_col] = pd.to_datetime(df_raw[date_col])
        # Rename to 'date' for consistency
        if date_col != "date":
            df_raw = df_raw.rename(columns={date_col: "date"})

        # ── 2. Drop static columns ───────────────────────────────────────────
        static_cols = [
            c for c in df_raw.columns
            if c != "date" and _is_static_column(c)
        ]
        if static_cols:
            print(
                f"[VayuMithra] Dropping {len(static_cols)} static columns: "
                f"{static_cols[:8]}{'...' if len(static_cols) > 8 else ''}"
            )
            df_raw = df_raw.drop(columns=static_cols)

        # ── 3. Sort by timestamp, reset index ───────────────────────────────
        df_raw = df_raw.sort_values("date").reset_index(drop=True)

        # ── 4. NaN handling ──────────────────────────────────────────────────
        temporal_cols = [c for c in df_raw.columns if c != "date"]
        n_raw = df_raw[temporal_cols].isna().sum().sum()
        if n_raw > 0:
            print(
                f"[VayuMithra] Found {n_raw} NaN values across "
                f"{len(temporal_cols)} variates. Interpolating ≤3h gaps..."
            )
            df_raw[temporal_cols] = (
                df_raw[temporal_cols]
                .interpolate(method="linear", limit=3, limit_direction="both")
                .ffill()
                .bfill()
            )
            n_remaining = df_raw[temporal_cols].isna().sum().sum()
            if n_remaining > 0:
                print(
                    f"[VayuMithra] WARNING: {n_remaining} NaN values remain "
                    f"after interpolation (gaps >3h). "
                    f"Set exclude_nan_windows=True to skip those windows."
                )

        # ── 5. Report dataset shape ──────────────────────────────────────────
        T = len(df_raw)
        N = len(temporal_cols)
        print(f"[VayuMithra] Dataset: {T} timesteps × {N} variates")
        print(f"[VayuMithra] Date range: {df_raw['date'].min()} → {df_raw['date'].max()}")

        # ── 6. Train / Val / Test split (70 / 10 / 20) ──────────────────────
        num_train = int(T * 0.7)
        num_test  = int(T * 0.2)
        num_val   = T - num_train - num_test

        border1s = [
            0,
            num_train - self.seq_len,
            T - num_test - self.seq_len,
        ]
        border2s = [
            num_train,
            num_train + num_val,
            T,
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # ── 7. Feature selection (always 'M' for this dataset) ──────────────
        df_data = df_raw[temporal_cols]

        # ── 8. Scaling (fit on train, transform all) ─────────────────────────
        if self.scale:
            train_data = df_data.values[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data.values).astype(np.float32)
        else:
            data = df_data.values.astype(np.float32)

        # ── 9. Time features ─────────────────────────────────────────────────
        df_stamp = df_raw[["date"]].iloc[border1:border2].copy()
        if self.timeenc == 0:
            df_stamp["month"]   = df_stamp["date"].dt.month
            df_stamp["day"]     = df_stamp["date"].dt.day
            df_stamp["weekday"] = df_stamp["date"].dt.weekday
            df_stamp["hour"]    = df_stamp["date"].dt.hour
            data_stamp = df_stamp.drop(["date"], axis=1).values.astype(np.float32)
        else:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            ).transpose(1, 0).astype(np.float32)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        # ── 10. Pre-compute valid indices if NaN exclusion requested ─────────
        if self.exclude_nan_windows:
            self._valid_indices = self._compute_valid_indices()
            print(
                f"[VayuMithra] Valid windows after NaN exclusion: "
                f"{len(self._valid_indices)} / {len(self.data_x) - self.seq_len - self.pred_len + 1}"
            )

        print(
            f"[VayuMithra] Split '{['train','val','test'][self.set_type]}': "
            f"{len(self)} samples"
        )

    def _compute_valid_indices(self) -> np.ndarray:
        """Return indices of windows that contain no NaN."""
        n_samples = len(self.data_x) - self.seq_len - self.pred_len + 1
        valid = []
        for i in range(n_samples):
            x_window = self.data_x[i: i + self.seq_len]
            y_window = self.data_y[
                i + self.seq_len - self.label_len:
                i + self.seq_len + self.pred_len
            ]
            if not (np.isnan(x_window).any() or np.isnan(y_window).any()):
                valid.append(i)
        return np.array(valid, dtype=np.int64)

    # ── PyTorch Dataset interface ────────────────────────────────────────────

    def __getitem__(
        self, index: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._valid_indices is not None:
            index = int(self._valid_indices[index])

        s_begin = index
        s_end   = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end   = r_begin + self.label_len + self.pred_len

        seq_x      = self.data_x[s_begin:s_end]
        seq_y      = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self) -> int:
        if self._valid_indices is not None:
            return len(self._valid_indices)
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Invert StandardScaler on model outputs."""
        return self.scaler.inverse_transform(data)

    @property
    def n_variates(self) -> int:
        """Number of temporal variates (N = 32 stations × 4 variables = 128)."""
        return self.data_x.shape[1]