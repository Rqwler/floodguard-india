"""Feature engineering shared by training and live inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import SEQUENCE_LENGTH

FEATURES = [
    "discharge_log",
    "discharge_delta_1",
    "discharge_delta_3",
    "discharge_delta_7",
    "discharge_ratio_30",
    "discharge_mean_7_log",
    "discharge_std_7_log",
    "rain_1",
    "rain_3",
    "rain_7",
    "rain_14",
    "rain_30",
    "soil_moisture",
    "soil_anomaly_30",
    "temp_mean",
    "et0",
    "wind_max",
    "season_sin",
    "season_cos",
]

FEATURE_LABELS = {
    "discharge_log": "Log river discharge",
    "discharge_delta_1": "1-day discharge change",
    "discharge_delta_3": "3-day discharge change",
    "discharge_delta_7": "7-day discharge change",
    "discharge_ratio_30": "Discharge vs 30-day median",
    "discharge_mean_7_log": "7-day mean discharge",
    "discharge_std_7_log": "7-day discharge volatility",
    "rain_1": "Rainfall, last 24 h",
    "rain_3": "Rainfall, last 3 days",
    "rain_7": "Rainfall, last 7 days",
    "rain_14": "Rainfall, last 14 days",
    "rain_30": "Rainfall, last 30 days",
    "soil_moisture": "Root-zone soil moisture",
    "soil_anomaly_30": "Soil moisture anomaly",
    "temp_mean": "Mean temperature",
    "et0": "Evapotranspiration",
    "wind_max": "Max wind speed",
    "season_sin": "Seasonality (sin)",
    "season_cos": "Seasonality (cos)",
}


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create causal (backward-looking) hydrological features.

    The returned frame also contains ``target_discharge`` (next day's observed
    discharge) so training can derive labels. Live inference simply ignores it.
    """
    if df.empty:
        raise ValueError("Cannot build features from an empty frame")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    # Make the calendar continuous so rolling windows represent days, not rows.
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_index)
    df.index.name = "date"

    df["discharge"] = pd.to_numeric(df["discharge"], errors="coerce").interpolate(
        limit=2, limit_area="inside"
    ).ffill(limit=2)
    for column in ("precip", "temp_mean", "temp_max", "temp_min", "et0", "wind_max", "soil_moisture"):
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["precip"] = df["precip"].fillna(0.0).clip(lower=0)
    for column in ("temp_mean", "et0", "wind_max", "soil_moisture"):
        df[column] = df[column].interpolate(limit=3, limit_area="inside").ffill().bfill()

    discharge = df["discharge"].clip(lower=0)
    discharge_log = np.log1p(discharge)

    df["discharge_log"] = discharge_log
    for horizon in (1, 3, 7):
        previous = discharge.shift(horizon)
        df[f"discharge_delta_{horizon}"] = (
            (discharge - previous) / (previous + 1e-3)
        ).clip(-1.5, 5.0)
    rolling_median_30 = discharge.rolling(30, min_periods=10).median()
    df["discharge_ratio_30"] = (discharge / (rolling_median_30 + 1e-3)).clip(0, 12)
    df["discharge_mean_7_log"] = discharge_log.rolling(7, min_periods=3).mean()
    df["discharge_std_7_log"] = discharge_log.rolling(7, min_periods=3).std()

    rain = df["precip"]
    for window in (1, 3, 7, 14, 30):
        df[f"rain_{window}"] = rain.rolling(window, min_periods=1).sum()

    soil = df["soil_moisture"]
    df["soil_anomaly_30"] = soil - soil.rolling(30, min_periods=10).mean()

    day_of_year = df.index.day_of_year.to_numpy()
    df["season_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["season_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    df["target_discharge"] = discharge.shift(-1)
    df = df.reset_index()
    # Initial rolling windows are warm-up; later gaps are filled conservatively.
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return df


def compute_thresholds(discharge_values: pd.Series | np.ndarray) -> dict[str, float]:
    values = np.asarray(discharge_values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 30:
        raise ValueError("Not enough discharge observations to compute thresholds")
    quantiles = {
        "median": 0.50,
        "q80": 0.80,
        "q90": 0.90,
        "q92": 0.92,
        "q95": 0.95,
        "q97": 0.97,
        "q99": 0.99,
    }
    thresholds: dict[str, float] = {}
    last = 0.0
    for name, q in quantiles.items():
        value = float(np.quantile(values, q))
        value = max(value, last + 1e-4)
        thresholds[name] = round(value, 4)
        last = value
    thresholds["watch"] = thresholds["q80"]
    thresholds["warning"] = thresholds["q92"]
    thresholds["severe"] = thresholds["q97"]
    return thresholds


def add_labels(df: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    target = df["target_discharge"]
    conditions = [
        target < thresholds["watch"],
        (target >= thresholds["watch"]) & (target < thresholds["warning"]),
        (target >= thresholds["warning"]) & (target < thresholds["severe"]),
        target >= thresholds["severe"],
    ]
    df = df.copy()
    df["label"] = np.select(conditions, [0, 1, 2, 3], default=np.nan)
    df.loc[target.isna(), "label"] = np.nan
    return df


def fit_scaler(feature_frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    scaler: dict[str, dict[str, float]] = {}
    for feature in FEATURES:
        values = pd.to_numeric(feature_frame[feature], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            center, scale = 0.0, 1.0
        else:
            center = float(np.median(values))
            q25, q75 = np.quantile(values, [0.25, 0.75])
            scale = float(q75 - q25)
            if not np.isfinite(scale) or scale < 1e-8:
                scale = float(np.std(values))
            if not np.isfinite(scale) or scale < 1e-8:
                scale = 1.0
        scaler[feature] = {"center": center, "scale": scale}
    return scaler


def scale_features(df: pd.DataFrame, scaler: dict[str, dict[str, float]]) -> np.ndarray:
    matrix = np.empty((len(df), len(FEATURES)), dtype=np.float32)
    for index, feature in enumerate(FEATURES):
        stats = scaler.get(feature, {"center": 0.0, "scale": 1.0})
        values = pd.to_numeric(df[feature], errors="coerce").to_numpy(dtype=float)
        values = (values - stats["center"]) / (stats["scale"] + 1e-9)
        values = np.clip(values, -10, 10)
        values[~np.isfinite(values)] = 0.0
        matrix[:, index] = values.astype(np.float32)
    return matrix


def build_sequence_dataset(
    feature_df: pd.DataFrame,
    scaler: dict[str, dict[str, float]],
    region_index: int,
    sequence_length: int = SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[pd.Timestamp]]:
    """Create (samples, sequence, features), labels and target dates.

    A sequence ending on day ``t`` predicts the discharge class on day ``t+1``.
    """
    scaled = scale_features(feature_df, scaler)
    labels = feature_df["label"].to_numpy()
    dates = pd.to_datetime(feature_df["date"]).to_numpy()
    samples, targets, regions, target_dates = [], [], [], []
    for end_index in range(sequence_length - 1, len(feature_df)):
        label = labels[end_index]
        if not np.isfinite(label):
            continue
        samples.append(scaled[end_index - sequence_length + 1 : end_index + 1])
        targets.append(int(label))
        regions.append(region_index)
        target_dates.append(pd.Timestamp(dates[end_index]) + pd.Timedelta(days=1))
    if not samples:
        return (
            np.empty((0, sequence_length, len(FEATURES)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            [],
        )
    return (
        np.stack(samples).astype(np.float32),
        np.asarray(targets, dtype=np.int64),
        np.asarray(regions, dtype=np.int64),
        target_dates,
    )


def risk_level_from_discharge(discharge: float, thresholds: dict[str, float]) -> int:
    if not np.isfinite(discharge):
        return 0
    if discharge >= thresholds["severe"]:
        return 3
    if discharge >= thresholds["warning"]:
        return 2
    if discharge >= thresholds["watch"]:
        return 1
    return 0


def summarise_feature_health(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(df)),
        "date_start": str(pd.to_datetime(df["date"]).min().date()) if len(df) else None,
        "date_end": str(pd.to_datetime(df["date"]).max().date()) if len(df) else None,
        "missing_values": int(df[FEATURES].isna().sum().sum()) if len(df) else 0,
    }
