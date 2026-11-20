"""Real-time and historical data collection.

Data providers (all free, keyless):
  * Open-Meteo Flood API — GloFAS river discharge (past + 10 day forecast)
  * Open-Meteo Forecast API — ECMWF/ICON weather + soil moisture
  * Open-Meteo Archive API — ERA5-Land historical weather

The functions cache successful responses on disk. If the network is unavailable
the dashboard keeps working from cache, and only falls back to clearly-labelled
synthetic data as a last resort.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import zlib
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from .config import (
    FLASH_FORECAST_DAYS,
    HISTORY_CACHE_HOURS,
    HISTORY_DIR,
    HISTORY_LAG_DAYS,
    IST,
    LIVE_DIR,
    OPENMETEO_API_KEY,
    TRAIN_START_DATE,
    ensure_directories,
)
from .regions import Region

FLOOD_URL = "https://flood-api.open-meteo.com/v1/flood"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

FLOOD_DAILY = (
    "river_discharge,river_discharge_mean,river_discharge_median,"
    "river_discharge_max,river_discharge_min"
)
ARCHIVE_DAILY = (
    "precipitation_sum,temperature_2m_mean,temperature_2m_max,"
    "temperature_2m_min,et0_fao_evapotranspiration,wind_speed_10m_max,"
    "soil_moisture_0_to_100cm_mean"
)
LIVE_HOURLY = (
    "precipitation,temperature_2m,et0_fao_evapotranspiration,"
    "wind_speed_10m,soil_moisture_0_to_10cm,soil_moisture_10_to_40cm,"
    "soil_moisture_40_to_100cm"
)

DAILY_WEATHER_COLUMNS = [
    "precip",
    "temp_mean",
    "temp_max",
    "temp_min",
    "et0",
    "wind_max",
    "soil_moisture",
]


def _headers() -> dict[str, str]:
    return {"User-Agent": "FloodGuardIndia/1.0 (local research dashboard)"}


def _get_json(url: str, params: dict[str, Any], retries: int = 3) -> dict[str, Any]:
    if OPENMETEO_API_KEY:
        params = {**params, "apikey": OPENMETEO_API_KEY}
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with httpx.Client(timeout=90, follow_redirects=True, headers=_headers()) as client:
                response = client.get(url, params=params)
                if response.status_code == 429:
                    time.sleep(2 + attempt * 3)
                    continue
                response.raise_for_status()
                return response.json()
        except Exception as exc:  # noqa: BLE001 - network layer deliberately broad
            last_error = exc
            time.sleep(1.5 + attempt * 2)
    raise RuntimeError(f"Open-Meteo request failed: {last_error}") from last_error


def _cache_age_hours(path: Path) -> float:
    if not path.exists():
        return math.inf
    return (time.time() - path.stat().st_mtime) / 3600.0


def _history_paths(region: Region) -> tuple[Path, Path]:
    return HISTORY_DIR / f"{region.id}.csv", HISTORY_DIR / f"{region.id}.json"


def _live_paths(region: Region) -> tuple[Path, Path]:
    return LIVE_DIR / f"{region.id}.csv", LIVE_DIR / f"{region.id}.json"


def _read_cached(path: Path, meta_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(path, parse_dates=["date"])
    meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
    return df, meta


def _write_cached(df: pd.DataFrame, path: Path, meta_path: Path, meta: dict[str, Any]) -> None:
    ensure_directories()
    df.to_csv(path, index=False, date_format="%Y-%m-%d")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _standardise_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns, sort dates and fill only small gaps.

    Discharge is intentionally *not* filled over long gaps: it is the target
    source and large invented stretches would corrupt training labels.
    """
    if "date" not in df.columns:
        raise ValueError("Data frame has no date column")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
    numeric = [c for c in df.columns if c != "date"]
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if "discharge" not in df.columns:
        raise ValueError("Data frame has no discharge column")
    df["discharge"] = df["discharge"].interpolate(limit=2, limit_area="inside").ffill(limit=2)
    for column in ("precip",):
        if column in df.columns:
            df[column] = df[column].fillna(0.0).clip(lower=0)
    for column in ("temp_mean", "temp_max", "temp_min", "et0", "wind_max", "soil_moisture"):
        if column in df.columns:
            df[column] = df[column].interpolate(limit=3, limit_area="inside").ffill().bfill()
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Historical data used for model training
# ---------------------------------------------------------------------------

def fetch_history(region: Region, force: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    ensure_directories()
    csv_path, json_path = _history_paths(region)
    if not force and csv_path.exists() and _cache_age_hours(csv_path) < HISTORY_CACHE_HOURS:
        df, cached_meta = _read_cached(csv_path, json_path)
        cached_meta.update({"status": "cached", "region_id": region.id})
        return _standardise_frame(df), cached_meta

    end = datetime.now(IST).date() - timedelta(days=HISTORY_LAG_DAYS)
    start = date.fromisoformat(TRAIN_START_DATE)
    if end <= start:
        end = start + timedelta(days=365)
    try:
        flood = _get_json(
            FLOOD_URL,
            {
                "latitude": region.latitude,
                "longitude": region.longitude,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "daily": FLOOD_DAILY,
                "timezone": "Asia/Kolkata",
            },
        )
        weather = _get_json(
            ARCHIVE_URL,
            {
                "latitude": region.latitude,
                "longitude": region.longitude,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "daily": ARCHIVE_DAILY,
                "timezone": "Asia/Kolkata",
            },
        )

        flood_daily = flood.get("daily", {})
        weather_daily = weather.get("daily", {})
        flood_df = pd.DataFrame(
            {
                "date": flood_daily.get("time", []),
                "discharge": flood_daily.get("river_discharge", []),
                "discharge_mean": flood_daily.get("river_discharge_mean", []),
                "discharge_median": flood_daily.get("river_discharge_median", []),
                "discharge_max": flood_daily.get("river_discharge_max", []),
                "discharge_min": flood_daily.get("river_discharge_min", []),
            }
        )
        weather_df = pd.DataFrame(
            {
                "date": weather_daily.get("time", []),
                "precip": weather_daily.get("precipitation_sum", []),
                "temp_mean": weather_daily.get("temperature_2m_mean", []),
                "temp_max": weather_daily.get("temperature_2m_max", []),
                "temp_min": weather_daily.get("temperature_2m_min", []),
                "et0": weather_daily.get("et0_fao_evapotranspiration", []),
                "wind_max": weather_daily.get("wind_speed_10m_max", []),
                "soil_moisture": weather_daily.get("soil_moisture_0_to_100cm_mean", []),
            }
        )
        df = flood_df.merge(weather_df, on="date", how="inner")
        df = _standardise_frame(df)
        df = df.dropna(subset=["discharge"]).reset_index(drop=True)
        if len(df) < 500:
            raise RuntimeError(f"Only {len(df)} usable historical rows returned")

        meta = {
            "region_id": region.id,
            "fetched_at": datetime.now(IST).isoformat(timespec="seconds"),
            "start": str(df["date"].min().date()),
            "end": str(df["date"].max().date()),
            "rows": int(len(df)),
            "status": "live",
            "provider": "Open-Meteo Flood API (GloFAS) + ERA5-Land Archive",
            "grid_latitude": flood.get("latitude"),
            "grid_longitude": flood.get("longitude"),
        }
        _write_cached(df, csv_path, json_path, meta)
        return df, meta
    except Exception as exc:  # noqa: BLE001
        if csv_path.exists():
            df, cached_meta = _read_cached(csv_path, json_path)
            cached_meta.update(
                {
                    "status": "cached",
                    "region_id": region.id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return _standardise_frame(df), cached_meta
        df = synthetic_history(region, start, end)
        return df, {
            "region_id": region.id,
            "fetched_at": datetime.now(IST).isoformat(timespec="seconds"),
            "start": str(df["date"].min().date()),
            "end": str(df["date"].max().date()),
            "rows": int(len(df)),
            "status": "demo",
            "provider": "Synthetic fallback (data provider unreachable)",
            "error": f"{type(exc).__name__}: {exc}",
        }


# ---------------------------------------------------------------------------
# Live window used for inference
# ---------------------------------------------------------------------------

def fetch_live(region: Region, force: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    ensure_directories()
    csv_path, json_path = _live_paths(region)
    if not force and csv_path.exists() and _cache_age_hours(csv_path) < 0.1:
        df, meta = _read_cached(csv_path, json_path)
        meta.update({"status": "cached", "region_id": region.id})
        return _standardise_frame(df), meta

    today = datetime.now(IST).date()
    try:
        flood = _get_json(
            FLOOD_URL,
            {
                "latitude": region.latitude,
                "longitude": region.longitude,
                "daily": FLOOD_DAILY,
                "past_days": 92,
                "forecast_days": 10,
                "timezone": "Asia/Kolkata",
            },
        )
        weather = _get_json(
            FORECAST_URL,
            {
                "latitude": region.latitude,
                "longitude": region.longitude,
                "hourly": LIVE_HOURLY,
                "past_days": 92,
                "forecast_days": 10,
                "timezone": "Asia/Kolkata",
            },
        )
        hourly = weather.get("hourly", {})
        hdf = pd.DataFrame(
            {
                "time": hourly.get("time", []),
                "precip": hourly.get("precipitation", []),
                "temp": hourly.get("temperature_2m", []),
                "et0": hourly.get("et0_fao_evapotranspiration", []),
                "wind": hourly.get("wind_speed_10m", []),
                "sm0": hourly.get("soil_moisture_0_to_10cm", []),
                "sm10": hourly.get("soil_moisture_10_to_40cm", []),
                "sm100": hourly.get("soil_moisture_40_to_100cm", []),
            }
        )
        if hdf.empty:
            raise RuntimeError("Weather API returned no hourly rows")
        for column in ("precip", "temp", "et0", "wind", "sm0", "sm10", "sm100"):
            hdf[column] = pd.to_numeric(hdf[column], errors="coerce")
        hdf["date"] = pd.to_datetime(hdf["time"], errors="coerce").dt.normalize()
        hdf["soil"] = (
            0.1 * hdf["sm0"] + 0.3 * hdf["sm10"] + 0.6 * hdf["sm100"]
        )
        weather_daily = (
            hdf.dropna(subset=["date"])
            .groupby("date", as_index=False)
            .agg(
                precip=("precip", "sum"),
                temp_mean=("temp", "mean"),
                temp_max=("temp", "max"),
                temp_min=("temp", "min"),
                et0=("et0", "sum"),
                wind_max=("wind", "max"),
                soil_moisture=("soil", "mean"),
            )
        )

        flood_daily = flood.get("daily", {})
        fdf = pd.DataFrame(
            {
                "date": flood_daily.get("time", []),
                "discharge": flood_daily.get("river_discharge", []),
                "discharge_mean": flood_daily.get("river_discharge_mean", []),
                "discharge_median": flood_daily.get("river_discharge_median", []),
                "discharge_max": flood_daily.get("river_discharge_max", []),
                "discharge_min": flood_daily.get("river_discharge_min", []),
            }
        )
        fdf["date"] = pd.to_datetime(fdf["date"], errors="coerce").dt.normalize()
        df = weather_daily.merge(fdf, on="date", how="outer").sort_values("date")
        df = _standardise_frame(df)
        df = df[
            (df["date"] >= pd.Timestamp(today - timedelta(days=92)))
            & (df["date"] <= pd.Timestamp(today + timedelta(days=FLASH_FORECAST_DAYS - 1)))
        ]
        df = df.dropna(subset=["discharge"]).reset_index(drop=True)
        if len(df) < 40:
            raise RuntimeError(f"Only {len(df)} usable live rows returned")

        meta = {
            "region_id": region.id,
            "fetched_at": datetime.now(IST).isoformat(timespec="seconds"),
            "start": str(df["date"].min().date()),
            "end": str(df["date"].max().date()),
            "rows": int(len(df)),
            "status": "live",
            "provider": "Open-Meteo Forecast API + GloFAS Forecast",
            "grid_latitude": flood.get("latitude"),
            "grid_longitude": flood.get("longitude"),
        }
        _write_cached(df, csv_path, json_path, meta)
        return df, meta
    except Exception as exc:  # noqa: BLE001
        if csv_path.exists():
            df, meta = _read_cached(csv_path, json_path)
            meta.update(
                {
                    "status": "cached",
                    "region_id": region.id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return _standardise_frame(df), meta
        df = synthetic_live(region, today)
        return df, {
            "region_id": region.id,
            "fetched_at": datetime.now(IST).isoformat(timespec="seconds"),
            "start": str(df["date"].min().date()),
            "end": str(df["date"].max().date()),
            "rows": int(len(df)),
            "status": "demo",
            "provider": "Synthetic fallback (data provider unreachable)",
            "error": f"{type(exc).__name__}: {exc}",
        }


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------

_REGION_BASE = {
    "assam-brahmaputra": (4200.0, 34000.0, 0.85),
    "bihar-ganga": (1900.0, 32000.0, 0.80),
    "kerala-periyar": (110.0, 900.0, 1.00),
    "mumbai-ulhas": (4.0, 900.0, 1.15),
    "westbengal-hooghly": (220.0, 2100.0, 0.72),
}


def _synthetic_frame(region: Region, start: date, end: date) -> pd.DataFrame:
    """Plausible, reproducible monsoon data for offline demonstration."""
    seed = zlib.crc32(region.id.encode()) + start.toordinal()
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    doy = dates.dayofyear.to_numpy()
    # Broad monsoon hump peaking around late August, plus pre-monsoon bursts.
    monsoon = np.exp(-0.5 * ((doy - 225) / 48.0) ** 2)
    pre_monsoon = 0.25 * np.exp(-0.5 * ((doy - 135) / 22.0) ** 2)
    daily_rain = rng.gamma(shape=0.55, scale=9.0, size=len(dates))
    storms = (rng.random(len(dates)) < (0.03 + 0.21 * monsoon)) * rng.gamma(
        shape=1.9, scale=34.0, size=len(dates)
    )
    rain = (daily_rain * (0.25 + 1.5 * monsoon) + storms) * (1.0 + pre_monsoon)
    rain = np.clip(rain, 0, 380)
    base, peak, runoff = _REGION_BASE[region.id]
    rain_signal = pd.Series(rain).rolling(7, min_periods=1).sum().to_numpy()
    seasonal = 0.45 + 0.55 * monsoon
    discharge = base * seasonal + runoff * rain_signal * (peak / 500.0)
    discharge *= np.exp(rng.normal(0, 0.10, len(dates)))
    discharge = pd.Series(discharge).rolling(3, min_periods=1).mean().to_numpy()
    temperature = 25.5 + 4.5 * np.sin(2 * np.pi * (doy - 110) / 365.25) - 2.5 * monsoon
    soil = np.clip(
        0.18 + 0.22 * monsoon + 0.0011 * pd.Series(rain).rolling(10, min_periods=1).sum().to_numpy(),
        0.12,
        0.48,
    )
    return pd.DataFrame(
        {
            "date": dates,
            "discharge": discharge,
            "discharge_mean": discharge,
            "discharge_median": discharge,
            "discharge_max": discharge * 1.08,
            "discharge_min": discharge * 0.93,
            "precip": rain,
            "temp_mean": temperature,
            "temp_max": temperature + 5.0,
            "temp_min": temperature - 4.0,
            "et0": np.clip(5.2 - 2.4 * monsoon, 1.2, 7.0),
            "wind_max": 9 + 9 * monsoon + rng.gamma(1.4, 2.0, len(dates)),
            "soil_moisture": soil,
        }
    )


def synthetic_history(region: Region, start: date, end: date) -> pd.DataFrame:
    return _standardise_frame(_synthetic_frame(region, start, end))


def synthetic_live(region: Region, today: date) -> pd.DataFrame:
    end = today + timedelta(days=FLASH_FORECAST_DAYS - 1)
    start = today - timedelta(days=110)
    return _standardise_frame(_synthetic_frame(region, start, end))


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def describe_data_status(statuses: list[str]) -> str:
    if not statuses:
        return "unknown"
    if all(status == "live" for status in statuses):
        return "live"
    if any(status == "demo" for status in statuses):
        return "demo"
    if any(status == "cached" for status in statuses):
        return "cached"
    return "live"


def file_fingerprint(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha1(path.read_bytes()).hexdigest()[:10]
