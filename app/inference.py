"""Live inference: detection of the current state plus an 8-day risk outlook."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    ALERT_HORIZON_DAYS,
    CHART_HISTORY_DAYS,
    HYBRID_MODEL_WEIGHT,
    IST,
    RISK_COLORS,
    RISK_LABELS,
    RISK_SCORE_ANCHORS,
    RISK_SCORE_BANDS,
    SEQUENCE_LENGTH,
)
from .features import (
    FEATURES,
    prepare_features,
    risk_level_from_discharge,
    scale_features,
)
from .model import load_model_bundle, predict_probabilities
from .regions import Region, REGION_INDEX

PROBABILITY_SCORE = np.asarray(RISK_SCORE_ANCHORS, dtype=float)


def _round(value: Any, digits: int = 2) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return round(number, digits)


def _score_from_discharge(discharge: float, thresholds: dict[str, float]) -> float:
    """Transparent 0-100 score for the *observed* discharge."""
    if not np.isfinite(discharge) or discharge <= 0:
        return 0.0
    watch = thresholds["watch"]
    warning = thresholds["warning"]
    severe = thresholds["severe"]
    if discharge >= severe:
        excess = min(1.0, (discharge - severe) / (severe * 0.8 + 1e-6))
        return round(90 + 10 * excess, 2)
    if discharge >= warning:
        return round(70 + 20 * (discharge - warning) / (severe - warning + 1e-9), 2)
    if discharge >= watch:
        return round(35 + 35 * (discharge - watch) / (warning - watch + 1e-9), 2)
    return round(35 * discharge / (watch + 1e-9), 2)


def _heuristic_probabilities(discharge: float, thresholds: dict[str, float]) -> np.ndarray:
    level = risk_level_from_discharge(discharge, thresholds)
    probabilities = np.full(4, 0.04, dtype=float)
    probabilities[level] = 0.82
    if level > 0:
        probabilities[level - 1] = 0.12
    else:
        probabilities[1] = 0.10
    if level < 3:
        probabilities[level + 1] = 0.02
    return probabilities / probabilities.sum()


def _hydrology_prior(discharge: float, thresholds: dict[str, float]) -> np.ndarray:
    """Smooth probability prior from GloFAS discharge relative to thresholds.

    This acts as a hydrological guardrail around the LSTM, keeping the
    operational forecast physically anchored to the river signal.
    """
    if not np.isfinite(discharge) or discharge <= 0:
        return np.array([0.97, 0.03, 0.0, 0.0], dtype=float)
    watch = max(thresholds["watch"], 1e-6)
    warning = max(thresholds["warning"], watch + 1e-6)
    severe = max(thresholds["severe"], warning + 1e-6)
    if discharge >= severe:
        probabilities = np.array([0.01, 0.03, 0.20, 0.76], dtype=float)
    elif discharge >= warning:
        fraction = (discharge - warning) / (severe - warning + 1e-9)
        probabilities = np.array(
            [0.01, 0.04, 0.70 - 0.22 * fraction, 0.25 + 0.22 * fraction],
            dtype=float,
        )
    elif discharge >= watch:
        fraction = (discharge - watch) / (warning - watch + 1e-9)
        probabilities = np.array(
            [0.08 * (1 - fraction), 0.62, 0.28 + 0.08 * fraction, 0.02 + 0.04 * fraction],
            dtype=float,
        )
    else:
        fraction = min(1.0, discharge / watch)
        probabilities = np.array(
            [0.94 - 0.30 * fraction, 0.05 + 0.22 * fraction, 0.01 + 0.07 * fraction, 0.01 * fraction],
            dtype=float,
        )
    return probabilities / probabilities.sum()


def _level_from_score(score: float) -> int:
    if score >= RISK_SCORE_BANDS[2]:
        return 3
    if score >= RISK_SCORE_BANDS[1]:
        return 2
    if score >= RISK_SCORE_BANDS[0]:
        return 1
    return 0


def _probability_payload(
    probabilities: np.ndarray,
    *,
    model_probabilities: np.ndarray | None = None,
    hydrology_probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / max(probabilities.sum(), 1e-9)
    score = float(np.dot(probabilities, PROBABILITY_SCORE))
    level = _level_from_score(score)
    model_level = int(np.argmax(model_probabilities)) if model_probabilities is not None else None
    model_score = (
        float(np.dot(np.asarray(model_probabilities, dtype=float), PROBABILITY_SCORE))
        if model_probabilities is not None
        else None
    )
    hydrology_score = (
        float(np.dot(np.asarray(hydrology_probabilities, dtype=float), PROBABILITY_SCORE))
        if hydrology_probabilities is not None
        else None
    )
    return {
        "level": level,
        "label": RISK_LABELS[level],
        "short_label": RISK_LABELS[level].upper(),
        "color": RISK_COLORS[level],
        "risk_score": round(score, 1),
        "confidence": round(float(probabilities[level] * 100), 1),
        "alert_probability": round(float(probabilities[2:].sum() * 100), 1),
        "probabilities": {
            RISK_LABELS[index]: round(float(value * 100), 1)
            for index, value in enumerate(probabilities)
        },
        "model_level": model_level,
        "model_risk_score": round(model_score, 1) if model_score is not None else None,
        "hydrology_risk_score": round(hydrology_score, 1) if hydrology_score is not None else None,
        "model_probabilities": (
            {
                RISK_LABELS[index]: round(float(value * 100), 1)
                for index, value in enumerate(model_probabilities)
            }
            if model_probabilities is not None
            else None
        ),
    }


def _driver(label: str, value: str, detail: str, impact: str) -> dict[str, str]:
    return {"label": label, "value": value, "detail": detail, "impact": impact}


class FloodPredictor:
    def __init__(self) -> None:
        self.bundle = load_model_bundle()
        self.metadata = self.bundle["metadata"] if self.bundle else None
        self.model_ready = self.bundle is not None

    def reload(self) -> None:
        self.bundle = load_model_bundle()
        self.metadata = self.bundle["metadata"] if self.bundle else None
        self.model_ready = self.bundle is not None

    @property
    def thresholds(self) -> dict[str, dict[str, float]]:
        if not self.metadata:
            return {}
        return self.metadata.get("thresholds", {})

    @property
    def scalers(self) -> dict[str, dict[str, dict[str, float]]]:
        if not self.metadata:
            return {}
        return self.metadata.get("scalers", {})

    def predict_region(self, region: Region, raw_df: pd.DataFrame) -> dict[str, Any]:
        features = prepare_features(raw_df)
        thresholds = self.thresholds.get(region.id) or self._fallback_thresholds(features)
        today = pd.Timestamp(datetime.now(IST).date())
        latest_index = features.index[features["date"] <= today]
        if len(latest_index) == 0:
            latest_index = features.index
        latest_position = int(latest_index[-1])
        latest_row = features.iloc[latest_position]
        observed_level = risk_level_from_discharge(float(latest_row["discharge"]), thresholds)
        observed_score = _score_from_discharge(float(latest_row["discharge"]), thresholds)

        # One-day-ahead history for the chart and eight forecast days.
        history_targets = [
            today - pd.Timedelta(days=offset) for offset in range(CHART_HISTORY_DAYS - 1, -1, -1)
        ]
        forecast_targets = [today + pd.Timedelta(days=offset) for offset in range(1, 9)]
        target_dates = history_targets + forecast_targets

        feature_dates = pd.to_datetime(features["date"]).reset_index(drop=True)
        date_to_position = {timestamp.normalize(): index for index, timestamp in enumerate(feature_dates)}
        scaled = scale_features(features, self.scalers.get(region.id, {})) if self.model_ready else None

        sequences: list[np.ndarray] = []
        valid_targets: list[pd.Timestamp] = []
        for target in target_dates:
            sequence_end = target - pd.Timedelta(days=1)
            position = date_to_position.get(sequence_end.normalize())
            if position is None or position < SEQUENCE_LENGTH - 1:
                continue
            if scaled is not None:
                sequences.append(scaled[position - SEQUENCE_LENGTH + 1 : position + 1])
            else:
                sequences.append(np.zeros((SEQUENCE_LENGTH, len(FEATURES)), dtype=np.float32))
            valid_targets.append(target)

        if self.model_ready and sequences:
            model_probabilities = predict_probabilities(
                self.bundle,
                np.stack(sequences),
                np.full(len(sequences), REGION_INDEX[region.id], dtype=np.int64),
            )
        else:
            model_probabilities = None

        prediction_map: dict[pd.Timestamp, dict[str, Any]] = {}
        for index, target in enumerate(valid_targets):
            target_discharge = self._discharge_on_date(
                features, target, float(latest_row["discharge"])
            )
            hydrology_prior = _hydrology_prior(target_discharge, thresholds)
            if model_probabilities is not None:
                blended = (
                    HYBRID_MODEL_WEIGHT * model_probabilities[index]
                    + (1.0 - HYBRID_MODEL_WEIGHT) * hydrology_prior
                )
                payload = _probability_payload(
                    blended,
                    model_probabilities=model_probabilities[index],
                    hydrology_probabilities=hydrology_prior,
                )
            else:
                payload = _probability_payload(
                    hydrology_prior,
                    hydrology_probabilities=hydrology_prior,
                )
            prediction_map[target.normalize()] = payload

        fallback_prior = _hydrology_prior(float(latest_row["discharge"]), thresholds)
        fallback_payload = _probability_payload(fallback_prior, hydrology_probabilities=fallback_prior)
        forecast = []
        for target in forecast_targets:
            payload = prediction_map.get(target.normalize(), fallback_payload).copy()
            payload["date"] = target.date().isoformat()
            row = self._row_on_date(features, target)
            payload["discharge"] = _round(row.get("discharge")) if row is not None else None
            payload["rain"] = _round(row.get("precip")) if row is not None else None
            payload["is_forecast"] = True
            forecast.append(payload)

        series = []
        for offset in range(CHART_HISTORY_DAYS - 1, -1, -1):
            target = today - pd.Timedelta(days=offset)
            row = self._row_on_date(features, target)
            risk = prediction_map.get(target.normalize(), fallback_payload)
            series.append(
                {
                    "date": target.date().isoformat(),
                    "discharge": _round(row.get("discharge")) if row is not None else None,
                    "rain": _round(row.get("precip")) if row is not None else None,
                    "risk_score": risk["risk_score"],
                    "level": risk["level"],
                    "is_forecast": False,
                }
            )
        for item in forecast:
            row = self._row_on_date(features, pd.Timestamp(item["date"]))
            series.append(
                {
                    "date": item["date"],
                    "discharge": _round(row.get("discharge")) if row is not None else item["discharge"],
                    "rain": _round(row.get("precip")) if row is not None else item["rain"],
                    "risk_score": item["risk_score"],
                    "level": item["level"],
                    "is_forecast": True,
                }
            )

        active_horizon = forecast[:ALERT_HORIZON_DAYS]
        peak = max(forecast, key=lambda item: (item["level"], item["risk_score"]))
        next_24h = forecast[0]
        trend_delta = next_24h["risk_score"] - observed_score
        if trend_delta > 5:
            trend = "rising"
        elif trend_delta < -5:
            trend = "easing"
        else:
            trend = "steady"

        active_level = max(observed_level, max(item["level"] for item in active_horizon))
        active_candidates = [item for item in active_horizon if item["level"] == active_level]
        if observed_level == active_level:
            active_score = observed_score
            active_date = today.date().isoformat()
            active_source = "observed"
            active_label = RISK_LABELS[observed_level]
        else:
            chosen = max(active_candidates, key=lambda item: item["risk_score"])
            active_score = chosen["risk_score"]
            active_date = chosen["date"]
            active_source = "forecast"
            active_label = chosen["label"]

        drivers = self._drivers(latest_row, thresholds, forecast)
        return {
            "region_id": region.id,
            "generated_at": datetime.now(IST).isoformat(timespec="seconds"),
            "model_mode": "deep_learning_ensemble" if self.model_ready else "hydrology_prior",
            "ensemble": {
                "model_weight": HYBRID_MODEL_WEIGHT,
                "hydrology_weight": round(1.0 - HYBRID_MODEL_WEIGHT, 2),
                "model": "BiLSTM temporal-attention + GloFAS discharge guardrail",
            },
            "thresholds": {
                "watch": _round(thresholds["watch"]),
                "warning": _round(thresholds["warning"]),
                "severe": _round(thresholds["severe"]),
                "median": _round(thresholds["median"]),
            },
            "current": {
                "date": latest_row["date"].date().isoformat(),
                "discharge": _round(latest_row["discharge"]),
                "rain_24h": _round(latest_row["precip"]),
                "rain_7d": _round(latest_row["rain_7"]),
                "soil_moisture": _round(latest_row["soil_moisture"], 3),
                "discharge_ratio_30": _round(latest_row["discharge_ratio_30"]),
                "observed_level": observed_level,
                "observed_label": RISK_LABELS[observed_level],
                "observed_score": observed_score,
                "next_24h": next_24h,
                "trend": trend,
                "trend_delta": round(trend_delta, 1),
            },
            "forecast": forecast,
            "series": series,
            "peak": {
                "date": peak["date"],
                "level": peak["level"],
                "label": peak["label"],
                "risk_score": peak["risk_score"],
            },
            "active": {
                "level": active_level,
                "label": active_label,
                "score": active_score,
                "date": active_date,
                "source": active_source,
                "horizon_days": ALERT_HORIZON_DAYS,
            },
            "drivers": drivers,
        }

    @staticmethod
    def _row_on_date(features: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series | None:
        matches = features[features["date"] == pd.Timestamp(timestamp).normalize()]
        if len(matches) == 0:
            return None
        return matches.iloc[-1]

    def _discharge_on_date(self, features: pd.DataFrame, target: pd.Timestamp, fallback: float) -> float:
        row = self._row_on_date(features, target)
        if row is None:
            return float(fallback)
        value = row.get("discharge")
        return float(value) if pd.notna(value) else float(fallback)

    @staticmethod
    def _fallback_thresholds(features: pd.DataFrame) -> dict[str, float]:
        values = features["discharge"].dropna().to_numpy()
        return {
            "median": float(np.quantile(values, 0.5)),
            "watch": float(np.quantile(values, 0.8)),
            "warning": float(np.quantile(values, 0.92)),
            "severe": float(np.quantile(values, 0.97)),
        }

    def _drivers(
        self,
        latest_row: pd.Series,
        thresholds: dict[str, float],
        forecast: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        rain_24h = float(latest_row.get("precip", 0) or 0)
        rain_7d = float(latest_row.get("rain_7", 0) or 0)
        discharge = float(latest_row.get("discharge", 0) or 0)
        ratio = float(latest_row.get("discharge_ratio_30", 0) or 0)
        soil = float(latest_row.get("soil_moisture", 0) or 0)
        soil_anomaly = float(latest_row.get("soil_anomaly_30", 0) or 0)
        delta_3d = float(latest_row.get("discharge_delta_3", 0) or 0) * 100

        rain_impact = "high" if rain_7d >= 150 or rain_24h >= 50 else "medium" if rain_7d >= 60 or rain_24h >= 20 else "low"
        discharge_impact = "high" if discharge >= thresholds["warning"] else "medium" if discharge >= thresholds["watch"] else "low"
        ratio_impact = "high" if ratio >= 1.8 else "medium" if ratio >= 1.25 else "low"
        trend_impact = "high" if delta_3d >= 25 else "medium" if delta_3d >= 10 else "low"
        soil_impact = "high" if soil_anomaly > 0.04 else "medium" if soil_anomaly > 0.015 else "low"

        return [
            _driver(
                "Rainfall",
                f"{rain_7d:.0f} mm / 7d",
                f"{rain_24h:.0f} mm in the last 24 hours; monsoon catchment loading.",
                rain_impact,
            ),
            _driver(
                "River discharge",
                f"{discharge:,.0f} m³/s",
                f"{ratio:.2f}× the 30-day median; warning level is {thresholds['warning']:,.0f} m³/s.",
                ratio_impact if ratio_impact != "low" else discharge_impact,
            ),
            _driver(
                "3-day trend",
                f"{delta_3d:+.0f}%",
                "Rate of change in the river channel over the last three days.",
                trend_impact,
            ),
            _driver(
                "Soil saturation",
                f"{soil:.2f} m³/m³",
                f"{'Above' if soil_anomaly >= 0 else 'Below'} the 30-day norm by {abs(soil_anomaly):.3f}; controls runoff response.",
                soil_impact,
            ),
        ]


def build_region_payload(region: Region, prediction: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "id": region.id,
        "name": region.name,
        "state": region.state,
        "area": region.area,
        "river": region.river,
        "station": region.station,
        "latitude": region.latitude,
        "longitude": region.longitude,
        "color": region.color,
        "description": region.description,
        "data_status": meta.get("status", "unknown"),
        "data_source": meta.get("provider", "Open-Meteo"),
        "fetched_at": meta.get("fetched_at"),
        "grid": {
            "latitude": meta.get("grid_latitude"),
            "longitude": meta.get("grid_longitude"),
        },
        **prediction,
    }
    return payload
