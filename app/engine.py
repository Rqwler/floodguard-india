"""Application engine: data refresh, inference, alerts and dashboard cache."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Any

from .alerts import evaluate_alerts, send_test_alert
from .config import (
    ALERT_COOLDOWN_HOURS,
    ALERT_HORIZON_DAYS,
    ALERT_MIN_LEVEL,
    APP_BASE_URL,
    AUTO_TRAIN,
    DASHBOARD_CACHE,
    HYBRID_MODEL_WEIGHT,
    IST,
    REFRESH_MINUTES,
    RISK_LABELS,
    ensure_directories,
    smtp_configured,
)
from .data import describe_data_status, fetch_live
from .db import (
    alert_counts,
    count_subscribers,
    init_db,
    insert_alert,
    recent_alerts,
    save_predictions,
    save_snapshot,
)
from .emails import render_alert_email, send_email
from .inference import FloodPredictor, build_region_payload
from .model import train_model
from .regions import REGIONS, REGION_BY_ID

logger = logging.getLogger("floodguard.engine")


class FloodEngine:
    def __init__(self) -> None:
        self.predictor: FloodPredictor | None = None
        self.dashboard: dict[str, Any] | None = None
        self.region_payloads: dict[str, dict[str, Any]] = {}
        self.last_error: str | None = None
        self.refreshing = False
        self._lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._last_refresh: datetime | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def initialize(self) -> None:
        ensure_directories()
        init_db()
        self._load_cached_dashboard()
        if self.predictor is None:
            self.predictor = FloodPredictor()
        if not self.predictor.model_ready and AUTO_TRAIN:
            logger.info("No model artifact found. Training a new model (first run only)…")
            try:
                await asyncio.to_thread(train_model, False, 45, 8, True)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Model training failed; using hydrological prior")
                self.last_error = f"Training failed: {exc}"
            self.predictor.reload()
        logger.info("Predictor ready: %s", "BiLSTM ensemble" if self.predictor.model_ready else "hydrology prior")
        try:
            await self.refresh(force_fetch=True, reason="startup")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Initial refresh failed")
            self.last_error = f"Initial refresh failed: {exc}"
        self._task = asyncio.create_task(self._background_loop(), name="floodguard-refresh")

    async def shutdown(self) -> None:
        self._stop_event.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _background_loop(self) -> None:
        interval = REFRESH_MINUTES * 60
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
                break
            except asyncio.TimeoutError:
                logger.info("Scheduled refresh starting")
                try:
                    await self.refresh(force_fetch=True, reason="scheduled")
                except Exception:  # noqa: BLE001
                    logger.exception("Scheduled refresh failed")

    # ------------------------------------------------------------------
    # Refresh / inference
    # ------------------------------------------------------------------
    async def refresh(self, force_fetch: bool = True, reason: str = "manual") -> dict[str, Any]:
        async with self._lock:
            self.refreshing = True
            started = time.time()
            try:
                results = await asyncio.gather(
                    *(
                        asyncio.to_thread(fetch_live, region, force_fetch)
                        for region in REGIONS
                    ),
                    return_exceptions=True,
                )
                fetched: dict[str, tuple[Any, dict[str, Any]]] = {}
                for region, result in zip(REGIONS, results):
                    if isinstance(result, Exception):
                        logger.error("Fetch failed for %s: %s", region.id, result)
                        continue
                    fetched[region.id] = result

                if not fetched:
                    raise RuntimeError("No basin data could be loaded")

                payloads: list[dict[str, Any]] = []
                if self.predictor is None:
                    self.predictor = FloodPredictor()
                for region in REGIONS:
                    if region.id not in fetched:
                        continue
                    df, meta = fetched[region.id]
                    try:
                        prediction = await asyncio.to_thread(self.predictor.predict_region, region, df)
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("Inference failed for %s", region.id)
                        meta = {**meta, "status": "degraded", "error": str(exc)}
                        continue
                    payload = build_region_payload(region, prediction, meta)
                    payloads.append(payload)
                    self.region_payloads[region.id] = payload
                    save_snapshot(
                        region.id,
                        meta.get("status", "unknown"),
                        {
                            "generated_at": prediction["generated_at"],
                            "current": prediction["current"],
                            "forecast": prediction["forecast"],
                            "thresholds": prediction["thresholds"],
                        },
                    )
                    save_predictions(region.id, prediction["forecast"])

                if not payloads:
                    raise RuntimeError("Inference produced no basin payloads")

                try:
                    await asyncio.to_thread(evaluate_alerts, payloads)
                except Exception:  # noqa: BLE001
                    logger.exception("Alert evaluation failed")

                self.dashboard = self._build_dashboard(payloads, started, reason)
                self._last_refresh = datetime.now(IST)
                self.last_error = None
                self._write_cache()
                logger.info(
                    "Refresh complete in %.1fs — %s active warning(s), source=%s",
                    time.time() - started,
                    self.dashboard["summary"]["active_warnings"],
                    self.dashboard["data_source"]["status"],
                )
                return self.dashboard
            finally:
                self.refreshing = False

    def _build_dashboard(
        self,
        payloads: list[dict[str, Any]],
        started: float,
        reason: str,
    ) -> dict[str, Any]:
        now = datetime.now(IST)
        statuses = [payload.get("data_status", "unknown") for payload in payloads]
        source_status = describe_data_status(statuses)
        active = [payload for payload in payloads if payload["active"]["level"] >= ALERT_MIN_LEVEL]
        max_level = max((payload["active"]["level"] for payload in payloads), default=0)
        peak = max(payloads, key=lambda item: item["active"]["level"] * 1000 + item["active"]["score"], default=None)
        model_block = self._model_block()
        alerts = recent_alerts(12)
        counts = alert_counts()
        subscriber_count = count_subscribers()
        model_metadata = self.predictor.metadata if self.predictor else None
        return {
            "generated_at": now.isoformat(timespec="seconds"),
            "generated_in_seconds": round(time.time() - started, 1),
            "refresh_reason": reason,
            "refreshing": False,
            "summary": {
                "regions_monitored": len(payloads),
                "active_warnings": len(active),
                "max_level": max_level,
                "max_label": RISK_LABELS[max_level],
                "max_risk_score": round(max((payload["active"]["score"] for payload in payloads), default=0), 1),
                "peak_region": peak["name"] if peak else None,
                "peak_region_id": peak["id"] if peak else None,
                "subscribers": subscriber_count,
                "alerts_sent": counts.get("sent", 0) + counts.get("simulated", 0) + counts.get("partial", 0),
                "alerts_simulated": counts.get("simulated", 0),
                "alerts_failed": counts.get("failed", 0),
                "alert_threshold": RISK_LABELS[ALERT_MIN_LEVEL],
            },
            "data_source": {
                "provider": "Open-Meteo · GloFAS + ERA5-Land / ECMWF",
                "status": source_status,
                "last_refresh": now.isoformat(timespec="seconds"),
                "next_refresh": (now + timedelta(minutes=REFRESH_MINUTES)).isoformat(timespec="seconds"),
                "refresh_minutes": REFRESH_MINUTES,
                "history": "2019 → present (model training)",
                "attempted_regions": len(statuses),
                "live_regions": statuses.count("live"),
                "cached_regions": statuses.count("cached"),
                "demo_regions": statuses.count("demo"),
                "errors": [
                    {
                        "region_id": payload["id"],
                        "region": payload["name"],
                        "status": payload["data_status"],
                        "error": payload.get("error"),
                    }
                    for payload in payloads
                    if payload.get("data_status") != "live"
                ],
            },
            "model": model_block,
            "regions": payloads,
            "alerts": alerts,
            "alert_settings": {
                "threshold_level": ALERT_MIN_LEVEL,
                "threshold_label": RISK_LABELS[ALERT_MIN_LEVEL],
                "horizon_days": ALERT_HORIZON_DAYS,
                "cooldown_hours": ALERT_COOLDOWN_HOURS,
                "email_mode": "smtp" if smtp_configured() else "simulation",
                "simulation_note": (
                    "SMTP is not configured. Alerts are written to data/outbox and shown here."
                    if not smtp_configured()
                    else "Live email delivery is configured."
                ),
                "dashboard_url": APP_BASE_URL,
            },
            "model_dataset_mode": model_metadata.get("data_mode") if model_metadata else None,
        }

    def _model_block(self) -> dict[str, Any]:
        if not self.predictor or not self.predictor.metadata:
            return {
                "status": "heuristic",
                "architecture": "Hydrological thresholds + GloFAS forecast",
                "note": "Model artifact missing. Predictions use discharge thresholds until training completes.",
            }
        metadata = self.predictor.metadata
        test = metadata.get("metrics", {}).get("test", {})
        validation = metadata.get("metrics", {}).get("validation", {})
        return {
            "status": "trained",
            "architecture": (
                f"BiLSTM {metadata.get('model_config', {}).get('num_layers', 2)}×"
                f"{metadata.get('model_config', {}).get('hidden_size', 64)} "
                "+ temporal attention + region embedding"
            ),
            "trained_at": metadata.get("trained_at"),
            "training_seconds": metadata.get("training_seconds"),
            "data_mode": metadata.get("data_mode"),
            "best_epoch": metadata.get("best_epoch"),
            "epochs_run": metadata.get("epochs_run"),
            "sequence_length": metadata.get("sequence_length"),
            "split": metadata.get("split"),
            "test": test,
            "validation": validation,
            "history": metadata.get("history"),
            "feature_names": metadata.get("feature_names"),
            "feature_labels": metadata.get("feature_labels"),
            "ensemble": {
                "model_weight": HYBRID_MODEL_WEIGHT,
                "hydrology_weight": round(1 - HYBRID_MODEL_WEIGHT, 2),
                "description": "LSTM probabilities blended with a GloFAS discharge-threshold prior.",
            },
            "notes": metadata.get("notes"),
        }

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def get_dashboard(self) -> dict[str, Any] | None:
        if self.dashboard:
            return self.dashboard
        self._load_cached_dashboard()
        return self.dashboard

    def region_payload(self, region_id: str) -> dict[str, Any]:
        if region_id not in REGION_BY_ID:
            raise KeyError("Unknown basin")
        payload = self.region_payloads.get(region_id)
        if payload is None and self.dashboard:
            payload = next(
                (region for region in self.dashboard.get("regions", []) if region["id"] == region_id),
                None,
            )
        if payload is None:
            raise KeyError("Basin data is not available yet")
        return payload

    def send_test(self, email: str, region_id: str) -> dict[str, Any]:
        payload = self.region_payload(region_id)
        return send_test_alert(email, region_id, payload)

    def send_demo_alert(self, email: str, region_id: str) -> dict[str, Any]:
        """Send a clearly-labelled synthetic Warning email and record it.

        Useful for validating the full subscription → HTML email → dashboard
        activity path outside the monsoon season.
        """
        payload = self.region_payload(region_id)
        tomorrow = (datetime.now(IST) + timedelta(days=1)).date().isoformat()
        target = {
            "date": tomorrow,
            "level": 2,
            "label": "Warning",
            "risk_score": 82.0,
            "alert_probability": 91.0,
            "source": "demo",
            "discharge": payload.get("current", {}).get("discharge"),
        }
        from .alerts import _normalise_email  # local to keep the demo path self-contained

        address = _normalise_email(email)
        from .db import get_subscriber_by_email

        subscriber = get_subscriber_by_email(address)
        token = subscriber["token"] if subscriber else "demo-preview"
        subject, html_body, text_body = render_alert_email(
            payload, target, recipient=address, token=token, test=True
        )
        subject = "DEMO · " + subject.removeprefix("TEST · ")
        result = send_email(address, subject, html_body, text_body)
        record = insert_alert(
            alert_key=f"demo:{region_id}:{uuid.uuid4().hex}",
            region_id=region_id,
            region_name=payload.get("name", region_id),
            level=2,
            level_label="Warning",
            risk_score=82.0,
            target_date=tomorrow,
            headline=f"Demo Warning alert · {payload.get('name')}",
            details={
                "demo": True,
                "source": "manual demo",
                "email_status": result.get("status"),
                "attempted": 1,
                "delivered": 1 if result.get("status") == "sent" else 0,
                "simulated": 1 if result.get("status") == "simulated" else 0,
            },
            recipients=1,
            status=result.get("status", "failed"),
        )
        return {
            "status": result.get("status"),
            "mode": result.get("mode"),
            "message": result.get("message"),
            "alert_id": record.get("id"),
            "email": address,
            "region": payload.get("name"),
            "note": "Synthetic demo warning; no real flood is implied.",
        }

    def status(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model": (
                "trained"
                if self.predictor and self.predictor.model_ready
                else "heuristic"
            ),
            "dashboard_ready": self.dashboard is not None,
            "refreshing": self.refreshing,
            "last_refresh": self._last_refresh.isoformat(timespec="seconds") if self._last_refresh else None,
            "last_error": self.last_error,
            "regions": [region.id for region in REGIONS],
        }

    def _load_cached_dashboard(self) -> None:
        if self.dashboard is not None or not DASHBOARD_CACHE.exists():
            return
        try:
            self.dashboard = json.loads(DASHBOARD_CACHE.read_text(encoding="utf-8"))
            self.dashboard["refreshing"] = False
        except Exception:  # noqa: BLE001
            logger.warning("Could not read dashboard cache", exc_info=True)

    def _write_cache(self) -> None:
        if not self.dashboard:
            return
        ensure_directories()
        temp_path = DASHBOARD_CACHE.with_suffix(".tmp")
        temp_path.write_text(json.dumps(self.dashboard, indent=2), encoding="utf-8")
        temp_path.replace(DASHBOARD_CACHE)


engine = FloodEngine()
