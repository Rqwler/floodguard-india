"""Alert subscriptions, evaluation and delivery orchestration."""

from __future__ import annotations

import re
import secrets
from datetime import datetime, timedelta
from typing import Any

from .config import (
    ALERT_COOLDOWN_HOURS,
    ALERT_HORIZON_DAYS,
    ALERT_MIN_LEVEL,
    IST,
    RISK_LABELS,
)
from .db import (
    active_subscribers_for_region,
    alert_exists,
    deactivate_subscriber,
    get_subscriber_by_email,
    get_subscriber_by_token,
    insert_alert,
    mark_subscribers_alerted,
    recent_region_alert,
    upsert_subscriber,
)
from .emails import render_alert_email, send_email
from .regions import REGION_BY_ID, REGION_IDS

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


def _normalise_email(email: str) -> str:
    value = (email or "").strip().lower()
    if not EMAIL_RE.match(value):
        raise ValueError("Please enter a valid email address")
    return value


def _normalise_regions(regions: list[str] | None) -> list[str]:
    cleaned = [region for region in (regions or []) if region in REGION_BY_ID]
    return cleaned or list(REGION_IDS)


def subscribe(email: str, regions: list[str] | None) -> dict[str, Any]:
    address = _normalise_email(email)
    selected = _normalise_regions(regions)
    token = secrets.token_urlsafe(24)
    subscriber = upsert_subscriber(address, selected, token)
    return {
        "email": subscriber["email"],
        "regions": subscriber["regions"],
        "active": subscriber["active"],
        "created_at": subscriber["created_at"],
        "message": "Subscribed. Alerts are sent when the ensemble risk reaches Warning.",
    }


def unsubscribe(token: str) -> dict[str, Any] | None:
    subscriber = get_subscriber_by_token(token)
    if not subscriber or not subscriber.get("active"):
        return None
    deactivate_subscriber(token)
    return {"email": subscriber["email"], "regions": subscriber["regions"]}


def _target_for_alert(region: dict[str, Any]) -> tuple[dict[str, Any], str]:
    current = region.get("current", {})
    observed_level = int(current.get("observed_level", 0))
    observed_target = {
        "date": current.get("date"),
        "level": observed_level,
        "label": RISK_LABELS[observed_level],
        "risk_score": current.get("observed_score", 0),
        "alert_probability": 100.0 if observed_level >= 2 else 0.0,
        "source": "observed",
        "discharge": current.get("discharge"),
    }
    forecast = region.get("forecast", [])[:ALERT_HORIZON_DAYS]
    best_forecast = max(
        forecast,
        key=lambda item: (int(item.get("level", 0)), float(item.get("risk_score", 0))),
        default=None,
    )
    if best_forecast is None:
        return observed_target, "observed"
    if observed_level >= int(best_forecast.get("level", 0)):
        return observed_target, "observed"
    return {
        **best_forecast,
        "source": "forecast",
    }, "forecast"


def _aggregate_status(statuses: list[str]) -> str:
    if not statuses:
        return "failed"
    if all(status == "sent" for status in statuses):
        return "sent"
    if all(status == "simulated" for status in statuses):
        return "simulated"
    if any(status in {"sent", "simulated"} for status in statuses):
        return "partial"
    return "failed"


def evaluate_alerts(region_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Inspect an alert horizon and deliver mail if a basin turns abnormal."""
    produced: list[dict[str, Any]] = []
    now = datetime.now(IST)
    cooldown_since = (now - timedelta(hours=ALERT_COOLDOWN_HOURS)).isoformat(timespec="seconds")
    for region in region_payloads:
        target, source = _target_for_alert(region)
        level = int(target.get("level", 0))
        if level < ALERT_MIN_LEVEL:
            continue

        region_id = region["id"]
        target_date = str(target.get("date") or current_date(region))
        key_base = f"{region_id}:{target_date}:{level}:{source}"
        subscribers = active_subscribers_for_region(region_id)
        last_alert = recent_region_alert(region_id, cooldown_since)
        if last_alert and int(last_alert.get("level", 0)) >= level:
            continue

        if not subscribers:
            key = f"{key_base}:no-sub"
            if alert_exists(key):
                continue
            record = insert_alert(
                alert_key=key,
                region_id=region_id,
                region_name=region.get("name", region_id),
                level=level,
                level_label=RISK_LABELS[level],
                risk_score=float(target.get("risk_score", 0)),
                target_date=target_date,
                headline=target.get("label", RISK_LABELS[level]) + " flood risk detected",
                details={
                    "source": source,
                    "no_subscribers": True,
                    "region_area": region.get("area"),
                    "river": region.get("river"),
                    "thresholds": region.get("thresholds"),
                    "model_mode": region.get("model_mode"),
                },
                recipients=0,
                status="no_subscribers",
            )
            produced.append(record)
            continue

        key = f"{key_base}:alert"
        if alert_exists(key):
            continue

        delivered = 0
        simulated = 0
        failed = 0
        statuses: list[str] = []
        alert_ids: list[int] = []
        for subscriber in subscribers:
            subject, html_body, text_body = render_alert_email(
                region,
                target,
                recipient=subscriber["email"],
                token=subscriber["token"],
                test=False,
            )
            result = send_email(subscriber["email"], subject, html_body, text_body)
            statuses.append(result["status"])
            if result["status"] == "sent":
                delivered += 1
                alert_ids.append(int(subscriber["id"]))
            elif result["status"] == "simulated":
                simulated += 1
                alert_ids.append(int(subscriber["id"]))
            else:
                failed += 1

        mark_subscribers_alerted(alert_ids)
        status = _aggregate_status(statuses)
        headline = (
            f"{RISK_LABELS[level]} flood risk · {region.get('name')} · "
            f"{target.get('river', region.get('river'))} at {region.get('station')}"
        )
        record = insert_alert(
            alert_key=key,
            region_id=region_id,
            region_name=region.get("name", region_id),
            level=level,
            level_label=RISK_LABELS[level],
            risk_score=float(target.get("risk_score", 0)),
            target_date=target_date,
            headline=headline,
            details={
                "source": source,
                "region_area": region.get("area"),
                "river": region.get("river"),
                "thresholds": region.get("thresholds"),
                "model_mode": region.get("model_mode"),
                "alert_probability": target.get("alert_probability"),
                "attempted": len(subscribers),
                "delivered": delivered,
                "simulated": simulated,
                "failed": failed,
                "outlook": [
                    {
                        "date": item.get("date"),
                        "label": item.get("label"),
                        "risk_score": item.get("risk_score"),
                    }
                    for item in region.get("forecast", [])[:5]
                ],
            },
            recipients=len(subscribers),
            status=status,
        )
        produced.append(record)
    return produced


def current_date(region: dict[str, Any]) -> str:
    return str(region.get("current", {}).get("date") or "")


def send_test_alert(email: str, region_id: str, region_payload: dict[str, Any]) -> dict[str, Any]:
    address = _normalise_email(email)
    if region_id not in REGION_BY_ID:
        raise ValueError("Unknown basin")
    subscriber = get_subscriber_by_email(address)
    token = subscriber.get("token") if subscriber else secrets.token_urlsafe(18)
    target, _ = _target_for_alert(region_payload)
    subject, html_body, text_body = render_alert_email(
        region_payload,
        target,
        recipient=address,
        token=token,
        test=True,
    )
    subject = "TEST · " + subject
    result = send_email(address, subject, html_body, text_body)
    return {
        "email": address,
        "region": region_payload.get("name"),
        "level": target.get("label"),
        "mode": result.get("mode"),
        "status": result.get("status"),
        "message": result.get("message"),
    }
