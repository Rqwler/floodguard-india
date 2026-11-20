"""HTML email rendering and SMTP delivery.

Without SMTP credentials the platform runs in *simulation mode*: every email is
written to ``data/outbox`` so the complete alert path can still be inspected and
tested locally. Add the SMTP_* variables in .env to send real mail.
"""

from __future__ import annotations

import html
import smtplib
import ssl
import uuid
from datetime import datetime
from email.message import EmailMessage
from email.utils import formataddr
from pathlib import Path
from string import Template
from typing import Any

from .config import (
    APP_BASE_URL,
    IST,
    OUTBOX_DIR,
    RISK_COLORS,
    SMTP_FROM,
    SMTP_FROM_NAME,
    SMTP_HOST,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_USE_SSL,
    SMTP_USE_TLS,
    SMTP_USER,
    ensure_directories,
    smtp_configured,
)

_ALERT_TEMPLATE = Template(
    """<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#07111f;font-family:'Segoe UI',Arial,sans-serif;color:#e6f1ff;">
  <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background:#07111f;padding:28px 12px;">
    <tr><td align="center">
      <table role="presentation" width="600" cellspacing="0" cellpadding="0" style="max-width:600px;width:100%;background:#0b1a2d;border:1px solid #1e3a55;border-radius:16px;overflow:hidden;">
        <tr><td style="padding:22px 28px;background:linear-gradient(135deg,#0c2338,#10314a);border-bottom:1px solid #1e3a55;">
          <div style="font-size:12px;letter-spacing:2px;text-transform:uppercase;color:#67e8f9;font-weight:700;">FloodGuard India</div>
          <div style="font-size:24px;font-weight:800;margin-top:6px;color:#f8fafc;">$headline</div>
          <div style="font-size:13px;color:#9fb6cc;margin-top:6px;">$region_line</div>
        </td></tr>
        <tr><td style="padding:22px 28px 4px;">
          <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
            <tr>
              <td style="padding:14px;background:#0f2439;border-radius:12px;border:1px solid #1e3a55;">
                <div style="font-size:11px;letter-spacing:1.5px;color:#9fb6cc;text-transform:uppercase;">Alert level</div>
                <div style="font-size:24px;font-weight:800;color:$risk_color;margin-top:4px;">$level_label</div>
                <div style="font-size:12px;color:#9fb6cc;margin-top:2px;">Target date: $target_date</div>
              </td>
              <td width="12"></td>
              <td style="padding:14px;background:#0f2439;border-radius:12px;border:1px solid #1e3a55;">
                <div style="font-size:11px;letter-spacing:1.5px;color:#9fb6cc;text-transform:uppercase;">Risk score</div>
                <div style="font-size:24px;font-weight:800;color:#f8fafc;margin-top:4px;">$risk_score<span style="font-size:14px;color:#9fb6cc;">/100</span></div>
                <div style="font-size:12px;color:#9fb6cc;margin-top:2px;">Warning probability: $alert_probability%</div>
              </td>
            </tr>
          </table>
        </td></tr>
        <tr><td style="padding:18px 28px 0;">
          <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="font-size:14px;line-height:1.6;">
            <tr><td style="color:#9fb6cc;padding:4px 0;">River / station</td><td align="right" style="color:#f8fafc;font-weight:600;">$river</td></tr>
            <tr><td style="color:#9fb6cc;padding:4px 0;">Observed discharge</td><td align="right" style="color:#f8fafc;font-weight:600;">$discharge m³/s</td></tr>
            <tr><td style="color:#9fb6cc;padding:4px 0;">Rainfall (7 days)</td><td align="right" style="color:#f8fafc;font-weight:600;">$rain mm</td></tr>
            <tr><td style="color:#9fb6cc;padding:4px 0;">Warning threshold</td><td align="right" style="color:#f8fafc;font-weight:600;">$warning_threshold m³/s</td></tr>
          </table>
        </td></tr>
        <tr><td style="padding:20px 28px 0;">
          <div style="font-size:12px;letter-spacing:1.4px;color:#67e8f9;text-transform:uppercase;font-weight:700;margin-bottom:8px;">Next 8 days</div>
          <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="font-size:12px;color:#dbeafe;">
            $outlook_rows
          </table>
        </td></tr>
        <tr><td style="padding:20px 28px 4px;">
          <div style="font-size:13px;color:#b9cadb;line-height:1.65;">$message</div>
        </td></tr>
        <tr><td style="padding:18px 28px 24px;">
          <a href="$dashboard_url" style="display:inline-block;background:#22d3ee;color:#06111f;text-decoration:none;font-weight:800;font-size:13px;padding:11px 18px;border-radius:9px;">Open live dashboard</a>
        </td></tr>
        <tr><td style="padding:16px 28px 22px;border-top:1px solid #1e3a55;color:#71869a;font-size:11px;line-height:1.6;">
          You are receiving this because you subscribed to FloodGuard alerts. This is an AI-assisted early warning, not an official government advisory. Always follow IMD, CWC and local administration guidance.
          <br><br><a href="$unsubscribe_url" style="color:#67e8f9;">Unsubscribe</a>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""
)

_TEST_MESSAGE = (
    "This is a delivery test from FloodGuard India. Your alert subscription is working. "
    "Real messages are sent automatically when the ensemble risk for a selected basin "
    "reaches Warning or Severe."
)


def _format_date(value: str) -> str:
    try:
        return datetime.fromisoformat(value).strftime("%d %b %Y")
    except (TypeError, ValueError):
        return str(value or "—")


def _outlook_rows(forecast: list[dict[str, Any]]) -> str:
    rows = []
    for item in forecast[:8]:
        color = RISK_COLORS[int(item.get("level", 0))]
        rows.append(
            "<tr>"
            f'<td style="padding:6px 0;color:#9fb6cc;">{html.escape(_format_date(item.get("date", "")))}</td>'
            f'<td style="padding:6px 0;color:#f8fafc;text-align:center;">{item.get("discharge") or "—"} m³/s</td>'
            f'<td style="padding:6px 0;text-align:right;color:{color};font-weight:700;">'
            f'{html.escape(str(item.get("label", "Normal")))} · {item.get("risk_score", 0)}/100</td>'
            "</tr>"
        )
    return "".join(rows)


def render_alert_email(
    region: dict[str, Any],
    target: dict[str, Any],
    *,
    recipient: str,
    token: str,
    test: bool = False,
) -> tuple[str, str, str]:
    level = int(target.get("level", 0))
    level_label = str(target.get("label", "Warning"))
    color = RISK_COLORS[level]
    current = region.get("current", {})
    thresholds = region.get("thresholds", {})
    region_line = (
        f"{region.get('name')} · {region.get('area')} · "
        f"{region.get('river')} at {region.get('station')}"
    )
    prefix = "TEST · " if test else ""
    subject = (
        f"{prefix}{level_label.upper()} flood risk · {region.get('name')} "
        f"({region.get('river')} at {region.get('station')}) · {_format_date(target.get('date', ''))}"
    )
    headline = f"{prefix}{level_label} flood risk"
    message = (
        "River discharge is forecast to move into an abnormal range for this basin. "
        "Please review local conditions and follow official instructions."
        if not test
        else _TEST_MESSAGE
    )
    unsubscribe_url = f"{APP_BASE_URL}/api/unsubscribe?token={token}"
    dashboard_url = f"{APP_BASE_URL}/?region={region.get('id')}"
    body = _ALERT_TEMPLATE.safe_substitute(
        headline=html.escape(headline),
        region_line=html.escape(region_line),
        level_label=html.escape(level_label),
        risk_color=color,
        target_date=_format_date(target.get("date", "")),
        risk_score=target.get("risk_score", 0),
        alert_probability=target.get("alert_probability", 0),
        river=html.escape(str(region.get("river") or "")),
        discharge=current.get("discharge", "—"),
        rain=current.get("rain_7d", "—"),
        warning_threshold=thresholds.get("warning", "—"),
        outlook_rows=_outlook_rows(region.get("forecast", [])),
        message=html.escape(message),
        dashboard_url=html.escape(dashboard_url),
        unsubscribe_url=html.escape(unsubscribe_url),
    )
    text = (
        f"{headline} — {region_line}\n"
        f"Target: {_format_date(target.get('date', ''))}\n"
        f"Risk score: {target.get('risk_score', 0)}/100, warning probability {target.get('alert_probability', 0)}%\n"
        f"Observed discharge: {current.get('discharge', '—')} m³/s; warning threshold {thresholds.get('warning', '—')} m³/s\n"
        f"Dashboard: {dashboard_url}\nUnsubscribe: {unsubscribe_url}\n"
        "AI-assisted early warning. Follow IMD/CWC and local authority guidance."
    )
    return subject, body, text


def send_email(recipient: str, subject: str, html_body: str, text_body: str) -> dict[str, Any]:
    ensure_directories()
    if not smtp_configured():
        timestamp = datetime.now(IST).strftime("%Y%m%d_%H%M%S")
        filename = f"alert_{timestamp}_{uuid.uuid4().hex[:8]}.html"
        path = OUTBOX_DIR / filename
        path.write_text(
            f"<!-- To: {html.escape(recipient)} -->\n<!-- Subject: {html.escape(subject)} -->\n{html_body}",
            encoding="utf-8",
        )
        text_path = OUTBOX_DIR / filename.replace(".html", ".txt")
        text_path.write_text(f"To: {recipient}\nSubject: {subject}\n\n{text_body}", encoding="utf-8")
        return {
            "status": "simulated",
            "mode": "outbox",
            "message": "SMTP is not configured; email saved to data/outbox.",
            "file": str(path),
        }

    message = EmailMessage()
    message["From"] = formataddr((SMTP_FROM_NAME, SMTP_FROM))
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(text_body)
    message.add_alternative(html_body, subtype="html")
    try:
        if SMTP_USE_SSL:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=30, context=context) as server:
                if SMTP_USER:
                    server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(message)
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
                server.ehlo()
                if SMTP_USE_TLS:
                    server.starttls(context=ssl.create_default_context())
                    server.ehlo()
                if SMTP_USER:
                    server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(message)
        return {"status": "sent", "mode": "smtp", "message": "Email sent."}
    except Exception as exc:  # noqa: BLE001 - delivery failures are surfaced to the dashboard
        return {
            "status": "failed",
            "mode": "smtp",
            "message": f"{type(exc).__name__}: {exc}",
        }
