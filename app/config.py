"""Central configuration for FloodGuard India.

All values can be overridden in a local .env file. The defaults are chosen so
that the project runs out of the box with Open-Meteo's free APIs and a local
SQLite database.
"""

from __future__ import annotations

import os
from datetime import timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
HISTORY_DIR = DATA_DIR / "history"
LIVE_DIR = DATA_DIR / "live"
OUTBOX_DIR = DATA_DIR / "outbox"
MODEL_DIR = BASE_DIR / "models"
WEB_DIR = BASE_DIR / "web"

DB_PATH = DATA_DIR / "floodguard.db"
DASHBOARD_CACHE = DATA_DIR / "dashboard_cache.json"
MODEL_PATH = MODEL_DIR / "flood_lstm.pt"
METADATA_PATH = MODEL_DIR / "metadata.json"

IST = timezone(timedelta(hours=5, minutes=30))

TRAIN_START_DATE = os.getenv("TRAIN_START_DATE", "2019-01-01")
HISTORY_LAG_DAYS = int(os.getenv("HISTORY_LAG_DAYS", "7"))
HISTORY_CACHE_HOURS = int(os.getenv("HISTORY_CACHE_HOURS", "24"))

SEQUENCE_LENGTH = 14
HIDDEN_SIZE = 64
NUM_LAYERS = 2
REGION_EMBEDDING = 8
DROPOUT = 0.25

AUTO_TRAIN = os.getenv("AUTO_TRAIN", "true").strip().lower() in {"1", "true", "yes", "on"}
REFRESH_MINUTES = max(5, int(os.getenv("REFRESH_MINUTES", "30")))
ALERT_MIN_LEVEL = max(1, min(3, int(os.getenv("ALERT_MIN_LEVEL", "2"))))
ALERT_HORIZON_DAYS = max(1, min(7, int(os.getenv("ALERT_HORIZON_DAYS", "3"))))
ALERT_COOLDOWN_HOURS = max(1, int(os.getenv("ALERT_COOLDOWN_HOURS", "12")))

def _public_base_url() -> str:
    """Resolve the public URL used in emails, preferring explicit config."""
    explicit = os.getenv("APP_BASE_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    render_url = os.getenv("RENDER_EXTERNAL_URL", "").strip()
    if render_url:
        return render_url.rstrip("/")
    railway_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN", "").strip()
    if railway_domain:
        return f"https://{railway_domain}".rstrip("/")
    return "http://localhost:8000"


APP_BASE_URL = _public_base_url()
OPENMETEO_API_KEY = os.getenv("OPENMETEO_API_KEY", "").strip()

SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").strip()
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER).strip()
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "FloodGuard India").strip()
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").strip().lower() in {"1", "true", "yes", "on"}
SMTP_USE_SSL = os.getenv("SMTP_USE_SSL", "false").strip().lower() in {"1", "true", "yes", "on"}

RISK_LABELS = ["Normal", "Watch", "Warning", "Severe"]
RISK_SHORT = ["NORMAL", "WATCH", "WARNING", "SEVERE"]
RISK_COLORS = ["#2dd4bf", "#fbbf24", "#fb923c", "#fb7185"]
RISK_SCORE_ANCHORS = [0, 35, 70, 100]
RISK_SCORE_BANDS = [40, 65, 85]
HYBRID_MODEL_WEIGHT = max(0.0, min(1.0, float(os.getenv("HYBRID_MODEL_WEIGHT", "0.4"))))

FLASH_FORECAST_DAYS = 8
CHART_HISTORY_DAYS = 60


def ensure_directories() -> None:
    for directory in (DATA_DIR, HISTORY_DIR, LIVE_DIR, OUTBOX_DIR, MODEL_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def smtp_configured() -> bool:
    return bool(SMTP_HOST and SMTP_FROM and (SMTP_USER or SMTP_HOST))
