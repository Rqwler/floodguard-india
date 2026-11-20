"""FastAPI application for FloodGuard India."""

from __future__ import annotations

import html
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .alerts import subscribe, unsubscribe
from .config import WEB_DIR
from .db import recent_alerts
from .engine import engine
from .regions import REGIONS, REGION_BY_ID

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("floodguard.api")


class SubscribeRequest(BaseModel):
    email: str
    regions: list[str] = Field(default_factory=list)


class TestAlertRequest(BaseModel):
    email: str
    region_id: str = "assam-brahmaputra"


class DemoAlertRequest(BaseModel):
    email: str
    region_id: str = "assam-brahmaputra"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FloodGuard India")
    await engine.initialize()
    logger.info("FloodGuard India is ready")
    yield
    await engine.shutdown()
    logger.info("FloodGuard India stopped")


app = FastAPI(
    title="FloodGuard India",
    description=(
        "AI-powered flood detection and early warning for five Indian river basins. "
        "Live GloFAS river discharge and Open-Meteo weather forecasts feed a "
        "region-aware BiLSTM."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

if WEB_DIR.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")


@app.get("/", include_in_schema=False)
async def dashboard_page():
    index = WEB_DIR / "index.html"
    if not index.exists():
        return HTMLResponse("<h1>Dashboard files missing</h1>", status_code=500)
    return FileResponse(index, media_type="text/html", headers={"Cache-Control": "no-store"})


@app.get("/api/health")
async def health():
    return engine.status()


@app.get("/health", include_in_schema=False)
async def health_alias():
    return engine.status()


@app.get("/api/regions")
async def regions():
    dashboard = engine.get_dashboard()
    current = {region["id"]: region.get("active") for region in (dashboard or {}).get("regions", [])}
    return {
        "regions": [
            {**region.to_dict(), "active": current.get(region.id)} for region in REGIONS
        ]
    }


@app.get("/api/dashboard")
async def dashboard():
    payload = engine.get_dashboard()
    if payload is None:
        try:
            payload = await engine.refresh(force_fetch=True, reason="first-request")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"Live data is not ready yet: {exc}") from exc
    return JSONResponse(content=payload, headers={"Cache-Control": "no-store"})


@app.get("/api/regions/{region_id}")
async def region_detail(region_id: str):
    try:
        payload = engine.region_payload(region_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown or unavailable basin") from exc
    return JSONResponse(content=payload, headers={"Cache-Control": "no-store"})


@app.post("/api/refresh")
async def refresh_data():
    try:
        payload = await engine.refresh(force_fetch=True, reason="manual")
        return JSONResponse(content=payload, headers={"Cache-Control": "no-store"})
    except Exception as exc:  # noqa: BLE001
        logger.exception("Manual refresh failed")
        raise HTTPException(status_code=502, detail=f"Refresh failed: {exc}") from exc


@app.post("/api/subscribe")
async def subscribe_email(body: SubscribeRequest):
    try:
        result = subscribe(body.email, body.regions)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/unsubscribe", response_class=HTMLResponse)
async def unsubscribe_page(token: str = Query(default="")):
    result = unsubscribe(token) if token else None
    if result:
        title = "Subscription cancelled"
        message = f"<strong>{html.escape(result['email'])}</strong> will no longer receive FloodGuard alerts."
        color = "#2dd4bf"
    else:
        title = "Link not recognised"
        message = "This unsubscribe link is invalid or has already been used."
        color = "#fb7185"
    return HTMLResponse(
        f"""<!doctype html><html><head><meta charset="utf-8">
        <meta name="viewport" content="width=device-width,initial-scale=1">
        <title>{title} · FloodGuard India</title></head>
        <body style="margin:0;min-height:100vh;display:grid;place-items:center;background:#07111f;color:#e6f1ff;font-family:'Segoe UI',Arial,sans-serif;">
        <div style="max-width:520px;padding:36px;border:1px solid #1e3a55;background:#0b1a2d;border-radius:18px;margin:24px;">
        <div style="font-size:11px;letter-spacing:2px;text-transform:uppercase;color:{color};font-weight:700;">FloodGuard India</div>
        <h1 style="font-size:26px;margin:10px 0 12px;">{title}</h1>
        <p style="color:#a9bfd3;line-height:1.7;">{message}</p>
        <a href="/" style="display:inline-block;margin-top:10px;background:#22d3ee;color:#06111f;text-decoration:none;font-weight:700;padding:10px 16px;border-radius:8px;">Return to dashboard</a>
        </div></body></html>"""
    )


@app.post("/api/alerts/test")
async def test_alert(body: TestAlertRequest):
    try:
        return engine.send_test(body.email, body.region_id)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Test alert failed")
        raise HTTPException(status_code=502, detail=f"Test alert failed: {exc}") from exc


@app.post("/api/alerts/demo")
async def demo_alert(body: DemoAlertRequest):
    try:
        return engine.send_demo_alert(body.email, body.region_id)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Demo alert failed")
        raise HTTPException(status_code=502, detail=f"Demo alert failed: {exc}") from exc


@app.get("/api/alerts")
async def alerts():
    return {"alerts": recent_alerts(25)}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return HTMLResponse("", status_code=204)


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
