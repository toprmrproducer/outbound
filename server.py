"""
FastAPI backend for the outbound calling dashboard.

Endpoints:
  POST   /api/call                  — Dispatch a single outbound call
  GET    /api/calls                 — Paginated call log
  GET    /api/appointments          — All/filtered appointments
  DELETE /api/appointments/{id}     — Cancel an appointment
  GET    /api/stats                 — Aggregate stats
  GET    /api/prompt                — Get saved system prompt
  POST   /api/prompt                — Save system prompt
  DELETE /api/prompt                — Reset to default
  GET    /api/settings              — Get all saved API keys/config (secrets masked)
  POST   /api/settings              — Save API keys/config (BYOK)
  GET    /api/errors                — Get error log
  DELETE /api/errors                — Clear error log

GET / serves the dashboard from ui/index.html.
"""

import json
import logging
import os
import random
import traceback
import ssl
import certifi
import aiohttp
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Patch ssl.create_default_context to use certifi's CA bundle globally.
_orig_create_default_context = ssl.create_default_context

def _certifi_create_default_context(purpose=ssl.Purpose.SERVER_AUTH, **kwargs):
    if not kwargs.get("cafile") and not kwargs.get("capath") and not kwargs.get("cadata"):
        kwargs["cafile"] = certifi.where()
    return _orig_create_default_context(purpose, **kwargs)

ssl.create_default_context = _certifi_create_default_context

from db import (
    SENSITIVE_KEYS,
    cancel_appointment,
    clear_errors,
    get_all_appointments,
    get_all_calls,
    get_all_settings,
    get_errors,
    get_logs,
    get_setting,
    get_stats,
    init_db,
    log_error,
    save_settings,
    set_setting,
)
from prompts import DEFAULT_SYSTEM_PROMPT

load_dotenv(".env", override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

init_db()

app = FastAPI(title="Outbound Caller Dashboard", version="1.0.0")


# ---------------------------------------------------------------------------
# Helper: DB setting → .env fallback
# ---------------------------------------------------------------------------

async def eff(key: str) -> str:
    """Return the DB-saved value for key, else the .env / os.environ value."""
    val = await get_setting(key, "")
    return val if val else os.getenv(key, "")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class CallRequest(BaseModel):
    phone: str
    lead_name: str = "there"
    business_name: str = "our company"
    service_type: str = "our service"
    system_prompt: Optional[str] = None


class PromptRequest(BaseModel):
    prompt: str


class SettingsRequest(BaseModel):
    settings: dict  # {KEY: value, ...}  — empty string = "don't overwrite"


# ---------------------------------------------------------------------------
# Global exception handler — logs to error_logs table
# ---------------------------------------------------------------------------

@app.middleware("http")
async def _error_logging_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Unhandled error on %s: %s", request.url.path, exc)
        try:
            await log_error(
                source="server",
                message=str(exc),
                detail=f"{request.method} {request.url.path}\n{tb[:1500]}",
            )
        except Exception:
            pass
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ---------------------------------------------------------------------------
# SIP Trunk auto-setup
# ---------------------------------------------------------------------------

def _lk_session():
    """Return an aiohttp session with SSL verification disabled (LiveKit compat)."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ctx))


async def _lk_client(session):
    url    = await eff("LIVEKIT_URL")
    key    = await eff("LIVEKIT_API_KEY")
    secret = await eff("LIVEKIT_API_SECRET")
    if not (url and key and secret):
        raise HTTPException(400, "LiveKit credentials not configured — go to Settings first.")
    from livekit import api as lk_api_module
    return lk_api_module.LiveKitAPI(url=url, api_key=key, api_secret=secret, session=session), lk_api_module


@app.post("/api/setup/trunk")
async def api_setup_trunk():
    """
    Create a LiveKit SIP outbound trunk using saved Vobiz credentials,
    then save the resulting trunk ID back to settings.
    Call this once after configuring Vobiz settings.
    """
    sip_domain = await eff("VOBIZ_SIP_DOMAIN")
    username   = await eff("VOBIZ_USERNAME")
    password   = await eff("VOBIZ_PASSWORD")
    phone      = await eff("VOBIZ_OUTBOUND_NUMBER")

    if not all([sip_domain, username, password, phone]):
        raise HTTPException(400, "Vobiz credentials incomplete — set SIP Domain, Username, Password, and Outbound Number first.")

    session = _lk_session()
    try:
        lk, lk_api = await _lk_client(session)
        trunk = await lk.sip.create_sip_outbound_trunk(
            lk_api.CreateSIPOutboundTrunkRequest(
                trunk=lk_api.SIPOutboundTrunkInfo(
                    name="Vobiz Outbound Trunk",
                    address=sip_domain,
                    auth_username=username,
                    auth_password=password,
                    numbers=[phone],
                )
            )
        )
        trunk_id = trunk.sip_trunk_id
        await set_setting("OUTBOUND_TRUNK_ID", trunk_id)
        logger.info("SIP trunk created: %s", trunk_id)
        await log_error("server", f"SIP trunk auto-created: {trunk_id}", level="info")
        return {"status": "created", "trunk_id": trunk_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Trunk setup failed: %s", exc)
        await log_error("server", f"Trunk setup failed: {exc}", level="error")
        raise HTTPException(500, str(exc))
    finally:
        await session.close()


@app.get("/api/setup/trunk")
async def api_list_trunks():
    """List existing SIP outbound trunks in this LiveKit project."""
    session = _lk_session()
    try:
        lk, lk_api = await _lk_client(session)
        trunks = await lk.sip.list_sip_outbound_trunk(lk_api.ListSIPOutboundTrunkRequest())
        return {"trunks": [{"id": t.sip_trunk_id, "name": t.name, "address": t.address} for t in (trunks.items or [])]}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))
    finally:
        await session.close()


# ---------------------------------------------------------------------------
# Call dispatch
# ---------------------------------------------------------------------------

@app.post("/api/call")
async def api_trigger_call(req: CallRequest):
    """Dispatch a single outbound call via LiveKit agent dispatch."""
    if not req.phone.startswith("+"):
        raise HTTPException(status_code=400, detail="Phone must start with '+' (E.164 format)")

    # Credentials: DB setting wins, .env is fallback
    url    = await eff("LIVEKIT_URL")
    key    = await eff("LIVEKIT_API_KEY")
    secret = await eff("LIVEKIT_API_SECRET")

    if not (url and key and secret):
        raise HTTPException(
            status_code=400,
            detail="LiveKit credentials not configured. Go to ⚙️ Settings and add your keys.",
        )

    session = _lk_session()
    try:
        lk, lk_api = await _lk_client(session)
        room_name = f"call-{req.phone.replace('+', '')}-{random.randint(1000, 9999)}"

        effective_prompt = req.system_prompt
        if not effective_prompt:
            saved = await get_setting("system_prompt", "")
            effective_prompt = saved if saved else None

        metadata = {
            "phone_number": req.phone,
            "lead_name": req.lead_name,
            "business_name": req.business_name,
            "service_type": req.service_type,
            "system_prompt": effective_prompt,
        }

        dispatch = await lk.agent_dispatch.create_dispatch(
            lk_api.CreateAgentDispatchRequest(
                agent_name="outbound-caller",
                room=room_name,
                metadata=json.dumps(metadata),
            )
        )
        await lk.aclose()

        return {
            "status": "dispatched",
            "room_name": room_name,
            "job_id": dispatch.id,
            "phone": req.phone,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Dispatch error: %s", exc)
        await log_error("server", f"Dispatch failed for {req.phone}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        await session.close()


# ---------------------------------------------------------------------------
# Call logs, appointments, stats
# ---------------------------------------------------------------------------

@app.get("/api/calls")
async def api_get_calls(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
):
    rows = await get_all_calls(page=page, limit=limit)
    return {"page": page, "limit": limit, "data": rows}


@app.get("/api/appointments")
async def api_get_appointments(date: Optional[str] = None):
    rows = await get_all_appointments(date_filter=date)
    return {"data": rows}


@app.delete("/api/appointments/{appointment_id}")
async def api_cancel_appointment(appointment_id: str):
    ok = await cancel_appointment(appointment_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Appointment not found or already cancelled")
    return {"status": "cancelled", "id": appointment_id}


@app.get("/api/stats")
async def api_get_stats():
    return await get_stats()


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

@app.get("/api/prompt")
async def api_get_prompt():
    saved = await get_setting("system_prompt", "")
    return {"prompt": saved if saved else DEFAULT_SYSTEM_PROMPT, "is_custom": bool(saved)}


@app.post("/api/prompt")
async def api_save_prompt(req: PromptRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    await set_setting("system_prompt", req.prompt.strip())
    return {"status": "saved"}


@app.delete("/api/prompt")
async def api_reset_prompt():
    await set_setting("system_prompt", "")
    return {"status": "reset", "prompt": DEFAULT_SYSTEM_PROMPT}


# ---------------------------------------------------------------------------
# BYOK Settings
# ---------------------------------------------------------------------------

@app.get("/api/settings")
async def api_get_settings():
    """
    Return all saved settings.
    Sensitive keys (API secrets, passwords) come back as
    {"value": "", "configured": true} — the raw secret is never sent to the browser.
    """
    return await get_all_settings()


@app.post("/api/settings")
async def api_save_settings(req: SettingsRequest):
    """
    Save a batch of settings.
    Only whitelisted keys are accepted. Empty values are skipped (won't wipe existing secrets).
    """
    ALLOWED_KEYS = {
        "LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET",
        "GOOGLE_API_KEY", "GEMINI_MODEL", "GEMINI_TTS_VOICE", "USE_GEMINI_REALTIME",
        "VOBIZ_SIP_DOMAIN", "VOBIZ_USERNAME", "VOBIZ_PASSWORD",
        "VOBIZ_OUTBOUND_NUMBER", "OUTBOUND_TRUNK_ID", "DEFAULT_TRANSFER_NUMBER",
        "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER",
    }
    filtered = {k: v for k, v in req.settings.items() if k in ALLOWED_KEYS}
    if not filtered:
        raise HTTPException(status_code=400, detail="No valid settings keys provided")
    await save_settings(filtered)
    return {"status": "saved", "keys_updated": list(filtered.keys())}


# ---------------------------------------------------------------------------
# Logs (all levels) + Errors (alias for backward compat)
# ---------------------------------------------------------------------------

@app.get("/api/logs")
async def api_get_logs(
    level: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = Query(200, ge=1, le=1000),
):
    rows = await get_logs(level=level, source=source, limit=limit)
    return {"data": rows}


@app.delete("/api/logs")
async def api_clear_logs():
    await clear_errors()
    return {"status": "cleared"}


@app.get("/api/errors")
async def api_get_errors(limit: int = Query(100, ge=1, le=500)):
    rows = await get_errors(limit=limit)
    return {"data": rows}


@app.delete("/api/errors")
async def api_clear_errors():
    await clear_errors()
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Serve UI
# ---------------------------------------------------------------------------

UI_DIR = Path(__file__).parent / "ui"

if UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def serve_dashboard():
        index = UI_DIR / "index.html"
        if not index.exists():
            return HTMLResponse("<h1>Dashboard not found</h1>", status_code=404)
        return HTMLResponse(index.read_text(encoding="utf-8"))
else:
    @app.get("/")
    async def no_ui():
        return {"message": "UI not found. Create the ui/ directory with index.html."}
