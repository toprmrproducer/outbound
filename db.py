import os
import uuid
from datetime import datetime, timedelta
from typing import Optional

# ---------------------------------------------------------------------------
# Hardcoded defaults — used as last-resort fallback when neither Supabase
# settings nor environment variables are configured by the user.
# Users override these via the Settings tab (stored in Supabase).
# ---------------------------------------------------------------------------
DEFAULTS = {
    "LIVEKIT_URL":             "wss://abc-dtz1tiod.livekit.cloud",
    "LIVEKIT_API_KEY":         "API4MSqHSSyiVdh",
    "LIVEKIT_API_SECRET":      "pG6TNGyYfi2djbRgxo8g1fky7DoI2C5w8nHSUFqxRjg",
    "GOOGLE_API_KEY":          "AIzaSyB9jUcS1xhEykGj9P3pGLndxHP3zyW-VOw",
    "GEMINI_MODEL":            "gemini-live-2.5-flash-native-audio",
    "GEMINI_TTS_VOICE":        "Aoede",
    "USE_GEMINI_REALTIME":     "true",
    "VOBIZ_SIP_DOMAIN":        "81b129db.sip.vobiz.ai",
    "VOBIZ_USERNAME":          "testyt",
    "VOBIZ_PASSWORD":          "test12345@",
    "VOBIZ_OUTBOUND_NUMBER":   "+918071387394",
    "OUTBOUND_TRUNK_ID":       "",
    "DEFAULT_TRANSFER_NUMBER": "+918071387394",
    "SUPABASE_URL":            "https://iocllooszyeeysfyxbfq.supabase.co",
    "SUPABASE_SERVICE_KEY":    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlvY2xsb29zenllZXlzZnl4YmZxIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3Njc5MjAzNiwiZXhwIjoyMDkyMzY4MDM2fQ.RERrvk0ZXEeqezxsPhXwtdtD_r-J-e2uR1Qb7SA4lPY",
    "DEEPGRAM_API_KEY":        "85e05deac6b575d89f80ad25fc7a2b662e52f14f",
}


def _default(key: str) -> str:
    """Return env var → hardcoded default (in that priority order)."""
    return os.getenv(key, DEFAULTS.get(key, ""))


SUPABASE_URL = _default("SUPABASE_URL")
SUPABASE_KEY = _default("SUPABASE_SERVICE_KEY")

SENSITIVE_KEYS = {
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
    "GOOGLE_API_KEY",
    "VOBIZ_PASSWORD",
    "TWILIO_AUTH_TOKEN",
    "SUPABASE_SERVICE_KEY",
}


def _sdb():
    """Synchronous Supabase client — used only at agent startup."""
    from supabase import create_client
    return create_client(_default("SUPABASE_URL"), _default("SUPABASE_SERVICE_KEY"))


async def _adb():
    """Async Supabase client — used for all server/agent async operations."""
    from supabase._async.client import create_client
    return await create_client(_default("SUPABASE_URL"), _default("SUPABASE_SERVICE_KEY"))


def init_db() -> None:
    """Verify Supabase is reachable and tables exist."""
    url = os.getenv("SUPABASE_URL", SUPABASE_URL)
    key = os.getenv("SUPABASE_SERVICE_KEY", SUPABASE_KEY)
    if not url or not key:
        print("⚠️  SUPABASE_URL or SUPABASE_SERVICE_KEY not set.")
        print("   Add them to your .env file or Settings tab.")
        return
    try:
        db = _sdb()
        db.table("settings").select("key").limit(1).execute()
        print("✅ Supabase connected")
    except Exception as exc:
        print(f"⚠️  Supabase connection failed: {exc}")
        print("   Run supabase_schema.sql in your Supabase Dashboard → SQL Editor")


# ── Settings ────────────────────────────────────────────────────────────────

async def get_all_settings() -> dict:
    db = await _adb()
    result = await db.table("settings").select("key, value").execute()
    # Start with .env values as baseline so keys are always visible
    KNOWN_KEYS = [
        "LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET",
        "GOOGLE_API_KEY", "GEMINI_MODEL", "GEMINI_TTS_VOICE", "USE_GEMINI_REALTIME",
        "VOBIZ_SIP_DOMAIN", "VOBIZ_USERNAME", "VOBIZ_PASSWORD",
        "VOBIZ_OUTBOUND_NUMBER", "OUTBOUND_TRUNK_ID", "DEFAULT_TRANSFER_NUMBER",
        "DEEPGRAM_API_KEY", "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER",
    ]
    out: dict = {}
    for k in KNOWN_KEYS:
        env_val = _default(k)
        if k in SENSITIVE_KEYS:
            out[k] = {"value": "", "configured": bool(env_val)}
        else:
            out[k] = {"value": env_val, "configured": bool(env_val)}

    # Supabase values override .env
    for row in (result.data or []):
        k, v = row["key"], row["value"]
        if k == "TEST_KEY":
            continue
        if k in SENSITIVE_KEYS:
            out[k] = {"value": "", "configured": bool(v)}
        else:
            out[k] = {"value": v, "configured": bool(v)}
    return out


async def save_settings(data: dict) -> None:
    db = await _adb()
    updated_at = datetime.now().isoformat()
    rows = [
        {"key": k, "value": str(v), "updated_at": updated_at}
        for k, v in data.items()
        if v is not None and v != ""
    ]
    if rows:
        await db.table("settings").upsert(rows, on_conflict="key").execute()


async def get_setting(key: str, default: str = "") -> str:
    db = await _adb()
    result = await db.table("settings").select("value").eq("key", key).maybe_single().execute()
    if result and result.data:
        return result.data["value"]
    return _default(key) or default


async def set_setting(key: str, value: str) -> None:
    db = await _adb()
    await db.table("settings").upsert(
        {"key": key, "value": value, "updated_at": datetime.now().isoformat()},
        on_conflict="key",
    ).execute()


# ── Error logs ───────────────────────────────────────────────────────────────

async def log_error(source: str, message: str, detail: str = "", level: str = "error") -> None:
    try:
        db = await _adb()
        await db.table("error_logs").insert({
            "id": str(uuid.uuid4()),
            "source": source,
            "level": level,
            "message": message[:500],
            "detail": detail[:2000],
            "timestamp": datetime.now().isoformat(),
        }).execute()
    except Exception:
        pass  # Never crash the caller because of a logging failure


async def get_errors(limit: int = 100) -> list:
    db = await _adb()
    result = await (
        db.table("error_logs")
        .select("*")
        .order("timestamp", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


async def get_logs(
    level: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 200,
) -> list:
    db = await _adb()
    query = db.table("error_logs").select("*").order("timestamp", desc=True).limit(limit)
    if level:
        query = query.eq("level", level)
    if source:
        query = query.eq("source", source)
    result = await query.execute()
    return result.data or []


async def clear_errors() -> None:
    db = await _adb()
    await db.table("error_logs").delete().neq("id", "").execute()


# ── Appointments ─────────────────────────────────────────────────────────────

async def insert_appointment(name: str, phone: str, date: str, time: str, service: str) -> str:
    full_id = str(uuid.uuid4())
    booking_id = full_id[:8].upper()
    db = await _adb()
    await db.table("appointments").insert({
        "id": full_id,
        "name": name,
        "phone": phone,
        "date": date,
        "time": time,
        "service": service,
        "status": "booked",
        "created_at": datetime.now().isoformat(),
    }).execute()
    return booking_id


async def check_slot(date: str, time: str) -> bool:
    db = await _adb()
    result = await (
        db.table("appointments")
        .select("id")
        .eq("date", date)
        .eq("time", time)
        .eq("status", "booked")
        .maybe_single()
        .execute()
    )
    return result.data is None


async def get_next_available(date: str, time: str) -> str:
    try:
        dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
    except ValueError:
        dt = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

    for _ in range(7 * 24):
        dt += timedelta(hours=1)
        if 9 <= dt.hour < 18:
            if await check_slot(dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M")):
                return f"{dt.strftime('%Y-%m-%d')} at {dt.strftime('%H:%M')}"

    return "no open slots found in the next 7 days"


async def get_all_appointments(date_filter: Optional[str] = None) -> list:
    db = await _adb()
    query = db.table("appointments").select("*").order("date").order("time")
    if date_filter:
        query = query.eq("date", date_filter)
    result = await query.execute()
    return result.data or []


async def cancel_appointment(appointment_id: str) -> bool:
    db = await _adb()
    result = await (
        db.table("appointments")
        .update({"status": "cancelled"})
        .eq("id", appointment_id)
        .eq("status", "booked")
        .execute()
    )
    return len(result.data or []) > 0


# ── Call logs ────────────────────────────────────────────────────────────────

async def log_call(
    phone_number: str,
    lead_name: Optional[str],
    outcome: str,
    reason: str,
    duration_seconds: int,
) -> None:
    db = await _adb()
    await db.table("call_logs").insert({
        "id": str(uuid.uuid4()),
        "phone_number": phone_number,
        "lead_name": lead_name,
        "outcome": outcome,
        "reason": reason,
        "duration_seconds": duration_seconds,
        "timestamp": datetime.now().isoformat(),
    }).execute()


async def get_all_calls(page: int = 1, limit: int = 20) -> list:
    db = await _adb()
    offset = (page - 1) * limit
    result = await (
        db.table("call_logs")
        .select("*")
        .order("timestamp", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )
    return result.data or []


# ── Stats ─────────────────────────────────────────────────────────────────────

async def get_stats() -> dict:
    db = await _adb()
    rows = (await db.table("call_logs").select("outcome, duration_seconds").execute()).data or []

    total_calls    = len(rows)
    booked         = sum(1 for r in rows if r.get("outcome") == "booked")
    not_interested = sum(1 for r in rows if r.get("outcome") == "not_interested")
    durations      = [r["duration_seconds"] for r in rows if r.get("duration_seconds")]
    avg_dur        = sum(durations) / len(durations) if durations else 0
    booking_rate   = round((booked / total_calls * 100) if total_calls else 0, 1)

    return {
        "total_calls": total_calls,
        "booked": booked,
        "not_interested": not_interested,
        "avg_duration_seconds": round(avg_dur, 1),
        "booking_rate_percent": booking_rate,
    }
