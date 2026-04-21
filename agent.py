# =============================================================================
# OUTBOUND AI APPOINTMENT BOOKING AGENT
# =============================================================================
#
# ARCHITECTURE DECISION:
#   PRIMARY PATH  → google.realtime.RealtimeModel (livekit-plugins-google >= 1.0)
#     Wraps Google Gemini Live API (gemini-2.0-flash-live-001).
#     Audio-in → Gemini Live → Audio-out in a single WebSocket session.
#     This collapses STT + LLM + TTS into ONE API call — no Deepgram, no
#     separate TTS needed. Cheapest possible path.
#
#   FALLBACK PATH → google.beta.realtime.RealtimeModel (older plugin versions)
#     Identical semantics; the module was graduated from beta to stable.
#
#   PIPELINE PATH → Deepgram STT + google.LLM + google.TTS
#     Activated when USE_GEMINI_REALTIME=false in .env.
#     Use this only if the Gemini Live WebSocket is unreliable in your region.
#
# COST BREAKDOWN (per 1-minute call, all-in):
#   Vobiz SIP trunk         ≈ ₹1.00 / min   (fixed telephony cost)
#   Gemini 2.0 Flash Live   ≈ $0.000 / min   (free tier up to quota, then
#                                              $0.075 / 1M audio tokens ≈ ₹0.03/min)
#   LiveKit Cloud           ≈ $0.002 / min   ≈ ₹0.17 / min   (free tier exists)
#   Deepgram STT            ≈ $0.007 / min   ≈ ₹0.58 / min   (PIPELINE PATH ONLY)
#   Google TTS              ≈ $0.000 / min   ≈ ₹0.01 / min   (PIPELINE PATH ONLY)
#   ─────────────────────────────────────────────────────────────────────
#   REALTIME PATH TOTAL     ≈ ₹1.20 / min   ✅ under ₹1.50 target
#   PIPELINE PATH TOTAL     ≈ ₹1.78 / min   ⚠️  slightly over; use realtime
#
# =============================================================================

import json
import logging
import os
import ssl
import certifi
from typing import Optional

from dotenv import load_dotenv

# Patch ssl.create_default_context so aiohttp (used by LiveKit) picks up
# certifi's CA bundle. Must happen before any network library is imported.
_orig_create_default_context = ssl.create_default_context

def _certifi_create_default_context(purpose=ssl.Purpose.SERVER_AUTH, **kwargs):
    if not kwargs.get("cafile") and not kwargs.get("capath") and not kwargs.get("cadata"):
        kwargs["cafile"] = certifi.where()
    return _orig_create_default_context(purpose, **kwargs)

ssl.create_default_context = _certifi_create_default_context

from livekit import agents, api
from livekit.agents import Agent, AgentSession, RoomInputOptions
from livekit.plugins import noise_cancellation, silero

from db import init_db, log_error
from prompts import build_prompt
from tools import AppointmentTools


async def _log(level: str, msg: str, detail: str = "") -> None:
    """Write a structured log entry to Supabase and Python logger."""
    if level == "info":
        logger.info(msg)
    elif level == "warning":
        logger.warning(msg)
    else:
        logger.error(msg)
    try:
        await log_error("agent", msg, detail, level)
    except Exception:
        pass

load_dotenv(".env")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("outbound-agent")

# Module-level constants are read after load_db_settings_to_env() runs in __main__,
# so DB-saved values win over .env values for these keys.
SIP_DOMAIN = os.getenv("VOBIZ_SIP_DOMAIN", "")


def load_db_settings_to_env() -> None:
    """
    Read every row from the Supabase settings table (synchronously) and push
    non-empty values into os.environ BEFORE the LiveKit worker starts.
    DB-saved API keys override .env values at startup.
    Called once in __main__ — not at import time.
    """
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_SERVICE_KEY", "")
    if not url or not key:
        logger.warning("SUPABASE_URL or SUPABASE_SERVICE_KEY not set — skipping DB settings load")
        return
    try:
        from supabase import create_client
        client = create_client(url, key)
        result = client.table("settings").select("key, value").execute()
        for row in (result.data or []):
            if row.get("value"):
                os.environ[row["key"]] = row["value"]
                logger.debug("Loaded setting from Supabase: %s", row["key"])
    except Exception as exc:
        logger.warning("Could not load settings from Supabase: %s", exc)

# ---------------------------------------------------------------------------
# Try to import Google plugin paths in order of preference
# ---------------------------------------------------------------------------

_google_realtime = None      # google.realtime.RealtimeModel  (stable, >= 1.0)
_google_beta_realtime = None  # google.beta.realtime.RealtimeModel  (legacy)
_google_llm = None            # google.LLM  (pipeline fallback)
_google_tts = None            # google.TTS  (pipeline fallback)

try:
    from livekit.plugins import google as _gp
    try:
        _google_realtime = _gp.realtime.RealtimeModel
        logger.info("Loaded google.realtime.RealtimeModel (stable path)")
    except AttributeError:
        pass

    try:
        _google_beta_realtime = _gp.beta.realtime.RealtimeModel
        logger.info("Loaded google.beta.realtime.RealtimeModel (beta path)")
    except AttributeError:
        pass

    try:
        _google_llm = _gp.LLM
        _google_tts = _gp.TTS
    except AttributeError:
        pass

except ImportError:
    logger.warning("livekit-plugins-google not installed — cannot use Gemini")

# Deepgram as pipeline-mode STT fallback
_deepgram_stt = None
try:
    from livekit.plugins import deepgram as _dg
    _deepgram_stt = _dg.STT
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

def _build_session(tools: list, system_prompt: str) -> AgentSession:
    """
    Build an AgentSession choosing the best available AI backend.
    Reads GEMINI_MODEL / GEMINI_TTS_VOICE / USE_GEMINI_REALTIME from os.environ
    at call time so DB-saved settings (loaded in __main__) take effect.
    Order of preference:
      1. google.realtime.RealtimeModel  (single-API Gemini Live audio)
      2. google.beta.realtime.RealtimeModel  (same, older package naming)
      3. Deepgram STT + google.LLM + google.TTS  (pipeline fallback)
    """
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-live-001")
    gemini_voice = os.getenv("GEMINI_TTS_VOICE", "Aoede")
    use_realtime = os.getenv("USE_GEMINI_REALTIME", "true").lower() != "false"

    RealtimeClass = _google_realtime or (_google_beta_realtime if use_realtime else None)

    if use_realtime and RealtimeClass is not None:
        logger.info(
            "SESSION MODE: Gemini Live realtime (%s, voice=%s)", gemini_model, gemini_voice
        )
        return AgentSession(
            llm=RealtimeClass(
                model=gemini_model,
                voice=gemini_voice,
                instructions=system_prompt,
            ),
            tools=tools,
        )

    # Pipeline fallback
    if _google_llm is None:
        raise RuntimeError(
            "No Google AI backend available. "
            "Run: pip install 'livekit-plugins-google>=1.0'"
        )

    logger.info("SESSION MODE: pipeline (Deepgram STT + Gemini LLM + Google TTS)")

    stt = _deepgram_stt(model="nova-3", language="multi") if _deepgram_stt else None
    tts = _google_tts() if _google_tts else None

    return AgentSession(
        stt=stt,
        llm=_google_llm(model="gemini-2.0-flash"),
        tts=tts,
        vad=silero.VAD.load(),
        tools=tools,
    )


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class OutboundAssistant(Agent):
    """Minimal Agent — all behaviour comes from the injected system prompt."""

    def __init__(self, instructions: str) -> None:
        super().__init__(instructions=instructions)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

async def entrypoint(ctx: agents.JobContext) -> None:
    """
    Main entrypoint called by the LiveKit worker for every dispatched job.

    Metadata fields read from ctx.job.metadata (JSON):
      phone_number   – E.164 number to dial (required for outbound)
      lead_name      – Name of the person being called
      business_name  – Name shown in the agent's persona
      service_type   – Service being offered
      system_prompt  – Full custom prompt (overrides default template if set)
    """
    await _log("info", f"Job started — room: {ctx.room.name}")

    # ------------------------------------------------------------------
    # Parse job metadata
    # ------------------------------------------------------------------
    phone_number: Optional[str] = None
    lead_name = "there"
    business_name = "our company"
    service_type = "our service"
    custom_prompt: Optional[str] = None

    if ctx.job.metadata:
        try:
            data = json.loads(ctx.job.metadata)
            phone_number = data.get("phone_number")
            lead_name = data.get("lead_name", lead_name)
            business_name = data.get("business_name", business_name)
            service_type = data.get("service_type", service_type)
            custom_prompt = data.get("system_prompt")
        except (json.JSONDecodeError, AttributeError):
            await _log("warning", "Invalid JSON in job metadata")

    await _log("info",
        f"Call job received — phone={phone_number} lead={lead_name} biz={business_name}",
        f"service={service_type} room={ctx.room.name}",
    )

    system_prompt = build_prompt(
        lead_name=lead_name,
        business_name=business_name,
        service_type=service_type,
        custom_prompt=custom_prompt,
    )

    # ------------------------------------------------------------------
    # Build tools and session
    # ------------------------------------------------------------------
    tool_ctx = AppointmentTools(ctx, phone_number, lead_name)

    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-live-001")
    await _log("info", f"Building AI session — model={gemini_model}")

    session = _build_session(tools=tool_ctx.all_tools, system_prompt=system_prompt)

    # ------------------------------------------------------------------
    # Connect to room and start session
    # ------------------------------------------------------------------
    await ctx.connect()
    await _log("info", f"Connected to LiveKit room: {ctx.room.name}")

    await session.start(
        room=ctx.room,
        agent=OutboundAssistant(instructions=system_prompt),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVCTelephony(),
            close_on_disconnect=True,
        ),
    )
    await _log("info", "Agent session started — AI ready")

    # ------------------------------------------------------------------
    # Outbound SIP dial
    # ------------------------------------------------------------------
    if phone_number:
        trunk_id = os.getenv("OUTBOUND_TRUNK_ID")
        if not trunk_id:
            await _log("error", "OUTBOUND_TRUNK_ID not set — cannot place outbound call")
            ctx.shutdown()
            return

        await _log("info", f"Dialing {phone_number} via SIP trunk {trunk_id}")
        try:
            await ctx.api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    room_name=ctx.room.name,
                    sip_trunk_id=trunk_id,
                    sip_call_to=phone_number,
                    participant_identity=f"sip_{phone_number}",
                    wait_until_answered=True,
                )
            )
            await _log("info", f"Call ANSWERED — {phone_number} picked up, agent speaking now")

            await session.generate_reply(
                instructions=(
                    f"The call just connected. Greet the lead and ask if you're speaking "
                    f"with {lead_name}, as per your instructions."
                )
            )

        except Exception as exc:
            await _log("error", f"SIP dial FAILED for {phone_number}: {exc}",
                       f"trunk_id={trunk_id} room={ctx.room.name}")
            ctx.shutdown()
    else:
        await _log("info", "No phone_number in metadata — treating as inbound/web call")
        await session.generate_reply(instructions="Greet the caller warmly.")


# ---------------------------------------------------------------------------
# Worker entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_db()                    # create tables if they don't exist
    load_db_settings_to_env()    # push DB-saved API keys into os.environ before worker starts
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="outbound-caller",
        )
    )
