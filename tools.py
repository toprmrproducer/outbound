import asyncio
import logging
import os
import time
from typing import Optional

from livekit import agents, api
from livekit.agents import llm

from db import check_slot, get_next_available, insert_appointment, log_call, log_error

logger = logging.getLogger("appointment-tools")

async def _log(msg: str, detail: str = "", level: str = "info") -> None:
    try:
        await log_error("agent", msg, detail, level)
    except Exception:
        pass


class AppointmentTools(llm.ToolContext):
    """All function tools available to the appointment-booking agent."""

    def __init__(
        self,
        ctx: agents.JobContext,
        phone_number: Optional[str] = None,
        lead_name: Optional[str] = None,
    ):
        # Set instance attrs before calling super so bound methods are ready
        self.ctx = ctx
        self.phone_number = phone_number
        self.lead_name = lead_name
        self._call_start_time = time.time()
        self._sip_domain = os.getenv("VOBIZ_SIP_DOMAIN", "")
        super().__init__(tools=[])

    @property
    def all_tools(self) -> list:
        return [
            self.check_availability,
            self.book_appointment,
            self.end_call,
            self.transfer_to_human,
            self.send_sms_confirmation,
        ]

    # ------------------------------------------------------------------ #
    #  Tool 1: check_availability                                          #
    # ------------------------------------------------------------------ #

    @llm.function_tool
    async def check_availability(self, date: str, time: str) -> str:
        """
        Check whether a date/time slot is available for booking.
        Call this BEFORE attempting to book whenever the lead proposes a date/time.
        date format: YYYY-MM-DD  |  time format: HH:MM (24-hour clock)
        Returns 'available' or 'unavailable: next available slot is <slot>'.
        """
        try:
            if await check_slot(date, time):
                await _log(f"Tool: check_availability({date} {time}) → available")
                return "available"
            next_slot = await get_next_available(date, time)
            await _log(f"Tool: check_availability({date} {time}) → unavailable, next: {next_slot}")
            return f"unavailable: next available slot is {next_slot}"
        except Exception as exc:
            logger.error("check_availability error: %s", exc)
            return "Unable to check availability right now — please suggest a date and I will confirm."

    # ------------------------------------------------------------------ #
    #  Tool 2: book_appointment                                            #
    # ------------------------------------------------------------------ #

    @llm.function_tool
    async def book_appointment(
        self, name: str, phone: str, date: str, time: str, service: str
    ) -> str:
        """
        Book an appointment after the lead has verbally confirmed date, time, and service.
        Call ONLY after the lead confirms all details.
        name: lead's full name
        phone: lead's phone number with country code
        date: YYYY-MM-DD
        time: HH:MM (24-hour clock)
        service: service type requested
        Returns a confirmation message with a short booking ID.
        """
        try:
            await _log(f"Tool: book_appointment — {name} {phone} {date} {time} {service}")
            booking_id = await insert_appointment(name, phone, date, time, service)
            await _log(f"Tool: booking CONFIRMED — ID={booking_id} for {name}")
            return (
                f"Confirmed! Your booking ID is {booking_id}. "
                f"We'll see you on {date} at {time} for your {service}."
            )
        except Exception as exc:
            logger.error("book_appointment error: %s", exc)
            return (
                "I had a technical issue saving the booking. "
                "Our team will call you back to confirm shortly."
            )

    # ------------------------------------------------------------------ #
    #  Tool 3: end_call                                                    #
    # ------------------------------------------------------------------ #

    @llm.function_tool
    async def end_call(self, outcome: str, reason: str = "") -> str:
        """
        End the call gracefully after logging the outcome.
        Call when:
        - Appointment booked successfully        → outcome='booked'
        - Lead is not interested                 → outcome='not_interested'
        - Wrong number reached                   → outcome='wrong_number'
        - Lead asked to call back later          → outcome='callback_requested'
        - Voicemail detected                     → outcome='voicemail'
        outcome: one of the values above
        reason: brief free-text description of why the call ended
        """
        await _log(f"Tool: end_call — outcome={outcome} reason={reason}", level="info")
        duration = int(time.time() - self._call_start_time)
        try:
            await log_call(
                phone_number=self.phone_number or "unknown",
                lead_name=self.lead_name,
                outcome=outcome,
                reason=reason,
                duration_seconds=duration,
            )
        except Exception as exc:
            logger.error("Failed to log call outcome: %s", exc)

        try:
            await self.ctx.room.disconnect()
        except Exception as exc:
            logger.warning("Room disconnect error: %s", exc)

        return "Call ended."

    # ------------------------------------------------------------------ #
    #  Tool 4: transfer_to_human                                           #
    # ------------------------------------------------------------------ #

    @llm.function_tool
    async def transfer_to_human(self, reason: str) -> str:
        """
        Transfer the call to a human agent using SIP REFER.
        Call when:
        - Lead explicitly asks to speak to a human
        - Lead raises a complex objection the AI cannot resolve
        - Lead is angry or upset
        reason: brief description of why the transfer is happening
        """
        destination = os.getenv("DEFAULT_TRANSFER_NUMBER", "")
        if not destination:
            return "Transfer unavailable: no fallback number is configured."

        # Build a valid SIP URI from the raw phone number
        if "@" not in destination:
            clean = destination.replace("tel:", "").replace("sip:", "")
            if self._sip_domain:
                destination = f"sip:{clean}@{self._sip_domain}"
            else:
                destination = f"tel:{clean}"
        elif not destination.startswith("sip:"):
            destination = f"sip:{destination}"

        # Identify the SIP participant to transfer
        participant_identity = None
        if self.phone_number:
            participant_identity = f"sip_{self.phone_number}"
        else:
            for p in self.ctx.room.remote_participants.values():
                participant_identity = p.identity
                break

        if not participant_identity:
            return "Transfer failed: could not identify the caller's participant."

        try:
            await self.ctx.api.sip.transfer_sip_participant(
                api.TransferSIPParticipantRequest(
                    room_name=self.ctx.room.name,
                    participant_identity=participant_identity,
                    transfer_to=destination,
                    play_dialtone=False,
                )
            )
            logger.info("Transfer initiated → %s (reason: %s)", destination, reason)
            await _log(f"Tool: transfer_to_human → {destination} | reason: {reason}")
            return "Transferring you to a human agent now. Please hold for a moment."
        except Exception as exc:
            logger.error("Transfer failed: %s", exc)
            return "I wasn't able to complete the transfer. Please call us back directly."

    # ------------------------------------------------------------------ #
    #  Tool 5: send_sms_confirmation (optional — skips if Twilio absent)  #
    # ------------------------------------------------------------------ #

    @llm.function_tool
    async def send_sms_confirmation(self, phone: str, message: str) -> str:
        """
        Send an SMS confirmation to the lead after a successful booking.
        Call ONLY after book_appointment succeeds.
        Skips silently if Twilio credentials are not configured.
        phone: lead's phone number with country code
        message: the confirmation text to send
        """
        sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        token = os.getenv("TWILIO_AUTH_TOKEN", "")
        from_num = os.getenv("TWILIO_FROM_NUMBER", "")

        if not (sid and token and from_num):
            return "SMS skipped: Twilio not configured."

        try:
            from twilio.rest import Client  # lazy import — only needed if Twilio is configured

            loop = asyncio.get_event_loop()
            client = Client(sid, token)
            # Twilio client is synchronous; run in thread pool to avoid blocking
            await loop.run_in_executor(
                None,
                lambda: client.messages.create(body=message, from_=from_num, to=phone),
            )
            logger.info("SMS confirmation sent to %s", phone)
            await _log(f"Tool: send_sms_confirmation → {phone}")
            return f"SMS confirmation sent to {phone}."
        except Exception as exc:
            logger.error("SMS send failed: %s", exc)
            return "SMS delivery failed, but your booking is confirmed in our system."
