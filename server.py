"""
Real-time AI phone-call server.

Handles two things:
  1. GET /twiml       -> Twilio calls this once the outbound call connects.
                          Responds with TwiML that opens a bidirectional
                          Media Stream back to this server's WebSocket.
  2. WS  /media-stream -> Twilio streams the caller's audio here in real time,
                          and expects audio frames streamed back the same way.

Loop per turn: buffer caller audio until a pause is detected -> transcribe
(Whisper via Voicebox) -> ask the LLM for a reply, using your --context as
the system prompt -> synthesize the reply -> resample/encode to Twilio's
format -> stream it back into the call.

Speech synthesis is language-routed: English ("en") goes to Voicebox, using
your cloned voice. Hindi ("hi") and Kannada ("kn") go to the local quantized
MMS-TTS models in indic_tts.py instead — see README.md for why (Voicebox's
cloning engines don't currently cover these two well, and the alternative
that does, IndicF5, is too slow on CPU for a live call). Those two use a
stock pretrained voice, not your cloned one. Run quantize_export.py once
before placing a Hindi/Kannada call.

AI + human hybrid: the LLM has a transfer_to_human tool. Its job is narrow —
verify who's calling and what they want, not handle everything — so once
it's satisfied the call is genuine it calls the tool instead of continuing
to talk, and the server hands off to notify_human()/transfer_call_to_human()
(both stubs below — fill in your telephony provider's actual transfer/SMS
API once you've picked one). This keeps both LLM token spend and AI-side
call minutes small per call, which is the point if cost is the priority.

Cost note: the model defaults to Haiku (ANTHROPIC_MODEL env var) rather than
Sonnet. Haiku is roughly half Sonnet's per-token price and is plenty capable
for "verify and decide whether to transfer" — you don't need Sonnet-level
reasoning for triage. Bump it back to a Sonnet model via the env var for
calls that need to carry more of the conversation themselves.

IMPORTANT: the exact request/response shape of Voicebox's /transcribe and
/generate endpoints below is my best inference from its public README, not
confirmed against its source. Voicebox's FastAPI backend auto-serves Swagger
docs — check http://127.0.0.1:17493/docs before your first real run and
adjust `transcribe_audio()` / `synthesize_speech()` to match the actual
field names if they differ.
"""

import asyncio
import audioop
import base64
import io
import json
import os

import requests
import uvicorn
from anthropic import Anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydub import AudioSegment

from indic_tts import synthesize_indic_wav_bytes

load_dotenv()

VOICEBOX_BASE_URL = os.environ.get("VOICEBOX_BASE_URL", "http://127.0.0.1:17493")
VOICEBOX_VOICE_ID = os.environ["VOICEBOX_VOICE_ID"]  # the cloned voice profile to speak with
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
PUBLIC_BASE_URL = os.environ["PUBLIC_BASE_URL"].rstrip("/")

# Haiku by default for cost — this role is triage (verify + decide), not deep
# reasoning. Override in .env with a Sonnet model ID for calls that need more.
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

# Where a verified/genuine caller gets handed off to. E.164 format.
HUMAN_PHONE_NUMBER = os.environ.get("HUMAN_PHONE_NUMBER", "")

anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
app = FastAPI()

# Twilio Media Streams use 8kHz, 8-bit mu-law, mono, sent as ~20ms (160-byte) frames.
SAMPLE_RATE = 8000
FRAME_BYTES = 160
SILENCE_RMS_THRESHOLD = 400   # tune against your mic/line noise
SILENCE_MS_TO_END_TURN = 700  # pause length that means "they're done talking"


@app.get("/twiml")
async def twiml(request: Request):
    context = request.query_params.get("context", "")
    language = request.query_params.get("language", "en")
    ws_url = PUBLIC_BASE_URL.replace("https://", "wss://").replace("http://", "ws://")
    twiml_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{ws_url}/media-stream">
      <Parameter name="context" value="{context}" />
      <Parameter name="language" value="{language}" />
    </Stream>
  </Connect>
</Response>"""
    return Response(content=twiml_xml, media_type="text/xml")


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    await websocket.accept()

    stream_sid = None
    call_sid = None
    context = ""
    language = "en"
    conversation = []  # [{"role": "user"/"assistant", "content": "..."}]

    audio_buffer = bytearray()
    silence_ms = 0
    speaking_started = False

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            event = msg.get("event")

            if event == "start":
                stream_sid = msg["start"]["streamSid"]
                # Twilio/Plivo both include the underlying call's ID here — you'll need it
                # to actually drive a transfer via your provider's REST API.
                call_sid = msg["start"].get("callSid") or msg["start"].get("call_uuid")
                params = msg["start"].get("customParameters", {})
                context = params.get("context", "")
                language = params.get("language", "en")
                # Optional: speak an opening line right away instead of waiting for them to talk first.
                opener, transfer = generate_reply(context, conversation, language, opener=True)
                conversation.append({"role": "assistant", "content": opener})
                await speak(websocket, stream_sid, opener, language)
                if transfer:
                    await handoff_to_human(websocket, stream_sid, call_sid, language, transfer)
                    break

            elif event == "media":
                mulaw_chunk = base64.b64decode(msg["media"]["payload"])
                linear_chunk = audioop.ulaw2lin(mulaw_chunk, 2)
                rms = audioop.rms(linear_chunk, 2)

                if rms > SILENCE_RMS_THRESHOLD:
                    speaking_started = True
                    silence_ms = 0
                    audio_buffer.extend(mulaw_chunk)
                elif speaking_started:
                    silence_ms += 20  # each frame is ~20ms
                    audio_buffer.extend(mulaw_chunk)

                    if silence_ms >= SILENCE_MS_TO_END_TURN:
                        # Caller finished a turn -> transcribe, think, respond.
                        # These are blocking HTTP calls; fine for one call at a time (this
                        # scaffold's use case). For concurrent calls, wrap with
                        # asyncio.to_thread(...) so one call's network wait doesn't stall others.
                        transcript = transcribe_audio(bytes(audio_buffer))
                        audio_buffer.clear()
                        speaking_started = False
                        silence_ms = 0

                        if transcript.strip():
                            conversation.append({"role": "user", "content": transcript})
                            reply, transfer = generate_reply(context, conversation, language)
                            conversation.append({"role": "assistant", "content": reply})
                            await speak(websocket, stream_sid, reply, language)
                            if transfer:
                                await handoff_to_human(websocket, stream_sid, call_sid, language, transfer)
                                break

            elif event == "stop":
                break

    except WebSocketDisconnect:
        pass


def transcribe_audio(mulaw_bytes: bytes) -> str:
    """Send the buffered caller audio to Voicebox's Whisper-backed /transcribe endpoint."""
    linear = audioop.ulaw2lin(mulaw_bytes, 2)
    segment = AudioSegment(data=linear, sample_width=2, frame_rate=SAMPLE_RATE, channels=1)
    wav_io = io.BytesIO()
    segment.export(wav_io, format="wav")
    wav_io.seek(0)

    resp = requests.post(
        f"{VOICEBOX_BASE_URL}/transcribe",
        files={"file": ("audio.wav", wav_io, "audio/wav")},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("text", "")


LANGUAGE_NAMES = {"en": "English", "hi": "Hindi", "kn": "Kannada"}

TRANSFER_TOOL = {
    "name": "transfer_to_human",
    "description": (
        "Call this once you've verified the caller is genuine and understood what they "
        "need — do NOT try to resolve the request yourself, that's the human's job. "
        "Say a brief 'connecting you now' line in your reply text in the same turn you "
        "call this tool."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "One short phrase for why this is being transferred, e.g. 'billing dispute'.",
            },
            "summary": {
                "type": "string",
                "description": (
                    "2-3 sentences a human can read in a few seconds: who's calling, what "
                    "they need, and anything already confirmed (order number, account, etc.)."
                ),
            },
        },
        "required": ["reason", "summary"],
    },
}


def generate_reply(context: str, conversation: list, language: str = "en", opener: bool = False):
    """Ask the LLM what to say next. Returns (reply_text, transfer_or_None) — transfer is a
    {"reason": ..., "summary": ...} dict when the LLM decided this call is genuine and ready
    to hand off to a human, per the transfer_to_human tool below."""
    language_name = LANGUAGE_NAMES.get(language, "English")
    system_prompt = (
        "You are making/answering a phone call on behalf of the user described below. "
        f"Speak only in {language_name} — the whole reply, not just a greeting. "
        "Stay on topic, keep replies short and natural (like real speech, not an essay). "
        "Your job is narrow: verify the caller is genuine and understand what they need — "
        "do not try to resolve their actual request yourself. Once you've got enough to "
        "hand this off, call the transfer_to_human tool (and say a brief 'connecting you "
        "now' line in the same reply). If it's clearly not genuine (spam, wrong number, "
        "no real request), wrap up politely without transferring.\n\n"
        f"Call goal / context from the user: {context}"
    )

    if opener:
        messages = [{"role": "user", "content": "Start the call now with a brief, natural opening line."}]
    else:
        messages = conversation

    response = anthropic_client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=200,
        system=system_prompt,
        tools=[TRANSFER_TOOL],
        messages=messages,
    )

    text = "".join(block.text for block in response.content if block.type == "text")
    transfer = None
    for block in response.content:
        if block.type == "tool_use" and block.name == "transfer_to_human":
            transfer = {"reason": block.input.get("reason", ""), "summary": block.input.get("summary", "")}
            break

    return text, transfer


async def handoff_to_human(websocket: WebSocket, stream_sid: str, call_sid: str, language: str, transfer: dict):
    """Caller's been verified — get a human the context, then move the actual call over to
    them. notify_human() and transfer_call_to_human() are stubs: fill in your telephony
    provider's real SMS/transfer APIs once you've picked one (see README.md)."""
    notify_human(transfer["reason"], transfer["summary"])
    await transfer_call_to_human(call_sid)


def notify_human(reason: str, summary: str):
    """STUB — send yourself (or the on-call agent) the handoff context. Cheapest version:
    one SMS via your telephony provider's messaging API. Fill in once you've picked Plivo/
    Exotel/etc. — this is intentionally not wired to anything yet."""
    if not HUMAN_PHONE_NUMBER:
        print(f"[notify_human] (no HUMAN_PHONE_NUMBER set) {reason}: {summary}")
        return
    # TODO: e.g. Plivo — plivo_client.messages.create(src=..., dst=HUMAN_PHONE_NUMBER,
    #       text=f"{reason}: {summary}")
    print(f"[notify_human] would SMS {HUMAN_PHONE_NUMBER} -> {reason}: {summary}")


async def transfer_call_to_human(call_sid: str):
    """STUB — actually move the live call to HUMAN_PHONE_NUMBER via your provider's REST
    API (a warm/cold transfer or a "Connect"-style redirect), using call_sid to identify
    which call. Twilio: client.calls(call_sid).update(twiml=...) with a <Dial> to the
    human's number. Plivo: a transfer/redirect call against the same call_uuid. This
    scaffold just logs it — the WebSocket loop breaks right after, ending the AI's leg;
    without a real transfer call here, the caller will just hear the line go quiet, so
    don't treat this as done until you've wired in the real API call."""
    print(f"[transfer_call_to_human] would transfer call_sid={call_sid} to {HUMAN_PHONE_NUMBER}")


def synthesize_speech(text: str, language: str = "en") -> bytes:
    """Render `text` as WAV bytes. English uses Voicebox (your cloned voice, via its
    /generate endpoint); Hindi/Kannada use the local quantized MMS-TTS models instead
    (a stock voice, not cloned — see README.md for why, and run quantize_export.py
    once before using these two)."""
    if language in ("hi", "kn"):
        return synthesize_indic_wav_bytes(text, language)

    resp = requests.post(
        f"{VOICEBOX_BASE_URL}/generate",
        json={"text": text, "voice_id": VOICEBOX_VOICE_ID},
        timeout=30,
    )
    resp.raise_for_status()
    # Adjust this if Voicebox returns JSON with a base64 field instead of raw audio bytes —
    # check the actual response shape at http://127.0.0.1:17493/docs.
    return resp.content


async def speak(websocket: WebSocket, stream_sid: str, text: str, language: str = "en"):
    """Synthesize `text` and stream it into the live call as 20ms mu-law frames."""
    wav_bytes = synthesize_speech(text, language)
    segment = AudioSegment.from_file(io.BytesIO(wav_bytes))
    segment = segment.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
    linear_pcm = segment.raw_data
    mulaw = audioop.lin2ulaw(linear_pcm, 2)

    for i in range(0, len(mulaw), FRAME_BYTES):
        frame = mulaw[i : i + FRAME_BYTES]
        if not frame:
            continue
        await websocket.send_text(
            json.dumps(
                {
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": base64.b64encode(frame).decode("ascii")},
                }
            )
        )
        # Pace roughly real-time so Twilio's jitter buffer doesn't choke on a burst.
        await asyncio.sleep(0.02)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
