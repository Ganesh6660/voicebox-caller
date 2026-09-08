"""
Real-time AI phone-call server.

Handles two things:
  1. POST /answer       -> Plivo calls this once the outbound call connects
                            (or an inbound call arrives). Responds with
                            PlivoXML that opens a bidirectional audio Stream
                            back to this server's WebSocket.
  2. WS   /media-stream  -> Plivo streams the caller's audio here in real
                            time, and expects audio frames streamed back the
                            same way.

Loop per turn: buffer caller audio until a pause is detected -> transcribe
(Whisper via Voicebox) -> ask the LLM for a reply, using your --context as
the system prompt -> synthesize the reply -> resample/encode to Plivo's
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
to talk, and the server hands off to notify_human()/transfer_call_to_human(),
which text you the context (Plivo Messages API) and move the live call over
to HUMAN_PHONE_NUMBER (Plivo's call-transfer API, via the /transfer-xml
endpoint below). This keeps both LLM token spend and AI-side call minutes
small per call, which is the point if cost is the priority.

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

Plivo protocol notes (different from Twilio in a few real ways, not just
naming — see README.md's pipeline section):
  - The <Stream> element's WebSocket URL is its inner text content, not a
    `url` attribute: <Stream ...>wss://host/media-stream</Stream>.
  - `extraHeaders` on <Stream> is capped at 512 bytes and alphanumeric-only,
    so it can't carry a free-text --context string. Context/language/the
    call's CallUUID are passed as query params on the WebSocket URL instead.
  - Incoming events over the WebSocket are still JSON, same shape family as
    Twilio (`start`/`media`/`stop`), but the start event's stream id field
    is `streamId`, not `streamSid`.
  - Outgoing audio (server -> Plivo) is JSON too, NOT raw binary frames —
    but a different envelope than Twilio's: {"event": "playAudio", "media":
    {"contentType", "sampleRate", "payload"}}, no streamSid needed.
  - The call's CallUUID is most reliably captured from the POST body Plivo
    sends to answer_url, not parsed out of the WebSocket start event — this
    server does that and threads it through as a query param.
"""

import asyncio
import audioop
import base64
import io
import json
import os
import urllib.parse
import xml.sax.saxutils

import plivo
import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydub import AudioSegment

from indic_tts import synthesize_indic_wav_bytes

load_dotenv()

VOICEBOX_BASE_URL = os.environ.get("VOICEBOX_BASE_URL", "http://127.0.0.1:17493")
VOICEBOX_VOICE_ID = os.environ["VOICEBOX_VOICE_ID"]  # a Voicebox profile_id (cloned or preset)
VOICEBOX_ENGINE = os.environ.get("VOICEBOX_ENGINE", "kokoro")  # must match VOICEBOX_VOICE_ID's engine
PUBLIC_BASE_URL = os.environ["PUBLIC_BASE_URL"].rstrip("/")

# LLM: OpenRouter if OPENROUTER_API_KEY is set, else Anthropic direct. Lets you
# switch providers purely via .env — no code changes needed either way. Both
# get a "triage" role (verify + decide, not deep reasoning), so a cheap/fast
# model is the right default; override via OPENROUTER_MODEL/ANTHROPIC_MODEL in
# .env for calls that need the AI to carry more of the conversation itself.
if os.environ.get("OPENROUTER_API_KEY"):
    from openai import OpenAI

    LLM_PROVIDER = "openrouter"
    LLM_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-haiku-4.5")
    llm_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
else:
    from anthropic import Anthropic

    LLM_PROVIDER = "anthropic"
    LLM_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    llm_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Where a verified/genuine caller gets handed off to. E.164 format.
HUMAN_PHONE_NUMBER = os.environ.get("HUMAN_PHONE_NUMBER", "")

# Same Plivo credentials place_call.py uses — server.py needs its own client
# to actually place the handoff SMS and drive the live-call transfer.
PLIVO_AUTH_ID = os.environ["PLIVO_AUTH_ID"]
PLIVO_AUTH_TOKEN = os.environ["PLIVO_AUTH_TOKEN"]
PLIVO_FROM_NUMBER = os.environ["PLIVO_FROM_NUMBER"]

plivo_client = plivo.RestClient(PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN)
app = FastAPI()

# Plivo audio streaming, like Twilio, uses 8kHz 8-bit mu-law mono, sent as
# ~20ms (160-byte) frames — set via contentType="audio/x-mulaw;rate=8000" on
# <Stream> below, which keeps all the existing audioop encode/decode as-is.
SAMPLE_RATE = 8000
FRAME_BYTES = 160
SILENCE_RMS_THRESHOLD = 400   # tune against your mic/line noise
SILENCE_MS_TO_END_TURN = 700  # pause length that means "they're done talking"


@app.post("/answer")
async def answer(request: Request):
    """Plivo's answer_url webhook. context/language arrive as query params on
    this URL itself (set by place_call.py); CallUUID arrives in the POST body
    Plivo sends here — capture it now rather than relying on the WebSocket
    start event, and thread all three through as query params on the Stream
    URL since extraHeaders can't carry free-text context."""
    context = request.query_params.get("context", "")
    language = request.query_params.get("language", "en")

    form = await request.form()
    call_uuid = form.get("CallUUID", "")

    ws_url = PUBLIC_BASE_URL.replace("https://", "wss://").replace("http://", "ws://")
    stream_url = (
        f"{ws_url}/media-stream"
        f"?context={urllib.parse.quote(context)}"
        f"&language={urllib.parse.quote(language)}"
        f"&call_uuid={urllib.parse.quote(call_uuid)}"
    )

    # stream_url's query string has literal "&" separators — those are XML
    # metacharacters, so the URL must be XML-escaped before going into the
    # <Stream> element's text content, or Plivo will fail to parse the XML.
    stream_url_xml_safe = xml.sax.saxutils.escape(stream_url)

    plivo_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000">{stream_url_xml_safe}</Stream>
</Response>"""
    return Response(content=plivo_xml, media_type="text/xml")


@app.post("/transfer-xml")
async def transfer_xml():
    """aleg_url target for transfer_call_to_human()'s Call-transfer API request below.
    Plivo fetches this and swaps it in for the call's current instructions — since the
    A-leg was running the <Stream> from /answer, returning a <Dial> here is what actually
    ends the AI's WebSocket leg and connects the live caller straight to a human."""
    if not HUMAN_PHONE_NUMBER:
        # Nothing to dial — end the call cleanly rather than fail the transfer silently.
        return Response(
            content='<?xml version="1.0" encoding="UTF-8"?><Response><Hangup/></Response>',
            media_type="text/xml",
        )
    plivo_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Dial callerId="{xml.sax.saxutils.escape(PLIVO_FROM_NUMBER)}">
    <Number>{xml.sax.saxutils.escape(HUMAN_PHONE_NUMBER)}</Number>
  </Dial>
</Response>"""
    return Response(content=plivo_xml, media_type="text/xml")


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    await websocket.accept()

    call_sid = websocket.query_params.get("call_uuid", "")
    context = websocket.query_params.get("context", "")
    language = websocket.query_params.get("language", "en")
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
                # Fallback only — call_sid normally already came from the
                # answer webhook's CallUUID via the query params above.
                if not call_sid:
                    call_sid = msg["start"].get("callId") or msg["start"].get("call_uuid", "")
                # Optional: speak an opening line right away instead of waiting for them to talk first.
                opener, transfer = generate_reply(context, conversation, language, opener=True)
                conversation.append({"role": "assistant", "content": opener})
                await speak(websocket, opener, language)
                if transfer:
                    await handoff_to_human(websocket, call_sid, language, transfer)
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
                            await speak(websocket, reply, language)
                            if transfer:
                                await handoff_to_human(websocket, call_sid, language, transfer)
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

# Same tool, OpenAI/OpenRouter's function-calling shape (input_schema -> parameters,
# wrapped in {"type": "function", "function": {...}}) — used only in the openrouter branch.
TRANSFER_TOOL_OPENAI = {
    "type": "function",
    "function": {
        "name": TRANSFER_TOOL["name"],
        "description": TRANSFER_TOOL["description"],
        "parameters": TRANSFER_TOOL["input_schema"],
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

    if LLM_PROVIDER == "openrouter":
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=200,
            messages=[{"role": "system", "content": system_prompt}] + messages,
            tools=[TRANSFER_TOOL_OPENAI],
            extra_headers={"HTTP-Referer": PUBLIC_BASE_URL, "X-Title": "Voicebox Caller"},
        )
        choice = response.choices[0].message
        text = choice.content or ""
        transfer = None
        for tool_call in choice.tool_calls or []:
            if tool_call.function.name == "transfer_to_human":
                args = json.loads(tool_call.function.arguments)
                transfer = {"reason": args.get("reason", ""), "summary": args.get("summary", "")}
                break
        return text, transfer

    response = llm_client.messages.create(
        model=LLM_MODEL,
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


async def handoff_to_human(websocket: WebSocket, call_sid: str, language: str, transfer: dict):
    """Caller's been verified — get a human the context, then move the actual call over to
    them via Plivo's Messages and Call-transfer APIs."""
    notify_human(transfer["reason"], transfer["summary"])
    await transfer_call_to_human(call_sid)


def notify_human(reason: str, summary: str):
    """Text yourself (or the on-call agent) the handoff context via Plivo's Messages API —
    the cheapest way to get a human the context before the call lands on them. Failure here
    shouldn't block the actual call transfer, so it's caught and logged rather than raised."""
    if not HUMAN_PHONE_NUMBER:
        print(f"[notify_human] (no HUMAN_PHONE_NUMBER set) {reason}: {summary}")
        return
    try:
        response = plivo_client.messages.create(
            src=PLIVO_FROM_NUMBER,
            dst=HUMAN_PHONE_NUMBER,
            text=f"Incoming handoff — {reason}: {summary}",
        )
        print(f"[notify_human] SMS sent to {HUMAN_PHONE_NUMBER}, message_uuid={response.message_uuid}")
    except Exception as exc:  # Plivo SDK raises plivo.exceptions.PlivoRestError, etc.
        print(f"[notify_human] FAILED to SMS {HUMAN_PHONE_NUMBER} -> {reason}: {summary} ({exc})")


async def transfer_call_to_human(call_sid: str):
    """Move the live call to HUMAN_PHONE_NUMBER via Plivo's call-transfer API: it re-points
    the call's A-leg at /transfer-xml, which returns a <Dial> to the human — that's what
    actually ends the AI's WebSocket leg and connects the caller, not the `break` in the
    WebSocket loop (that just stops this server from listening/speaking on its side)."""
    if not call_sid:
        print("[transfer_call_to_human] no call_sid available — can't transfer, caller will just hear silence")
        return
    if not HUMAN_PHONE_NUMBER:
        print(f"[transfer_call_to_human] no HUMAN_PHONE_NUMBER set — can't transfer call_sid={call_sid}")
        return
    try:
        response = plivo_client.calls.transfer(
            call_uuid=call_sid,
            legs="aleg",
            aleg_url=f"{PUBLIC_BASE_URL}/transfer-xml",
            aleg_method="POST",
        )
        print(f"[transfer_call_to_human] transferred call_sid={call_sid} to {HUMAN_PHONE_NUMBER}: {response}")
    except Exception as exc:  # Plivo SDK raises plivo.exceptions.PlivoRestError, etc.
        print(f"[transfer_call_to_human] FAILED to transfer call_sid={call_sid} to {HUMAN_PHONE_NUMBER} ({exc})")


def synthesize_speech(text: str, language: str = "en") -> bytes:
    """Render `text` as WAV bytes. English uses Voicebox (your voice profile, via its
    /generate/stream endpoint); Hindi/Kannada use the local quantized MMS-TTS models
    instead (a stock voice, not cloned — see README.md for why, and run
    quantize_export.py once before using these two).

    Uses /generate/stream (not /generate) because /generate is async — it returns a
    job id you'd have to poll and then fetch from /audio/{generation_id}, which adds
    a round trip a live call can't afford. /generate/stream returns the WAV bytes
    directly. `engine` must be passed explicitly: the API defaults to "qwen" (GPU-
    oriented) whenever it's omitted, regardless of the profile's own default engine."""
    if language in ("hi", "kn"):
        return synthesize_indic_wav_bytes(text, language)

    resp = requests.post(
        f"{VOICEBOX_BASE_URL}/generate/stream",
        json={"text": text, "profile_id": VOICEBOX_VOICE_ID, "engine": VOICEBOX_ENGINE, "language": language},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.content


async def speak(websocket: WebSocket, text: str, language: str = "en"):
    """Synthesize `text` and stream it into the live call as 20ms mu-law frames, using
    Plivo's playAudio JSON envelope (event/media/contentType/sampleRate/payload — no
    streamSid, unlike Twilio)."""
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
                    "event": "playAudio",
                    "media": {
                        "contentType": "audio/x-mulaw",
                        "sampleRate": "8000",
                        "payload": base64.b64encode(frame).decode("ascii"),
                    },
                }
            )
        )
        # Pace roughly real-time so Plivo's jitter buffer doesn't choke on a burst.
        await asyncio.sleep(0.02)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
