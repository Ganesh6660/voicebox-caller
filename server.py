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
the system prompt -> synthesize the reply (Voicebox, your cloned voice) ->
resample/encode to Twilio's format -> stream it back into the call.

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
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from openai import OpenAI
from pydub import AudioSegment

load_dotenv()

VOICEBOX_BASE_URL = os.environ.get("VOICEBOX_BASE_URL", "http://127.0.0.1:17493")
VOICEBOX_VOICE_ID = os.environ["VOICEBOX_VOICE_ID"]  # the cloned voice profile to speak with
# Preset-voice profiles are locked to the engine they were created with (e.g. "kokoro");
# cloned-voice profiles are more flexible and can usually leave this as the Voicebox default.
VOICEBOX_ENGINE = os.environ.get("VOICEBOX_ENGINE") or None
# OpenRouter (openrouter.ai) proxies many models behind one OpenAI-compatible API.
# ":free"-suffixed models cost nothing but are rate-limited.
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "minimax/minimax-m2.7:free")
PUBLIC_BASE_URL = os.environ["PUBLIC_BASE_URL"].rstrip("/")

llm_client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
app = FastAPI()

# Twilio Media Streams use 8kHz, 8-bit mu-law, mono, sent as ~20ms (160-byte) frames.
SAMPLE_RATE = 8000
FRAME_BYTES = 160
SILENCE_RMS_THRESHOLD = 400   # tune against your mic/line noise
SILENCE_MS_TO_END_TURN = 700  # pause length that means "they're done talking"


@app.api_route("/twiml", methods=["GET", "POST"])
async def twiml(request: Request):
    context = request.query_params.get("context", "")
    ws_url = PUBLIC_BASE_URL.replace("https://", "wss://").replace("http://", "ws://")
    twiml_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{ws_url}/media-stream">
      <Parameter name="context" value="{context}" />
    </Stream>
  </Connect>
</Response>"""
    return Response(content=twiml_xml, media_type="text/xml")


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    await websocket.accept()

    stream_sid = None
    context = ""
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
                context = msg["start"].get("customParameters", {}).get("context", "")
                # Optional: speak an opening line right away instead of waiting for them to talk first.
                opener = generate_reply(context, conversation, opener=True)
                conversation.append({"role": "assistant", "content": opener})
                await speak(websocket, stream_sid, opener)

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
                            reply = generate_reply(context, conversation)
                            conversation.append({"role": "assistant", "content": reply})
                            await speak(websocket, stream_sid, reply)

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


def generate_reply(context: str, conversation: list, opener: bool = False) -> str:
    """Ask the LLM what to say next, given your original context and the conversation so far."""
    system_prompt = (
        "You are making a phone call on behalf of the user described below. "
        "Stay on topic, keep replies short and natural (like real speech, not an essay), "
        "and pursue the goal the user gave you. If the goal is accomplished or the other "
        "person wants to end the call, wrap up politely. "
        "Reply with ONLY the words to speak aloud — no stage directions, no asterisked "
        "actions, no narration, no quotation marks around the line.\n\n"
        f"Call goal / context from the user: {context}"
    )

    if opener:
        turns = [{"role": "user", "content": "Start the call now with a brief, natural opening line."}]
    else:
        turns = conversation

    response = llm_client.chat.completions.create(
        model=OPENROUTER_MODEL,
        max_tokens=200,
        messages=[{"role": "system", "content": system_prompt}, *turns],
    )
    return response.choices[0].message.content or ""


def synthesize_speech(text: str) -> bytes:
    """Render `text` in the configured Voicebox profile and return WAV bytes.

    Voicebox's /generate is async: it hands back a generation id immediately,
    and /generate/{id}/status is a server-sent-events stream that keeps the
    connection open, pushing status updates until the generation finishes.
    """
    payload = {"profile_id": VOICEBOX_VOICE_ID, "text": text}
    if VOICEBOX_ENGINE:
        payload["engine"] = VOICEBOX_ENGINE
    resp = requests.post(f"{VOICEBOX_BASE_URL}/generate", json=payload, timeout=30)
    resp.raise_for_status()
    generation_id = resp.json()["id"]

    status = None
    with requests.get(
        f"{VOICEBOX_BASE_URL}/generate/{generation_id}/status", stream=True, timeout=60
    ) as status_resp:
        status_resp.raise_for_status()
        for line in status_resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            event = json.loads(line[len("data: "):])
            status = event["status"]
            if status in ("completed", "failed", "error"):
                break

    if status != "completed":
        raise RuntimeError(f"Voicebox generation {generation_id} ended with status={status!r}")

    audio_resp = requests.get(
        f"{VOICEBOX_BASE_URL}/history/{generation_id}/export-audio", timeout=30
    )
    audio_resp.raise_for_status()
    return audio_resp.content


async def speak(websocket: WebSocket, stream_sid: str, text: str):
    """Synthesize `text` and stream it into the live call as 20ms mu-law frames."""
    wav_bytes = synthesize_speech(text)
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
