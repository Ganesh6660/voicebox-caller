# Voicebox Phone Caller

You type/paste some context ("what to talk about"), and the system places a
real phone call and carries a live, two-way conversation on your behalf —
speaking in a chosen voice (your own cloned voice, or a built-in preset one)
without you speaking yourself.

**Read this whole file before running anything.** The code is fully wired up
and has been tested end-to-end (LLM replies, speech generation in three
languages) — the one thing that hasn't been exercised on a real phone call
yet is the Plivo telephony leg itself, since that requires a funded Plivo
account. The pieces that matter most for call quality (audio format,
latency) are called out below.

## Status: what's done vs. what you still need to do

| Piece | Role | Status |
|---|---|---|
| LLM (OpenRouter or Anthropic) | Decides what to say each turn | ✅ Code done, tested |
| Voicebox (English) | Text-to-speech in a chosen voice | ✅ Working, tested |
| Hindi / Kannada speech | Text-to-speech for those two languages | ✅ Working, tested |
| Plivo | Actually dials the phone, streams live call audio | ⛔ **You need to set this up** — nothing else can dial a phone |
| ngrok / public URL | Lets Plivo reach your machine | ⛔ **You need to start this before each session** |

If you're picking this project up fresh: everything under "Setup" below gets
you to a running server. The only genuinely blocking step is getting a Plivo
account and phone number — see "What you need before this runs".

## Why Voicebox alone can't do this

Voicebox (the `jamiepine/voicebox` app) generates speech and clones voices,
but it has no telephony — it can't dial a number or carry audio over a phone
network. This project adds the missing piece: a telephony provider (Plivo)
that places the call and streams the live audio to a small server you run,
which stitches together listening → thinking → speaking.

## The pipeline

```
You type context  ──▶  place_call.py  ──▶  Plivo dials the number
                                                    │
                                    Plivo POSTs to /answer, then opens a
                                    bidirectional audio Stream WebSocket
                                    to server.py (/media-stream)
                                                    │
        caller talks ──▶ audio chunks ──▶  server.py buffers audio
                                                    │
                                          Speech-to-text (Whisper, via Voicebox)
                                                    │
                              transcript + your context ──▶ LLM (OpenRouter or Anthropic)
                                                    │
                                            LLM's reply text
                                                    │
                  English: Voicebox /generate/stream (your chosen voice)
                  Hindi/Kannada: local quantized MMS-TTS model (indic_tts.py)
                                                    │
                       convert to 8kHz mu-law, stream back as a
                       Plivo "playAudio" JSON message
                                                    │
                                     caller hears it in the call
```

This is the same basic architecture products like Bland.ai, Retell AI, and
Vapi use — you're just self-hosting the voice model instead of paying per
minute for one.

## What you need before this runs

1. **A Plivo account** (plivo.com) with a phone number that has Voice
   enabled, plus your `auth_id`/`auth_token` from the console. Plivo is
   pay-as-you-go with no trial-account number-verification restriction like
   Twilio's, but be considerate about who you're calling while testing —
   see the limitations below. Signup may ask for a company/work email; a
   personal Plivo account works fine too if that's not available to you.
2. **Voicebox running locally**, with its API reachable (default
   `http://127.0.0.1:17493`). Download it from the
   [jamiepine/voicebox releases page](https://github.com/jamiepine/voicebox/releases)
   (desktop app for Windows/macOS, or Docker/build-from-source for Linux) and
   launch it — it runs both a UI and a local API server.
   **You do not need to clone your own voice to get started** — Voicebox
   ships with built-in preset voices (see "Choosing a Voicebox voice"
   below) that work with zero setup.
3. **An LLM API key** — either:
   - **OpenRouter** (`OPENROUTER_API_KEY`, recommended for trying this out
     — one key, works with dozens of models), or
   - **Anthropic direct** (`ANTHROPIC_API_KEY`), used automatically as the
     fallback if `OPENROUTER_API_KEY` isn't set.

   See `generate_reply()` in `server.py` if you want to wire in a different
   provider entirely.
4. **A public URL for `server.py`.** Plivo needs to reach your machine over
   the internet. Easiest for testing: `ngrok http 8000` and use the ngrok
   URL. For anything beyond testing, deploy `server.py` on a real host.
5. **ffmpeg** installed (used for audio resampling/encoding).
6. **Windows only:** the
   [Microsoft Visual C++ Redistributable (x64)](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
   must be installed. Without it, `onnxruntime`'s native DLL fails to load
   and `server.py` crashes on startup with `DLL load failed while importing
   onnxruntime_pybind11_state` — this happens even for English-only calls,
   since `indic_tts.py` (and its `onnxruntime` import) is imported
   unconditionally at the top of `server.py`.

## Setup

```bash
cd voicebox-caller
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\Activate.ps1     # Windows PowerShell — use this line instead

pip install -r requirements.txt
cp .env.example .env   # fill in your keys
```

Run the server:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

In another terminal, expose it publicly (for testing):

```bash
ngrok http 8000
```

Put the ngrok HTTPS URL into `.env` as `PUBLIC_BASE_URL`.

Place a call:

```bash
python place_call.py --to "+9198XXXXXXX" \
  --context "Call Rahul, tell him the delivery moved to Tuesday 5pm, ask if that still works for him."
```

That `--context` string becomes the LLM's system prompt — it's the "content"
you're giving it. It drives the whole conversation; the LLM improvises the
back-and-forth around it based on what the other person actually says.

## Choosing your LLM provider

Set **one** of these two in `.env` — `server.py` picks OpenRouter
automatically if its key is present, otherwise it falls back to Anthropic:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=anthropic/claude-haiku-4.5   # must support tool/function calling
```

```bash
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-haiku-4-5-20251001
```

Both default to a cheap/fast model since the LLM's job here is narrow
(verify the caller, decide whether to hand off — see "AI + human hybrid" in
`server.py`), not deep reasoning. Bump to a bigger model via the `_MODEL`
env var if a call needs the AI to carry more of the conversation itself.
If using OpenRouter, double-check your chosen model supports tool calling at
[openrouter.ai/models](https://openrouter.ai/models) — the human-handoff
feature depends on it.

## Choosing a Voicebox voice

`VOICEBOX_VOICE_ID` in `.env` is a Voicebox **profile_id**, not a raw voice
name — you create a profile once via Voicebox's API, then reuse its id.

**Option A — use a built-in preset voice (no recording needed):**

```bash
# List available preset voices for an engine:
curl http://127.0.0.1:17493/profiles/presets/kokoro

# Create a profile from one of them:
curl -X POST http://127.0.0.1:17493/profiles \
  -H "Content-Type: application/json" \
  -d '{"name":"My Voice","voice_type":"preset","preset_engine":"kokoro","preset_voice_id":"am_michael","language":"en","default_engine":"kokoro"}'
```

The response's `"id"` field is your `VOICEBOX_VOICE_ID`.

**Option B — clone your own voice:** use Voicebox's desktop app to record or
import a sample and create a `voice_type: "cloned"` profile, then use that
profile's id the same way.

Either way, also set `VOICEBOX_ENGINE` in `.env` to match the profile's
engine (e.g. `kokoro`). This is required, not optional — Voicebox's
`/generate` API defaults to the GPU-oriented `qwen` engine whenever `engine`
is omitted from a request, regardless of what the profile itself was
created with.

**No GPU?** Stick to `kokoro` or `luxtts` — both are CPU-fast. The other
engines (`qwen`, `chatterbox`, `tada`, etc.) are GPU-oriented and will be
slow-to-unusable on CPU-only hardware.

## The honest limitations

- **Latency is the whole game.** Real conversation needs the
  listen → transcribe → think → speak loop to happen in well under 2
  seconds or it feels broken. This project uses simple end-of-utterance
  buffering (waits for a pause, then transcribes) rather than fully
  streaming STT, which is easiest to get working but adds delay. Swapping
  in a streaming STT service (Deepgram, AssemblyAI) instead of local
  Whisper will cut latency a lot if it's not fast enough on your hardware.
- **Voicebox's synthesis speed depends on your GPU/CPU and which engine you
  pick.** Lighter engines (Kokoro, LuxTTS) will feel much more real-time
  than heavier voice-cloning ones. You may need to test a couple to find the
  latency/quality tradeoff that works.
- **Voicebox's `/generate` endpoint is asynchronous** — it returns a job id
  you'd need to poll, then fetch the result from `/audio/{generation_id}`,
  which adds a round trip a live call can't afford. `server.py` uses
  `/generate/stream` instead, which returns WAV bytes directly.
- **Audio format is a common trip-up.** Plivo's audio streaming (configured
  via `contentType="audio/x-mulaw;rate=8000"` on `<Stream>` in `server.py`)
  sends/expects 8kHz, 8-bit mu-law audio, same as Twilio. Voicebox outputs
  something else (16kHz+ PCM WAV) — `server.py` resamples and re-encodes
  this, but if you swap components, this conversion is the first place to
  check when audio sounds garbled or is silent.
- **Caller ID:** the call will show your Plivo number, not your personal
  SIM number. Making an automated call display an arbitrary number you
  don't control is caller-ID spoofing and is illegal in most places — don't
  try to work around this.
- **Even "just testing on friends"** — let them know beforehand that
  they're about to get a call from an AI. Using a cloned voice to call
  someone who thinks it's really you, without a heads-up, is the kind of
  thing that damages trust fast even as a joke.
- **If you ever move past friends/testing to real customers:** India's TRAI
  rules apply — DLT registration, 140/160-series sender IDs, DND-registry
  checks, explicit revocable consent, an upfront statement that the call is
  automated, and calls restricted to 10am–7pm. Penalties run ₹1,000 to
  ₹1.5 lakh per violation. This project does none of that compliance work
  for you — it's a personal testing setup, not a production telemarketing
  system.

## Hindi and Kannada support

Voicebox's cloning engines don't currently cover Hindi/Kannada well (Kokoro
does ship a handful of built-in Hindi preset voices, but no Kannada — check
`/profiles/presets/kokoro` for the current list). The open-source option
that clones in these languages — AI4Bharat's IndicF5 — is built on F5-TTS's
flow-matching architecture, which runs around 3x real-time even on a GPU;
on CPU-only hardware that delay is bad enough to break a live conversation.
So for these two languages, this project trades cloning away for speed:

- **English (`--language en`, default):** Voicebox, using whichever voice
  profile you set up (see above).
- **Hindi (`--language hi`) / Kannada (`--language kn`):** Meta's
  MMS-TTS — a tiny (36M-parameter) VITS model per language, exported to
  ONNX and quantized to int8 so it runs fast on CPU alone. This is a
  **stock pretrained voice, not your cloned one** — that's the deliberate
  tradeoff for staying real-time without a GPU.

**One-time setup before your first Hindi/Kannada call:**

```bash
pip install torch==2.4.1 transformers==4.46.3 onnx onnxruntime onnxscript
python quantize_export.py
```

Version pins matter here: newer `torch`/`transformers` combinations (tested:
torch 2.14 + transformers 5.16) fail during ONNX export with confusing
tensor dtype/tracing errors deep in the VITS model code — the versions
pinned above are confirmed working. `transformers` needs to stay on 4.46.3
afterward too, since `server.py`/`indic_tts.py` also import it at runtime;
this doesn't lose you anything, since only basic, stable APIs are used.

This downloads `facebook/mms-tts-hin` and `facebook/mms-tts-kan` from
Hugging Face (a few hundred MB total, one-time), exports each to ONNX, and
quantizes to int8 into `models/hi/` and `models/kn/`. After that, `torch`,
`onnx`, and `onnxscript` are no longer needed and can be uninstalled to save
disk space — `server.py` only ever touches the small quantized files via
`onnxruntime`:

```bash
pip uninstall -y torch onnx onnxscript
```

**License note:** `facebook/mms-tts-hin` and `facebook/mms-tts-kan` are
CC-BY-NC 4.0 — **non-commercial use only**. Fine for personal testing on
friends, but if this project ever turns into something you charge for or
run for customers, swap this piece out for
[AI4Bharat/Indic-TTS](https://github.com/AI4Bharat/Indic-TTS) instead —
same idea (fixed voice, not cloned, but fast), MIT licensed so commercial
use is fine, just a bit more setup work since it doesn't ship a ready HF
`transformers` integration the way MMS-TTS does.

**Usage:**

```bash
python place_call.py --to "+9198XXXXXXX" --context "..." --language hi
python place_call.py --to "+9198XXXXXXX" --context "..." --language kn
```

**Why quantized specifically:** the model is already small (36M params —
tiny by ML standards), but ONNX + int8 quantization is what pushes it from
"probably fine" to "reliably fast" on a plain laptop CPU, the same trick
lightweight CPU-first engines like Piper use by default. It's a one-time
export step, not something that runs per-call.

## Files

- `server.py` — the FastAPI app: PlivoXML answer endpoint + audio Stream
  WebSocket handler + the STT → LLM → TTS loop, language-routed between
  Voicebox (English) and the quantized Hindi/Kannada models.
- `place_call.py` — CLI to kick off an outbound call with your context and
  `--language`.
- `quantize_export.py` — one-time script: downloads, exports, and
  quantizes the Hindi/Kannada models (see above).
- `indic_tts.py` — runtime inference for the quantized Hindi/Kannada
  models, called from `server.py`.
- `requirements.txt` — Python dependencies.
- `.env.example` — copy to `.env` and fill in.
- `models/` — created by `quantize_export.py`; not committed to git (see
  `.gitignore`) since it's fully regenerable from that script.
- `voice_samples/` — anything you generate here (e.g. while testing voices)
  is gitignored; it's scratch output, not part of the project.
