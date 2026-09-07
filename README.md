# Voicebox Phone Caller — starter scaffold

This is a starting point for what you described: you type/paste some context
("what to talk about"), and the system places a real phone call and carries
a live, two-way conversation using your cloned voice — without you speaking.

**Read this whole file before running anything.** This is a working skeleton,
not a polished product. The pieces that matter most (audio format, latency)
are called out below.

## Why Voicebox alone can't do this

Voicebox (the `jamiepine/voicebox` app) generates speech and clones voices,
but it has no telephony — it can't dial a number or carry audio over a phone
network. This scaffold adds the missing piece: a telephony provider
(Plivo) that places the call and streams the live audio to a small server
you run, which stitches together listening → thinking → speaking.

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
                                          Speech-to-text (Whisper)
                                                    │
                              transcript + your context ──▶ LLM (Claude)
                                                    │
                                            LLM's reply text
                                                    │
                                  Voicebox /generate  (your cloned voice)
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
   see the limitations below.
2. **Voicebox running locally** with its API reachable (default
   `http://127.0.0.1:17493`), with a cloned voice profile already created for
   the voice you want to speak with.
3. **An Anthropic API key** (or swap in whatever LLM you prefer — see
   `generate_reply()` in `server.py`).
4. **A public URL for `server.py`.** Plivo needs to reach your machine over
   the internet. Easiest for testing: `ngrok http 8000` and use the ngrok
   URL. For anything beyond testing, deploy `server.py` on a real host.
5. **ffmpeg** installed (used for audio resampling/encoding).

## Setup

```bash
cd voicebox-caller
python3 -m venv venv && source venv/bin/activate
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

## The honest limitations

- **Latency is the whole game.** Real conversation needs the
  listen → transcribe → think → speak loop to happen in well under 2
  seconds or it feels broken. This scaffold uses simple end-of-utterance
  buffering (waits for a pause, then transcribes) rather than fully
  streaming STT, which is easiest to get working but adds delay. Swapping
  in a streaming STT service (Deepgram, AssemblyAI) instead of local
  Whisper will cut latency a lot if it's not fast enough on your hardware.
- **Voicebox's synthesis speed depends on your GPU/CPU and which of its
  seven engines you pick.** Lighter engines (Kokoro) will feel much more
  real-time than heavier voice-cloning ones. You may need to test a couple
  to find the latency/quality tradeoff that works.
- **Audio format is a common trip-up.** Plivo's audio streaming (configured
  via `contentType="audio/x-mulaw;rate=8000"` on `<Stream>` in `server.py`)
  sends/expects 8kHz, 8-bit mu-law audio, same as Twilio. Voicebox will
  output something else (likely 16kHz+ PCM WAV) — `server.py` resamples and
  re-encodes this, but if you swap components, this conversion is the first
  place to check when audio sounds garbled or is silent.
- **Caller ID:** the call will show your Plivo number, not your personal
  SIM number. Making an automated call display an arbitrary number you
  don't control is caller-ID spoofing and is illegal in most places — don't
  try to work around this.
- **Even "just testing on friends"** — let them know beforehand that
  they're about to get a call from an AI using your cloned voice. Cloning
  your own voice to call someone who thinks it's really you, without a
  heads-up, is the kind of thing that damages trust fast even as a joke.
- **If you ever move past friends/testing to real customers:** India's TRAI
  rules apply — DLT registration, 140/160-series sender IDs, DND-registry
  checks, explicit revocable consent, an upfront statement that the call is
  automated, and calls restricted to 10am–7pm. Penalties run ₹1,000 to
  ₹1.5 lakh per violation. This scaffold does none of that compliance work
  for you — it's a personal testing setup, not a production telemarketing
  system.

## Hindi and Kannada support

Voicebox's cloning engines don't currently cover Hindi/Kannada well. The
open-source option that does clone in these languages — AI4Bharat's
IndicF5 — is built on F5-TTS's flow-matching architecture, which runs
around 3x real-time even on a GPU; on CPU-only hardware (no GPU here) that
delay is bad enough to break a live conversation. So for these two
languages specifically, this project trades cloning away for speed:

- **English (`--language en`, default):** Voicebox, your cloned voice.
- **Hindi (`--language hi`) / Kannada (`--language kn`):** Meta's
  MMS-TTS — a tiny (36M-parameter) VITS model per language, exported to
  ONNX and quantized to int8 so it runs fast on CPU alone. This is a
  **stock pretrained voice, not your cloned one** — that's the deliberate
  tradeoff for staying real-time without a GPU.

**One-time setup before your first Hindi/Kannada call:**

```bash
pip install torch transformers onnx onnxruntime
python quantize_export.py
```

This downloads `facebook/mms-tts-hin` and `facebook/mms-tts-kan` from
Hugging Face (a few hundred MB, one-time), exports each to ONNX, and
quantizes to int8 into `models/hi/` and `models/kn/`. After that, `torch`
and `onnx` are no longer needed — `server.py` only ever touches the small
quantized files via `onnxruntime`.

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
