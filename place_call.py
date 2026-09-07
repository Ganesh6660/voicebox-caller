"""
Kick off an outbound AI-voice phone call.

Usage:
    python place_call.py --to "+9198XXXXXXX" --context "Tell them the delivery moved to Tuesday 5pm."
    python place_call.py --to "+9198XXXXXXX" --context "..." --language hi
    python place_call.py --to "+9198XXXXXXX" --context "..." --language kn

--language defaults to "en" (Voicebox, your cloned voice). "hi"/"kn" use the
local quantized MMS-TTS models instead (a stock voice, not cloned) — run
quantize_export.py once before using those. See README.md for why.

What this does:
    1. Stores your `--context` string somewhere the running server (server.py)
       can fetch it when Plivo asks for PlivoXML (we pass it as a URL query
       param, so no shared database is needed for this simple version).
    2. Calls the Plivo REST API to place the call, pointing Plivo at
       PUBLIC_BASE_URL/answer?context=... — that's the webhook Plivo hits
       once the call connects, which responds with PlivoXML that opens the
       bidirectional audio Stream back to server.py's WebSocket.

Requires server.py to already be running and reachable at PUBLIC_BASE_URL
(e.g. an ngrok tunnel) before you run this.
"""

import argparse
import os
import urllib.parse

import plivo
from dotenv import load_dotenv

load_dotenv()

PLIVO_AUTH_ID = os.environ["PLIVO_AUTH_ID"]
PLIVO_AUTH_TOKEN = os.environ["PLIVO_AUTH_TOKEN"]
PLIVO_FROM_NUMBER = os.environ["PLIVO_FROM_NUMBER"]
PUBLIC_BASE_URL = os.environ["PUBLIC_BASE_URL"].rstrip("/")


def main():
    parser = argparse.ArgumentParser(description="Place an AI-voice outbound call.")
    parser.add_argument("--to", required=True, help="Phone number to call, E.164 format e.g. +9198XXXXXXX")
    parser.add_argument(
        "--context",
        required=True,
        help="What you want the call to be about — becomes the LLM's system prompt/goal for the conversation.",
    )
    parser.add_argument(
        "--language",
        default="en",
        choices=["en", "hi", "kn"],
        help="Call language. 'en' = Voicebox (your cloned voice). 'hi'/'kn' = local quantized "
        "MMS-TTS (a stock voice, not cloned) — run quantize_export.py first. Default: en.",
    )
    args = parser.parse_args()

    encoded_context = urllib.parse.quote(args.context)
    answer_url = f"{PUBLIC_BASE_URL}/answer?context={encoded_context}&language={args.language}"

    client = plivo.RestClient(PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN)
    call = client.calls.create(
        from_=PLIVO_FROM_NUMBER,
        to_=args.to,
        answer_url=answer_url,
        answer_method="POST",
    )

    print(f"Call placed. UUID={call.request_uuid}  to={args.to}")
    print("Watch server.py's logs for the live conversation loop.")


if __name__ == "__main__":
    main()
