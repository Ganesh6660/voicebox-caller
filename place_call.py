"""
Kick off an outbound AI-voice phone call.

Usage:
    python place_call.py --to "+9198XXXXXXX" --context "Tell them the delivery moved to Tuesday 5pm."

What this does:
    1. Stores your `--context` string somewhere the running server (server.py)
       can fetch it when Twilio asks for TwiML (we pass it as a URL query
       param, so no shared database is needed for this simple version).
    2. Calls the Twilio REST API to place the call, pointing Twilio at
       PUBLIC_BASE_URL/twiml?context=... — that's the webhook Twilio hits
       once the call connects, which responds with TwiML that opens the
       Media Stream back to server.py's WebSocket.

Requires server.py to already be running and reachable at PUBLIC_BASE_URL
(e.g. an ngrok tunnel) before you run this.
"""

import argparse
import os
import urllib.parse

from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv()

TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
TWILIO_FROM_NUMBER = os.environ["TWILIO_FROM_NUMBER"]
PUBLIC_BASE_URL = os.environ["PUBLIC_BASE_URL"].rstrip("/")


def main():
    parser = argparse.ArgumentParser(description="Place an AI-voice outbound call.")
    parser.add_argument("--to", required=True, help="Phone number to call, E.164 format e.g. +9198XXXXXXX")
    parser.add_argument(
        "--context",
        required=True,
        help="What you want the call to be about — becomes the LLM's system prompt/goal for the conversation.",
    )
    args = parser.parse_args()

    encoded_context = urllib.parse.quote(args.context)
    twiml_url = f"{PUBLIC_BASE_URL}/twiml?context={encoded_context}"

    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    call = client.calls.create(
        to=args.to,
        from_=TWILIO_FROM_NUMBER,
        url=twiml_url,
        # Trial accounts reject an explicit `method` param on Calls.create
        # ("Invalid or disallowed parameters"); omitting it uses Twilio's
        # default of POST, which server.py's /twiml route now also accepts.
    )

    print(f"Call placed. SID={call.sid}  to={args.to}")
    print("Watch server.py's logs for the live conversation loop.")


if __name__ == "__main__":
    main()
