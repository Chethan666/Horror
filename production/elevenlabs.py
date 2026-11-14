"""ElevenLabs' own API, for a voice that belongs to the channel.

fal offers ElevenLabs' stock voices; "Brian", the host's, is heard on many faceless channels. A
voice made for the channel (ElevenLabs Voice Design, or a voice actor's licensed clone) lives in
the channel's own ElevenLabs account, so it is read through ElevenLabs' API with that account's
key (GitHub secret ELEVENLABS_API_KEY) and the voice's ID (`stills.host_voice.elevenlabs_id`).
Only the host's lines go this way; everything else stays on fal.

The answer has fal's shape ({"audio": {"url": data URI}}), so the studio caches and pays for it
like any other sound. Stock voices can be named ("Brian"): their IDs are looked up once.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import threading
import urllib.error
import urllib.request
from typing import Any, Callable

from production.media import MediaError

log = logging.getLogger(__name__)

API = "https://api.elevenlabs.io/v1"
PREFIX = "elevenlabs:"
KEY_ENV = "ELEVENLABS_API_KEY"
VOICE_ID = re.compile(r"^[A-Za-z0-9]{18,32}$")


def available() -> bool:
    return bool(os.environ.get(KEY_ENV))


def _http(url: str, *, method: str = "GET", body: Any = None, headers: dict[str, str] | None = None,
          timeout: int = 300) -> bytes:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method,
                                     headers={**(headers or {}), **({"Content-Type": "application/json"} if data else {})})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise MediaError(f"elevenlabs {method} {url.split('?')[0]}: HTTP {exc.code}: "
                         f"{exc.read().decode(errors='replace')[:300]}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise MediaError(f"elevenlabs {method} {url.split('?')[0]} failed: {exc}") from exc


class ElevenLabs:
    """Text to speech on ElevenLabs' own API; `run("elevenlabs:<model id>", {"text", "voice", ...})`."""

    def __init__(self, key: str | None = None, http: Callable[..., bytes] = _http):
        self.key = key or os.environ.get(KEY_ENV, "")
        if not self.key:
            raise MediaError(f"ElevenLabs: set {KEY_ENV}")
        self.http = http
        self.lock = threading.Lock()
        self._ids: dict[str, str] | None = None

    def voice_id(self, voice: str) -> str:
        if VOICE_ID.match(voice):
            return voice
        with self.lock:
            if self._ids is None:
                listing = json.loads(self.http(f"{API}/voices", headers={"xi-api-key": self.key}, timeout=60))
                self._ids = {str(v.get("name", "")).split(" - ")[0].strip().lower(): v.get("voice_id")
                             for v in listing.get("voices", []) if v.get("voice_id")}
        found = self._ids.get(voice.strip().lower())
        if not found:
            raise MediaError(f"ElevenLabs: no voice named {voice!r} in this account")
        return found

    def run(self, endpoint: str, args: dict[str, Any]) -> dict[str, Any]:
        model = endpoint[len(PREFIX):] if endpoint.startswith(PREFIX) else endpoint
        voice = self.voice_id(str(args.get("voice") or ""))
        settings = {k: args[k] for k in ("stability", "similarity_boost", "style") if k in args}
        body: dict[str, Any] = {"text": str(args.get("text", "")), "model_id": model or "eleven_v3"}
        if settings:
            body["voice_settings"] = settings
        audio = self.http(f"{API}/text-to-speech/{voice}?output_format=mp3_44100_128", method="POST", body=body,
                          headers={"xi-api-key": self.key, "Accept": "audio/mpeg"})
        if not audio:
            raise MediaError("ElevenLabs: no audio in the answer")
        return {"audio": {"url": "data:audio/mpeg;base64," + base64.b64encode(audio).decode()}}
