"""Google Cloud Vertex AI: Nano Banana 2.1 pictures and Gemini checks, paid from Google Cloud.

Plain HTTPS calls to Vertex AI's generateContent, signed in with a Google Cloud API key
(GitHub secret GEMINI_PLATFORM_API_KEY) or a service account key (GCP_SERVICE_ACCOUNT_JSON: the
key file's JSON), kept only in the environment. The location is `global` unless GCP_LOCATION
says otherwise; GCP_PROJECT names the project when the key alone does not.

Models are named in the config as "vertex:<model id>" (e.g. "vertex:gemini-nano-banana-2.1").
`use_vertex()` moves the configured roles from fal to Vertex when a key is present; the
`Providers` router sends "vertex:" models to Google and every other model to fal, and if a
Vertex call fails for any reason but a refused picture, the same request goes to the fal
model it replaced. Nothing changes for a run without the key.

Requests and answers follow Vertex AI's generateContent:
  request  {"contents": [{"role": "user", "parts": [{"inlineData": {...}}, ..., {"text": prompt}]}],
            "generationConfig": {"responseModalities": ["IMAGE"],
                                 "imageConfig": {"aspectRatio": "16:9", "imageSize": "2K"}}}
  answer   candidates[0].content.parts[].inlineData {"mimeType", "data" (base64)}
A setting the model rejects (HTTP 400 naming imageSize, imageConfig or responseModalities)
is dropped and the request sent again, so a model that takes fewer settings still answers.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import threading
import time
import urllib.parse
from typing import Any, Callable

from production.media import MediaError, _request

log = logging.getLogger(__name__)

TOKEN_URL = "https://oauth2.googleapis.com/token"
SCOPE = "https://www.googleapis.com/auth/cloud-platform"
PREFIX = "vertex:"
# Horror is the channel's genre: block only what is clearly harmful, so a dark scene is not refused.
SAFETY = [{"category": c, "threshold": "BLOCK_ONLY_HIGH"} for c in (
    "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT")]


KEY_ENV = "GEMINI_PLATFORM_API_KEY"  # a Google Cloud API key for Vertex AI (not an AI Studio key)


def available() -> bool:
    return bool(os.environ.get(KEY_ENV) or os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
                or os.environ.get("GCP_ACCESS_TOKEN"))


def client() -> "Vertex":
    """Vertex AI signed in with the Google Cloud API key (GEMINI_PLATFORM_API_KEY) when it is set,
    else with a service account key: the same models, requests and answers, billed to the Google
    Cloud project the key belongs to."""
    if os.environ.get(KEY_ENV):
        return VertexKey()
    return Vertex()


def _data(uri: str) -> tuple[str, str]:
    """(mime type, base64 data) of a data: URI."""
    if not uri.startswith("data:") or ";base64," not in uri:
        raise MediaError("vertex: reference pictures must be sent as data URIs")
    head, data = uri[5:].split(";base64,", 1)
    return head or "image/png", data


class Vertex:
    """generateContent on Vertex AI, with a service account's access token."""

    def __init__(self, credentials: dict[str, Any] | None = None, project: str | None = None,
                 location: str | None = None, http: Callable[..., Any] | None = None,
                 token: str | None = None):
        raw = os.environ.get("GCP_SERVICE_ACCOUNT_JSON", "")
        self.info = credentials if credentials is not None else (json.loads(raw) if raw.strip() else {})
        self.project = project or os.environ.get("GCP_PROJECT") or self.info.get("project_id", "")
        self.location = location or os.environ.get("GCP_LOCATION") or "global"
        self.http = http or _request
        self._token = token or os.environ.get("GCP_ACCESS_TOKEN") or ""
        self._expiry = float("inf") if self._token else 0.0
        self.lock = threading.Lock()
        if not self.project:
            raise MediaError("Vertex AI: no project (set GCP_PROJECT, or use a key file that names one)")
        if not self._token and not self.info.get("private_key"):
            raise MediaError("Vertex AI: set GCP_SERVICE_ACCOUNT_JSON (a service account key) as a secret")

    @property
    def host(self) -> str:
        return "aiplatform.googleapis.com" if self.location == "global" else f"{self.location}-aiplatform.googleapis.com"

    def token(self) -> str:
        """An access token from the service account key (a signed JWT exchanged at Google), kept
        until five minutes before it expires."""
        with self.lock:
            if self._token and time.time() < self._expiry - 300:
                return self._token
            from google.auth import crypt, jwt  # google-auth: only needed when a key signs in

            now = int(time.time())
            signer = crypt.RSASigner.from_service_account_info(self.info)
            assertion = jwt.encode(signer, {"iss": self.info["client_email"], "scope": SCOPE, "aud": TOKEN_URL,
                                            "iat": now, "exp": now + 3600})
            body = urllib.parse.urlencode({"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                                           "assertion": assertion.decode() if isinstance(assertion, bytes)
                                           else assertion}).encode()
            import urllib.request
            request = urllib.request.Request(TOKEN_URL, data=body, method="POST",
                                             headers={"Content-Type": "application/x-www-form-urlencoded"})
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    got = json.loads(response.read())
            except Exception as exc:  # noqa: BLE001 - any sign-in failure is reported the same way
                raise MediaError(f"Vertex AI sign-in failed: {str(exc)[:300]}") from exc
            self._token = got["access_token"]
            self._expiry = now + float(got.get("expires_in", 3600))
            return self._token

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token()}", "Content-Type": "application/json"}

    def url(self, model: str, method: str = "generateContent", location: str | None = None) -> str:
        where = location or self.location
        host = "aiplatform.googleapis.com" if where == "global" else f"{where}-aiplatform.googleapis.com"
        return (f"https://{host}/v1/projects/{self.project}/locations/{where}"
                f"/publishers/google/models/{model}:{method}")

    WAITS = (10.0, 30.0, 60.0)  # a busy model or a per-minute quota: wait and ask again before falling back

    def _post(self, model: str, body: dict[str, Any], method: str = "generateContent",
              location: str | None = None) -> dict[str, Any]:
        for k in range(len(self.WAITS) + 1):
            try:
                return self.http(self.url(model, method, location), method="POST", body=body, timeout=300,
                                 headers=self._headers())
            except MediaError as exc:
                text = str(exc)
                busy = any(w in text for w in ("HTTP 429", "HTTP 503", "RESOURCE_EXHAUSTED", "UNAVAILABLE"))
                if not busy or k == len(self.WAITS):
                    raise
                log.info("Vertex %s busy (%s); asking again in %.0f s", model, text[:120], self.WAITS[k])
                time.sleep(self.WAITS[k])
        raise MediaError(f"vertex {model}: no answer")

    @staticmethod
    def _parts(args: dict[str, Any]) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        for uri in list(args.get("image_urls") or []) + [args[k] for k in ("video_url", "audio_url") if args.get(k)]:
            mime, data = _data(str(uri))  # pictures, and a video or a sound to watch or hear (the screening)
            parts.append({"inlineData": {"mimeType": mime, "data": data}})
        parts.append({"text": str(args.get("prompt", ""))})
        return parts

    def run(self, endpoint: str, args: dict[str, Any]) -> dict[str, Any]:
        """The same answer shape as fal: {"images": [{"url": data URI}]} for a picture model,
        {"output": text} for a text model."""
        model = endpoint[len(PREFIX):] if endpoint.startswith(PREFIX) else endpoint
        if "lyria" in model:
            return self.music(model, args)
        if "veo" in model:
            return self.video(model, args)
        if "image" in model or "banana" in model:
            return self.image(model, args)
        return self.text(model, args)

    # Veo runs as a long job: asked, then fetched until it is done.
    VIDEO_LOCATIONS = ("us-central1", "global")
    POLL_SECONDS, VIDEO_TIMEOUT = 10.0, 900.0

    def video(self, model: str, args: dict[str, Any]) -> dict[str, Any]:
        """A short clip from Veo, started from a picture (image_url / start_image_url: a data URI), with
        no sound of its own (the edit has the narration and the score). Answered as fal answers:
        {"video": {"url": data URI}}."""
        start = args.get("image_url") or args.get("start_image_url") or args.get("image")
        instance: dict[str, Any] = {"prompt": str(args.get("prompt", ""))[:2000]}
        if start:
            mime, data = _data(str(start))
            instance["image"] = {"bytesBase64Encoded": data, "mimeType": mime}
        if args.get("last_frame_url"):  # the clip ends on this picture (the same as the first: a seamless loop)
            mime, data = _data(str(args["last_frame_url"]))
            instance["lastFrame"] = {"bytesBase64Encoded": data, "mimeType": mime}
        try:
            seconds = int(float(str(args.get("duration", args.get("durationSeconds", 8))).rstrip("s")))
        except ValueError:
            seconds = 8
        params = {"aspectRatio": str(args.get("aspect_ratio", "16:9")), "durationSeconds": seconds,
                  "sampleCount": 1, "generateAudio": bool(args.get("generate_audio", False)),
                  "resolution": str(args.get("resolution", "1080p")), "personGeneration": "allow_adult"}
        body = {"instances": [instance], "parameters": params}
        last: MediaError | None = None
        for where in ([self._video_at] if getattr(self, "_video_at", "") else list(self.VIDEO_LOCATIONS)):
            try:
                job = self._post(model, body, "predictLongRunning", where)
            except MediaError as exc:
                if any(w in str(exc) for w in ("HTTP 404", "not found", "NOT_FOUND", "not supported in")):
                    last = exc
                    continue
                raise
            self._video_at = where
            name = job.get("name", "")
            if not name:
                raise MediaError(f"vertex {model}: no job ({str(job)[:200]})")
            waited = 0.0
            while waited < self.VIDEO_TIMEOUT:
                done = self._post(model, {"operationName": name}, "fetchPredictOperation", where)
                if done.get("done"):
                    if done.get("error"):
                        raise MediaError(f"vertex {model}: {str(done['error'])[:300]}")
                    reply = done.get("response") or {}
                    for item in reply.get("videos") or []:
                        data = item.get("bytesBase64Encoded")
                        if data:
                            return {"video": {"url": f"data:{item.get('mimeType') or 'video/mp4'};base64,{data}"}}
                    raise MediaError(f"vertex {model}: no clip, content refused or filtered ({str(reply)[:200]})")
                time.sleep(self.POLL_SECONDS)
                waited += self.POLL_SECONDS
            raise MediaError(f"vertex {model}: the clip took longer than {self.VIDEO_TIMEOUT:.0f} s")
        raise last or MediaError(f"vertex {model}: no location serves it")

    # Lyria is served in fewer places than Gemini: the first location that knows the model is kept.
    MUSIC_LOCATIONS = ("global", "us-central1")

    def music(self, model: str, args: dict[str, Any]) -> dict[str, Any]:
        """A piece of instrumental music from Lyria (about 30 seconds, 48 kHz WAV), answered as fal
        answers: {"audio": {"url": data URI}}."""
        instance: dict[str, Any] = {"prompt": str(args.get("prompt", ""))[:1500]}
        if args.get("negative_prompt"):
            instance["negative_prompt"] = str(args["negative_prompt"])[:500]
        if args.get("seed") is not None:
            instance["seed"] = int(args["seed"])
        body = {"instances": [instance], "parameters": {"sample_count": 1}}
        last: MediaError | None = None
        for where in ([self._music_at] if getattr(self, "_music_at", "") else list(self.MUSIC_LOCATIONS)):
            try:
                reply = self._post(model, body, "predict", where)
            except MediaError as exc:
                text = str(exc)
                if any(w in text for w in ("HTTP 404", "not found", "NOT_FOUND", "not supported in")):
                    last = exc
                    continue
                raise
            self._music_at = where
            for item in reply.get("predictions") or []:
                data = item.get("bytesBase64Encoded") or item.get("audioContent")
                if data:
                    mime = item.get("mimeType") or "audio/wav"
                    return {"audio": {"url": f"data:{mime};base64,{data}"}}
            raise MediaError(f"vertex {model}: no music, content refused or empty ({str(reply)[:200]})")
        raise last or MediaError(f"vertex {model}: no location serves it")

    def image(self, model: str, args: dict[str, Any]) -> dict[str, Any]:
        image_config: dict[str, Any] = {}
        if args.get("aspect_ratio"):
            image_config["aspectRatio"] = str(args["aspect_ratio"])
        if args.get("resolution"):
            image_config["imageSize"] = str(args["resolution"])
        body: dict[str, Any] = {"contents": [{"role": "user", "parts": self._parts(args)}],
                                "generationConfig": {"responseModalities": ["IMAGE"]},
                                "safetySettings": SAFETY}
        if image_config:
            body["generationConfig"]["imageConfig"] = image_config
        if args.get("system_prompt"):
            body["systemInstruction"] = {"parts": [{"text": str(args["system_prompt"])}]}
        tries = [body]
        if "imageSize" in image_config:  # a model that takes no size still answers at its own
            smaller = copy.deepcopy(body)
            smaller["generationConfig"]["imageConfig"].pop("imageSize")
            tries.append(smaller)
        both = copy.deepcopy(tries[-1])
        both["generationConfig"]["responseModalities"] = ["TEXT", "IMAGE"]
        tries.append(both)
        last: MediaError | None = None
        for k, attempt in enumerate(tries):
            try:
                reply = self._post(model, attempt)
            except MediaError as exc:
                text = str(exc)
                if "HTTP 400" in text and any(w in text for w in ("imageSize", "image_size", "imageConfig",
                                                                 "responseModalities", "response_modalities")):
                    last = exc
                    log.info("Vertex %s rejected a setting, asking again without it: %s", model, text[:200])
                    continue
                raise
            for candidate in reply.get("candidates") or []:
                for part in (candidate.get("content") or {}).get("parts") or []:
                    inline = part.get("inlineData") or part.get("inline_data")
                    if inline and inline.get("data"):
                        mime = inline.get("mimeType") or inline.get("mime_type") or "image/png"
                        return {"images": [{"url": f"data:{mime};base64,{inline['data']}"}],
                                "usage": reply.get("usageMetadata", {})}
            finish = ", ".join(str(c.get("finishReason", "")) for c in reply.get("candidates") or [])
            blocked = (reply.get("promptFeedback") or {}).get("blockReason", "")
            # "content" in the message: the studio does not ask again for a refused picture
            raise MediaError(f"vertex {model}: no picture, content refused or empty "
                             f"(finish: {finish or 'none'}{', blocked: ' + blocked if blocked else ''})")
        raise last or MediaError(f"vertex {model}: no picture")

    # Checker models, newest first: a model Google has retired answers "not found" (run 110:
    # gemini-2.5-flash), and the next is tried; the first that answers is kept for the run. A Pro
    # model (the stronger checker, the screening) tries the other Pro models before any Flash one.
    TEXT_MODELS = ("gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.1-flash", "gemini-3-flash",
                   "gemini-flash-latest", "gemini-2.5-flash")
    PRO_MODELS = ("gemini-3.6-pro", "gemini-3.5-pro", "gemini-3.1-pro", "gemini-3-pro", "gemini-3-pro-preview",
                  "gemini-pro-latest", "gemini-2.5-pro")

    def text(self, model: str, args: dict[str, Any]) -> dict[str, Any]:
        family = "pro" if "pro" in model else "flash"
        found = getattr(self, "_found", None)
        if found is None:
            found = self._found = {}
        tried = found.get(family) or model
        ladder = (self.PRO_MODELS + self.TEXT_MODELS) if family == "pro" else self.TEXT_MODELS
        order = [tried] + [m for m in ladder if m != tried]
        last: MediaError | None = None
        for name in order:
            try:
                got = self._text(name, args)
                found[family] = name
                if family == "flash":
                    self._text_model = name  # what the model check reports
                return got
            except MediaError as exc:
                if "HTTP 404" not in str(exc):
                    raise
                last = exc
        raise last or MediaError(f"vertex {model}: no checker model answered")

    def _text(self, model: str, args: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {"contents": [{"role": "user", "parts": self._parts(args)}],
                                "generationConfig": {"maxOutputTokens": int(args.get("max_tokens") or 2048),
                                                     "temperature": float(args.get("temperature") or 0.2)},
                                "safetySettings": SAFETY}
        if args.get("system_prompt"):
            body["systemInstruction"] = {"parts": [{"text": str(args["system_prompt"])}]}
        reply = self._post(model, body)
        texts = [p.get("text", "") for c in reply.get("candidates") or []
                 for p in (c.get("content") or {}).get("parts") or [] if p.get("text")]
        if not texts:
            raise MediaError(f"vertex {model}: no answer ({str(reply)[:200]})")
        return {"output": "".join(texts)}

    def model_info(self, model: str) -> dict[str, Any]:
        """The model's record in Vertex's catalogue (free): proves the key, the API and the model."""
        return self.http(f"https://{self.host}/v1/publishers/google/models/{model}", timeout=60,
                         headers={"Authorization": f"Bearer {self.token()}"})


class VertexKey(Vertex):
    """Vertex AI with a Google Cloud API key (sent as the x-goog-api-key header, never in the URL):
    no service account. Without GCP_PROJECT the project-less address is used, which Vertex AI
    resolves from the key; with it, the project's own address."""

    def __init__(self, key: str | None = None, project: str | None = None, location: str | None = None,
                 http: Callable[..., Any] | None = None):
        self.key = key or os.environ.get(KEY_ENV, "")
        if not self.key:
            raise MediaError(f"Vertex AI: set {KEY_ENV} (a Google Cloud API key) as a secret")
        self.project = project or os.environ.get("GCP_PROJECT", "")
        self.location = location or os.environ.get("GCP_LOCATION") or "global"
        self.http = http or _request
        self.lock = threading.Lock()

    def token(self) -> str:
        return ""

    def _headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self.key, "Content-Type": "application/json"}

    def discover(self) -> str:
        """The key's project number, learnt for free: the project-less address answers a request
        for a model that does not exist with "projects/<number>/..." in its error. Kept out of the
        repository; the global location is then used (the project-less address means us-central1)."""
        with self.lock:
            if self.project or getattr(self, "_discovered", False):
                return self.project
            self._discovered = True  # asked once
            try:
                self.http(f"https://{self.host}/v1/publishers/google/models/no-such-model:generateContent",
                          method="POST", body={"contents": [{"role": "user", "parts": [{"text": "x"}]}]},
                          timeout=60, headers=self._headers())
            except MediaError as exc:
                found = re.search(r"projects/([\w-]+)/locations", str(exc))
                if found:
                    self.project = found.group(1)
            return self.project

    def url(self, model: str, method: str = "generateContent", location: str | None = None) -> str:
        if self.project or self.discover():
            return super().url(model, method, location)
        return f"https://{self.host}/v1/publishers/google/models/{model}:{method}"

    def model_info(self, model: str) -> dict[str, Any]:
        raise MediaError("the model catalogue needs a service account (API keys cannot read it)")


class Providers:
    """fal and Vertex behind one `run()`: "vertex:" models go to Google, every other to fal.
    A Vertex failure (not a refused picture) is sent to the fal model it replaced."""

    def __init__(self, fal: Any = None, vertex: Vertex | None = None,
                 fallback: dict[str, list[str]] | None = None, others: dict[str, Any] | None = None):
        self.fal, self.vertex, self.fallback = fal, vertex, dict(fallback or {})
        self.others = dict(others or {})  # other services by prefix, e.g. {"elevenlabs:": ElevenLabs()}
        self.fell_back = 0
        self.fal_empty = False  # set once fal answers that its balance is empty: it cannot be the fallback
        self.waited_for_quota = 0
        self.sleep = time.sleep

    # A busy Vertex model (a per-minute quota) that fal cannot stand in for: wait for the quota to refill
    # and ask Vertex again, rather than lose the picture (run 114: nine beats got HTTP 429 from Nano
    # Banana, fal's balance was empty, and two beats were left showing a neighbour's picture).
    QUOTA_WAITS = (90.0, 180.0)

    @staticmethod
    def _busy(exc: Exception) -> bool:
        return any(w in str(exc) for w in ("HTTP 429", "HTTP 503", "RESOURCE_EXHAUSTED", "UNAVAILABLE"))

    @staticmethod
    def _fal_empty(exc: Exception) -> bool:
        text = str(exc)
        return any(w in text for w in ("Exhausted balance", "balance is empty", "User is locked"))

    def _wait_for_vertex(self, endpoint: str, args: dict[str, Any], first: MediaError) -> dict[str, Any]:
        last = first
        for wait in self.QUOTA_WAITS:
            log.info("Vertex %s still busy and fal cannot stand in; asking again in %.0f s", endpoint, wait)
            self.sleep(wait)
            self.waited_for_quota += 1
            try:
                return self.vertex.run(endpoint, args)
            except MediaError as exc:
                if not self._busy(exc):
                    raise
                last = exc
        raise last

    def run(self, endpoint: str, args: dict[str, Any]) -> dict[str, Any]:
        for prefix, other in self.others.items():
            if endpoint.startswith(prefix):
                return other.run(endpoint, args)
        if not endpoint.startswith(PREFIX):
            if self.fal is None:
                raise MediaError(f"{endpoint}: no FAL_KEY")
            try:
                return self.fal.run(endpoint, args)
            except MediaError as exc:
                self.fal_empty = self.fal_empty or self._fal_empty(exc)
                raise
        try:
            if self.vertex is None:
                raise MediaError("Vertex AI: no key")
            return self.vertex.run(endpoint, args)
        except MediaError as exc:
            spares = self.fallback.get(endpoint) or []
            # with reference pictures, the fal model that takes them (".../edit"); without, the other
            refs = bool(args.get("image_urls"))
            spare = next((e for e in spares if e.endswith("/edit") == refs), spares[0] if spares else "")
            busy = self.vertex is not None and self._busy(exc)
            if self.fal is None or not spare or "content refused" in str(exc):
                if busy:
                    return self._wait_for_vertex(endpoint, args, exc)
                raise
            if busy and self.fal_empty:
                return self._wait_for_vertex(endpoint, args, exc)
            self.fell_back += 1
            log.warning("Vertex %s failed (%s); sent to %s on fal", endpoint, " ".join(str(exc).split())[:400], spare)
            try:
                return self.fal.run(spare, args)
            except MediaError as fal_exc:
                if not self._fal_empty(fal_exc):
                    raise
                self.fal_empty = True
                if busy:
                    return self._wait_for_vertex(endpoint, args, exc)
                raise

    def __getattr__(self, name: str) -> Any:  # schema(), price() and the rest stay fal's
        if self.fal is None:
            raise AttributeError(name)
        return getattr(self.fal, name)


def use_vertex(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """The stills config with its Nano Banana 2.1 roles (and the checker) moved to Vertex, and
    the fal model each Vertex model replaced. Unchanged when `vertex.enabled` is off."""
    settings = cfg.get("vertex") or {}
    if not settings.get("enabled", True):
        return cfg, {}
    cfg = copy.deepcopy(cfg)
    models: dict[str, str] = cfg.get("models", {})
    fallback: dict[str, list[str]] = {}
    moves = {str(k): str(v) for k, v in (settings.get("models") or {}).items()}  # fal endpoint -> vertex model
    roles = {str(k): str(v) for k, v in (settings.get("roles") or {}).items()}  # a role's own model (vision_pro)
    for role, endpoint in list(models.items()):
        if not endpoint or endpoint.startswith(PREFIX):
            continue
        target = roles.get(role) or moves.get(endpoint)
        if target:
            models[role] = PREFIX + target
            if endpoint not in fallback.setdefault(PREFIX + target, []):
                fallback[PREFIX + target].append(endpoint)
    prices = cfg.setdefault("prices", {})
    for target, usd in (settings.get("prices") or {}).items():
        prices[PREFIX + str(target)] = float(usd)
    return cfg, fallback
