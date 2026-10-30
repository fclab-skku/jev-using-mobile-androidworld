"""API providers. A provider is a registered endpoint + key + model that agents use through slots.

kinds:
  systemone  POST {base_url}/systemone  (TypeSafe Jev, or OpenRouter's drop-in route)
  chat       POST {base_url}/chat/completions  (any OpenAI-compatible endpoint)
"""

import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

import httpx

RETRY_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class ProviderError(RuntimeError):
    pass


class Provider:
    kind = ""
    path = ""

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key_env: str,
        model: str,
        timeout: float = 30.0,
        max_retries: int = 4,
        headers: Optional[Dict[str, str]] = None,
        body_defaults: Optional[Dict[str, Any]] = None,
        http2: bool = False,
        total_timeout: Optional[float] = None,
    ):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.headers = headers or {}
        self.body_defaults = body_defaults or {}
        # HTTP/1.1 by default: one httpx HTTP/2 connection shared by worker threads intermittently
        # raised KeyError(<stream id>) inside h2 during parallel runs.
        self.http2 = http2
        # Wall-clock limit for one attempt. `timeout` alone is httpx's per-operation limit (gap between bytes), which
        # never fires while a gateway keeps the connection alive with whitespace: OpenRouter then answers HTTP 200 with
        # {"error": "Upstream idle timeout exceeded"} after ~120 s. Exceeding the limit counts as a transport failure
        # and the attempt is retried.
        self.total_timeout = total_timeout
        self._client: Optional[httpx.Client] = None
        self._lock = threading.Lock()

    @property
    def url(self) -> str:
        return self.base_url + self.path

    @property
    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise ProviderError("Provider %s needs %s (set it in .env)." % (self.name, self.api_key_env))
        return key

    def client(self) -> httpx.Client:
        with self._lock:
            if self._client is None:
                self._client = httpx.Client(
                    http2=self.http2,
                    timeout=self.timeout,
                    trust_env=False,
                    limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
                )
            return self._client

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "url": self.url, "model": self.model, "body_defaults": self.body_defaults}

    def _send(self, payload: Dict[str, Any], headers: Dict[str, str]):
        if not self.total_timeout:
            return self.client().post(self.url, json=payload, headers=headers)
        deadline = time.monotonic() + self.total_timeout
        limits = httpx.Timeout(self.timeout, read=min(self.timeout, self.total_timeout))
        with self.client().stream("POST", self.url, json=payload, headers=headers, timeout=limits) as stream:
            chunks = []
            for chunk in stream.iter_bytes():
                chunks.append(chunk)
                if time.monotonic() > deadline:
                    raise TimeoutError("no complete response within %.0f s (total_timeout)" % self.total_timeout)
            return _Buffered(stream.status_code, stream.headers, b"".join(chunks))

    def post(self, body: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """POST with the provider's model/defaults applied. Returns (json, meta)."""
        payload = {**self.body_defaults, **body, "model": self.model}
        headers = {"Authorization": "Bearer %s" % self.api_key, **self.headers}
        started = time.perf_counter()
        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                response = self._send(payload, headers)
            except Exception as error:  # transport failures (httpx or lower layers) are retried
                last_error = "%s: %s" % (type(error).__name__, error)
            else:
                if response.status_code < 400:
                    meta = {
                        "provider": self.name,
                        "status": response.status_code,
                        "attempts": attempt + 1,
                        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                    }
                    return response.json(), meta
                last_error = "HTTP %d: %s" % (response.status_code, response.text[:500])
                if response.status_code not in RETRY_STATUSES:
                    break
                retry_after = response.headers.get("retry-after")
                if retry_after and attempt < self.max_retries:
                    try:
                        time.sleep(min(float(retry_after), 30))
                        continue
                    except ValueError:
                        pass
            if attempt < self.max_retries:
                time.sleep(min(0.5 * 2**attempt, 8))
        raise ProviderError("%s request failed: %s" % (self.name, last_error))


class _Buffered:
    """The parts of httpx.Response that post() reads, for a body collected from a stream."""

    def __init__(self, status_code: int, headers, content: bytes):
        self.status_code, self.headers, self.content = status_code, headers, content

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", "replace")

    def json(self):
        import json

        return json.loads(self.content)


class SystemOneProvider(Provider):
    kind = "systemone"
    path = "/systemone"


class ChatProvider(Provider):
    kind = "chat"
    path = "/chat/completions"


PROVIDER_KINDS = {cls.kind: cls for cls in (SystemOneProvider, ChatProvider)}
