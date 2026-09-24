from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import httpx


class HttpRequestError(RuntimeError):
    pass


class HttpClient:
    def __init__(
        self,
        timeout: float = 30.0,
        user_agent: str = "InferenceTracker/0.1",
        retries: int = 3,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client or httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": user_agent},
        )
        self._owns_client = client is None
        self.retries = max(0, retries)
        self._sleep = sleep

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self._request("GET", url, **kwargs)

    def post_json(self, url: str, payload: dict[str, Any], **kwargs: Any) -> httpx.Response:
        headers = dict(kwargs.pop("headers", {}))
        headers.setdefault("Content-Type", "application/json")
        return self._request("POST", url, json=payload, headers=headers, **kwargs)

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self._client.request(method, url, **kwargs)
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < self.retries:
                        self._sleep(2**attempt)
                        continue
                response.raise_for_status()
                return response
            except (httpx.HTTPError, OSError) as error:
                last_error = error
                if attempt >= self.retries:
                    break
                self._sleep(2**attempt)
        raise HttpRequestError(f"Request failed for {url}: {last_error}") from last_error

    def get_json(self, url: str, **kwargs: Any) -> Any:
        return self.get(url, **kwargs).json()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
