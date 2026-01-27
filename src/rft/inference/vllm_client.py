from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx


@dataclass(frozen=True)
class VLLMConfig:
    """
    OpenAI-compatible vLLM endpoint config.

    Typical vLLM serve command exposes:
      POST {base_url}/v1/chat/completions
    """
    base_url: str
    api_key: str = "EMPTY"
    timeout_sec: float = 120.0
    max_retries: int = 3
    retry_backoff_sec: float = 1.5

    @staticmethod
    def from_env() -> "VLLMConfig":
        base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000").rstrip("/")
        api_key = os.getenv("VLLM_API_KEY", "EMPTY")
        timeout = float(os.getenv("VLLM_TIMEOUT_SEC", "120"))
        max_retries = int(os.getenv("VLLM_MAX_RETRIES", "3"))
        backoff = float(os.getenv("VLLM_RETRY_BACKOFF_SEC", "1.5"))
        return VLLMConfig(
            base_url=base_url,
            api_key=api_key,
            timeout_sec=timeout,
            max_retries=max_retries,
            retry_backoff_sec=backoff,
        )


class VLLMClient:
    """
    Minimal OpenAI-compatible chat client for vLLM.

    Supports:
      - model
      - temperature/top_p/max_tokens
    """

    def __init__(self, config: VLLMConfig):
        self.config = config
        self._http = httpx.Client(
            timeout=httpx.Timeout(config.timeout_sec),
            headers={"Authorization": f"Bearer {config.api_key}"},
        )

    def close(self) -> None:
        try:
            self._http.close()
        except Exception:
            pass

    def chat_completion(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        seed: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Returns raw assistant content (string). Raises on hard failure.
        """
        url = f"{self.config.base_url}/v1/chat/completions"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        if extra:
            payload.update(extra)

        last_err: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                r = self._http.post(url, json=payload)
                r.raise_for_status()
                data = r.json()
                # OpenAI format: choices[0].message.content
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = e
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_backoff_sec * attempt)
                    continue
                break

        assert last_err is not None
        raise last_err
