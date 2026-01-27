from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from rft.inference.vllm_client import VLLMClient


@dataclass
class SamplingConfig:
    model: str
    n: int = 8
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 2048
    seed: Optional[int] = None


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sample_trajectories(
    *,
    client: VLLMClient,
    task_id: str,
    messages: List[Dict[str, str]],
    cfg: SamplingConfig,
    run_id: str,
    out_jsonl: Path,
) -> None:
    """
    For a single task prompt, sample N trajectories and write candidates.jsonl.

    IMPORTANT: This module does NOT parse outputs. It only records raw outputs.
    """
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with out_jsonl.open("a", encoding="utf-8") as f:
        for i in range(cfg.n):
            raw = client.chat_completion(
                model=cfg.model,
                messages=messages,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_tokens=cfg.max_tokens,
                seed=cfg.seed,
            )

            rec: Dict[str, Any] = {
                "run_id": run_id,
                "task_id": task_id,
                "sample_id": i,
                "model": cfg.model,
                "sampling": {
                    "temperature": cfg.temperature,
                    "top_p": cfg.top_p,
                    "max_tokens": cfg.max_tokens,
                    "seed": cfg.seed,
                },
                "input_messages": messages,
                "raw_output_text": raw,
                "created_at": _now_iso(),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
