from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class ParseResult:
    ok: bool
    status: str  # "OK" or "FORMAT_ERROR"
    thinking: Optional[str] = None
    program: Optional[str] = None
    details: Optional[str] = None


_THINKING_RE = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL | re.IGNORECASE)
# Match fenced blocks: ```python ... ```
_CODEBLOCK_RE = re.compile(
    r"```(?:python|py)\s*(.*?)```", re.DOTALL | re.IGNORECASE
)
# Any fenced block (fallback)
_ANY_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def _strip(s: str) -> str:
    return s.strip("\n\r\t ")


def extract_thinking(raw: str) -> Optional[str]:
    m = _THINKING_RE.search(raw)
    if not m:
        return None
    return _strip(m.group(1))


def extract_python_program(raw: str) -> Optional[str]:
    # Prefer ```python fenced blocks
    blocks = _CODEBLOCK_RE.findall(raw)
    if blocks:
        # Strategy: pick the last python block (often the final answer)
        return _strip(blocks[-1])

    # Fallback: any fenced block
    any_blocks = _ANY_FENCE_RE.findall(raw)
    if any_blocks:
        return _strip(any_blocks[-1])

    return None


def parse_output(raw: str) -> ParseResult:
    """
    Extract thinking + python program from a model output.

    Success criteria:
      - program exists (non-empty)
    """
    thinking = extract_thinking(raw)
    program = extract_python_program(raw)

    if program is None or len(program.strip()) == 0:
        return ParseResult(
            ok=False,
            status="FORMAT_ERROR",
            thinking=thinking,
            program=None,
            details="No fenced python code block found.",
        )

    return ParseResult(
        ok=True,
        status="OK",
        thinking=thinking,
        program=program,
        details=None,
    )
