from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


class FilterReason(str):
    THINKING_TOO_LONG = "THINKING_TOO_LONG"
    EMPTY_PROGRAM = "EMPTY_PROGRAM"
    CONTAMINATION = "CONTAMINATION"


def filter_record(
    rec: Dict,
    *,
    max_thinking_chars: int = 8000,
    contamination_checker=None,
) -> Tuple[bool, List[str]]:
    """
    Return (keep, reasons).

    If keep=False, reasons explains why.
    """
    reasons: List[str] = []

    thinking = rec.get("thinking")
    program = rec.get("program")

    if not program or not program.strip():
        reasons.append(FilterReason.EMPTY_PROGRAM)

    if thinking and len(thinking) > max_thinking_chars:
        reasons.append(FilterReason.THINKING_TOO_LONG)

    if contamination_checker is not None:
        if contamination_checker(rec):
            reasons.append(FilterReason.CONTAMINATION)

    return (len(reasons) == 0), reasons


def apply_filters(
    records: Iterable[Dict],
    **kwargs,
) -> List[Dict]:
    kept = []
    for rec in records:
        ok, reasons = filter_record(rec, **kwargs)
        if ok:
            kept.append(rec)
        else:
            rec["_filter_reasons"] = reasons
    return kept
