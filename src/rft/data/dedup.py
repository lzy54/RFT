from __future__ import annotations

import hashlib
from typing import Dict


def normalize_program(program: str) -> str:
    """
    Normalize program text before hashing.
    Conservative by design:
    - strip trailing whitespace
    - normalize line endings
    """
    lines = [ln.rstrip() for ln in program.strip().splitlines()]
    return "\n".join(lines)


def program_hash(program: str) -> str:
    """
    Stable hash for a program string.
    """
    norm = normalize_program(program)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def dedup_by_program_hash(records: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Deduplicate records by program hash.

    Input:
      records: key -> record (must contain record["program"])

    Output:
      deduplicated dict (first occurrence kept)
    """
    seen = {}
    for k, rec in records.items():
        prog = rec.get("program")
        if not prog:
            continue
        h = program_hash(prog)
        if h not in seen:
            rec["_program_hash"] = h
            seen[h] = rec
    return seen
