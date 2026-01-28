from __future__ import annotations

from collections import defaultdict
from typing import Dict, List


def build_dpo_pairs(records: List[Dict]) -> List[Dict]:
    """
    Build DPO preference pairs.

    Assumes:
      - records include both passed and failed samples
      - same prompt can be identified by (task_id, input_messages)
    """
    buckets = defaultdict(list)
    for rec in records:
        key = (rec["task_id"], tuple(m["content"] for m in rec["input_messages"]))
        buckets[key].append(rec)

    pairs = []
    for (_task_id, _), rs in buckets.items():
        passed = [r for r in rs if r.get("verdict", {}).get("passed")]
        failed = [r for r in rs if not r.get("verdict", {}).get("passed")]

        for p in passed:
            for f in failed:
                pairs.append({
                    "messages": p["input_messages"],
                    "chosen": {
                        "role": "assistant",
                        "content": f"<thinking>\n{p.get('thinking','')}\n</thinking>\n```python\n{p['program']}\n```",
                    },
                    "rejected": {
                        "role": "assistant",
                        "content": f"<thinking>\n{f.get('thinking','')}\n</thinking>\n```python\n{f.get('program','')}\n```",
                    },
                    "meta": {
                        "task_id": p["task_id"],
                    }
                })
    return pairs
