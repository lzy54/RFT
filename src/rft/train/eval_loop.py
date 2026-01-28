# rft/train/eval_loop.py
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Any


def _load_verdicts(verdicts_path: Path) -> list[Dict[str, Any]]:
    with verdicts_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def compute_stats(verdicts: list[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute overall pass rate, per-task pass rate, and failure breakdown.
    """
    total = 0
    passed = 0

    per_task_total = Counter()
    per_task_passed = Counter()
    failure_counter = Counter()

    for rec in verdicts:
        task_id = rec["task_id"]
        verdict = rec.get("verdict", {})

        total += 1
        per_task_total[task_id] += 1

        if verdict.get("passed"):
            passed += 1
            per_task_passed[task_id] += 1
        else:
            failure_counter[verdict.get("failure_tag", "UNKNOWN")] += 1

    overall_pass_rate = passed / total if total > 0 else 0.0

    per_task_rate = {
        task_id: per_task_passed[task_id] / per_task_total[task_id]
        for task_id in per_task_total
    }

    return {
        "overall": {
            "total": total,
            "passed": passed,
            "pass_rate": overall_pass_rate,
        },
        "per_task": per_task_rate,
        "failure_breakdown": dict(failure_counter),
    }


def write_report(stats: Dict[str, Any], out_path: Path) -> None:
    """
    Write a markdown report summarizing evaluation results.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        f.write("# RFT Evaluation Report\n\n")

        f.write("## Overall Performance\n\n")
        f.write(f"- Total samples: {stats['overall']['total']}\n")
        f.write(f"- Passed samples: {stats['overall']['passed']}\n")
        f.write(f"- Pass rate: **{stats['overall']['pass_rate']:.3f}**\n\n")

        f.write("## Per-task Pass Rate\n\n")
        f.write("| Task ID | Pass Rate |\n")
        f.write("|--------|-----------|\n")
        for task_id, rate in sorted(stats["per_task"].items()):
            f.write(f"| {task_id} | {rate:.3f} |\n")

        f.write("\n## Failure Breakdown\n\n")
        f.write("| Failure Tag | Count |\n")
        f.write("|-------------|-------|\n")
        for tag, cnt in stats["failure_breakdown"].items():
            f.write(f"| {tag} | {cnt} |\n")
