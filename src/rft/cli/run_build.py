from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import typer

from rft.data.dedup import dedup_by_program_hash
from rft.data.filters import apply_filters
from rft.data.sft_builder import build_sft_dataset
from rft.data.dpo_builder import build_dpo_pairs

app = typer.Typer(add_completion=False)


@app.command()
def main(
    verdicts_jsonl: Path = typer.Option(...),
    out_sft: Path = typer.Option(...),
    out_dpo: Path = typer.Option(None),
):
    records: Dict[str, Dict] = {}

    # 1) Load passed records
    with verdicts_jsonl.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            if rec.get("verdict", {}).get("passed"):
                records[f"{rec['task_id']}::{i}"] = rec

    # 2) Dedup
    records = dedup_by_program_hash(records)

    # 3) Filter
    kept = apply_filters(records.values())

    # 4) Build SFT
    sft = build_sft_dataset(kept)
    out_sft.parent.mkdir(parents=True, exist_ok=True)
    with out_sft.open("w", encoding="utf-8") as f:
        for ex in sft:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    typer.echo(f"[OK] wrote SFT dataset: {out_sft} ({len(sft)} examples)")

    # 5) Optional DPO
    if out_dpo:
        with verdicts_jsonl.open("r", encoding="utf-8") as f:
            all_records = [json.loads(l) for l in f]
        dpo = build_dpo_pairs(all_records)
        with out_dpo.open("w", encoding="utf-8") as f:
            for ex in dpo:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        typer.echo(f"[OK] wrote DPO dataset: {out_dpo} ({len(dpo)} pairs)")


if __name__ == "__main__":
    app()
