# rft/cli/run_eval.py
from __future__ import annotations

import subprocess
from pathlib import Path
import typer

from rft.train.eval_loop import compute_stats, write_report

app = typer.Typer(add_completion=False)


@app.command()
def main(
    run_dir: Path = typer.Option(
        ...,
        help="Directory to store evaluation run artifacts",
    ),
    checkpoint: Path = typer.Option(
        ...,
        help="Path to trained model checkpoint",
    ),
):
    """
    Run inference + verification using a trained checkpoint,
    then compute evaluation statistics.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    candidates = run_dir / "candidates.jsonl"
    verdicts = run_dir / "verdicts.jsonl"
    report = run_dir / "report.md"

    # 1) Inference
    typer.echo("[RFT][Eval] Running inference...")
    subprocess.run(
        [
            "python", "-m", "rft.cli.run_generate",
            "--checkpoint", str(checkpoint),
            "--out", str(candidates),
        ],
        check=True,
    )

    # 2) Verification
    typer.echo("[RFT][Eval] Running verification...")
    subprocess.run(
        [
            "python", "-m", "rft.cli.run_verify",
            "--candidates-jsonl", str(candidates),
            "--out-verdicts", str(verdicts),
        ],
        check=True,
    )

    # 3) Stats + report
    typer.echo("[RFT][Eval] Computing statistics...")
    verdict_records = []
    with verdicts.open("r", encoding="utf-8") as f:
        verdict_records = [json.loads(l) for l in f]

    stats = compute_stats(verdict_records)
    write_report(stats, report)

    typer.echo(f"[OK] Evaluation finished. Report written to {report}")


if __name__ == "__main__":
    app()
