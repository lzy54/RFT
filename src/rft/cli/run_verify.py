from __future__ import annotations

import json
from pathlib import Path

import typer

from rft.tasks.registry import TaskRegistry
from rft.verify.runner import run_eval

app = typer.Typer(add_completion=False)


@app.command()
def main(
    annotation_path: Path = typer.Option(...),
    candidates_jsonl: Path = typer.Option(...),
    verdicts_jsonl: Path = typer.Option(...),
    split: str = typer.Option("sab"),
    benchmark_root: Path = typer.Option(Path(".")),
    timeout_sec: int = typer.Option(300),
):
    registry = TaskRegistry(
        annotation_path=annotation_path,
        benchmark_root=benchmark_root,
        split=split,
    )


    verdicts_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with candidates_jsonl.open("r", encoding="utf-8") as fin, \
         verdicts_jsonl.open("w", encoding="utf-8") as fout:

        for line in fin:
            rec = json.loads(line)
            task = registry.get(rec["task_id"])

            program = rec.get("program")
            if not program:
                rec["verdict"] = {
                    "passed": False,
                    "failure_tag": "FORMAT_ERROR",
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            verdict = run_eval(
                task=task,
                program=program,
                benchmark_root=benchmark_root,
                timeout_sec=timeout_sec,
            )


            rec["verdict"] = {
                "passed": verdict.passed,
                "failure_tag": verdict.failure_tag.value if verdict.failure_tag else None,
                "runtime_sec": verdict.runtime_sec,
                "exit_code": verdict.exit_code,
                "stdout": verdict.stdout,
                "stderr": verdict.stderr,
                "timeout": verdict.timeout,
            }


            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    typer.echo(f"[OK] wrote verdicts to {verdicts_jsonl}")


if __name__ == "__main__":
    app()
