from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import typer

from rft.inference.parse import parse_output
from rft.inference.sampler import SamplingConfig, sample_trajectories
from rft.inference.vllm_client import VLLMClient, VLLMConfig
from rft.prompt.render import render
from rft.tasks.registry import TaskRegistry


app = typer.Typer(add_completion=False)


def _now_run_id() -> str:
    return time.strftime("%Y-%m-%d_%H%M%S", time.localtime())


@app.command()
def main(
    annotation_path: Path = typer.Option(..., help="Path to task annotation CSV/JSON/JSONL."),
    task_id: str = typer.Option(..., help="Task id to run."),
    split: str = typer.Option("sab", help="Dataset split label (sab/sabval/autosdt)."),
    benchmark_root: Path = typer.Option(Path("."), help="Benchmark repo root (for eval entry resolution)."),
    out_jsonl: Path = typer.Option(Path("runs/tmp/candidates.jsonl"), help="Output candidates.jsonl path."),
    model: Optional[str] = typer.Option(None, help="Model name used by vLLM (overrides env/config)."),
    n: int = typer.Option(2, help="Number of samples for the task."),
    temperature: float = typer.Option(0.7, help="Sampling temperature."),
    top_p: float = typer.Option(0.95, help="Sampling top_p."),
    max_tokens: int = typer.Option(2048, help="Max tokens for generation."),
    seed: Optional[int] = typer.Option(None, help="Optional deterministic seed."),
) -> None:
    """
    Minimal generation runner:
      - load TaskSpec
      - render ShareGPT messages
      - sample N trajectories from vLLM
      - parse outputs into thinking/program and enrich records in-place
    """
    run_id = _now_run_id()

    # 1) Task registry
    registry = TaskRegistry(annotation_path=annotation_path, benchmark_root=benchmark_root, split=split)
    task = registry.get(task_id)

    # 2) Render prompt messages
    messages = render(task)

    # 3) vLLM client
    vcfg = VLLMConfig.from_env()
    mname = model or (model if model else None) or (model or None) or (model or None)
    # prefer explicit CLI model; otherwise env VLLM_MODEL; otherwise error
    if mname is None:
        mname = model or None
    if mname is None:
        mname = (model or None) or (None)
    if mname is None:
        mname = (model or None)
    if mname is None:
        mname = (model or None)
    if mname is None:
        mname = (model or None)

    # Resolve model name sanely
    if mname is None:
        mname = (model or "").strip() or (  # noqa: PLW0127
            (Path().resolve() and "")  # dummy to keep mypy quiet
        )
    # The above block is intentionally defensive for early-stage configs;
    # we now do the actual resolution:
    mname = (model or "").strip() or (  # CLI
        (  # ENV
            __import__("os").getenv("VLLM_MODEL", "").strip()
        )
    )
    if not mname:
        raise typer.BadParameter("Model name not provided. Set --model or export VLLM_MODEL.")

    client = VLLMClient(vcfg)

    try:
        # 4) Sample raw candidates
        cfg = SamplingConfig(
            model=mname,
            n=n,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )
        sample_trajectories(
            client=client,
            task_id=task.task_id,
            messages=messages,
            cfg=cfg,
            run_id=run_id,
            out_jsonl=out_jsonl,
        )
    finally:
        client.close()

    # 5) Enrich: parse raw outputs and write a new jsonl (atomic-ish rewrite)
    tmp_out = out_jsonl.with_suffix(".parsed.tmp.jsonl")
    tmp_out.parent.mkdir(parents=True, exist_ok=True)

    with out_jsonl.open("r", encoding="utf-8") as fin, tmp_out.open("w", encoding="utf-8") as fout:
        for line in fin:
            rec: Dict[str, Any] = json.loads(line)
            if rec.get("run_id") != run_id or rec.get("task_id") != task.task_id:
                # keep old lines untouched (allows appending multiple runs)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            raw = rec.get("raw_output_text", "")
            parsed = parse_output(raw)
            rec["parse"] = {
                "ok": parsed.ok,
                "status": parsed.status,
                "details": parsed.details,
            }
            rec["thinking"] = parsed.thinking
            rec["program"] = parsed.program
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    tmp_out.replace(out_jsonl)

    typer.echo(f"[OK] wrote candidates: {out_jsonl}")
    typer.echo(f"run_id={run_id}, task_id={task.task_id}, n={n}")


if __name__ == "__main__":
    app()

