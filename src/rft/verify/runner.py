from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rft.tasks.spec import TaskSpec
from rft.verify.sandbox import eval_sandbox
from rft.verify.signals import FailureTag, classify_failure


@dataclass
class EvalVerdict:
    passed: bool
    failure_tag: Optional[FailureTag]
    stdout: str
    stderr: str
    exit_code: int
    runtime_sec: float
    timeout: bool


def run_eval(
    task: TaskSpec,
    program: str,
    *,
    benchmark_root: Path,
    timeout_sec: int,
) -> EvalVerdict:
    """
    Run a single program against task.eval_entrypoint.
    """
    timeout_sec = timeout_sec or task.eval_timeout_sec

    with eval_sandbox() as workdir:
        # ------------------------------------------------------------
        # 1. Write submission program
        # ------------------------------------------------------------
        prog_path = workdir / "submission.py"
        prog_path.write_text(program, encoding="utf-8")


        # ------------------------------------------------------------
        # 2. Prepare command and environment
        # ------------------------------------------------------------
        cmd = task.eval_entrypoint

        env = os.environ.copy()

        if task.benchmark_root is not None:
            # IMPORTANT:
            # PYTHONPATH must contain the *parent* of `benchmark/`,
            # not `benchmark/` itself.
            repo_root = Path(task.benchmark_root).resolve()
            if repo_root.name == "benchmark":
                repo_root = repo_root.parent

            env["PYTHONPATH"] = (
                str(repo_root)
                + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            )


        # ------------------------------------------------------------
        # 3. Execute verifier
        # ------------------------------------------------------------
        start = time.time()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(benchmark_root.resolve())


        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                env=env,                    
            )
            runtime = time.time() - start

            passed = proc.returncode == 0
            tag = None if passed else classify_failure(
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                timeout=False,
            )

            return EvalVerdict(
                passed=passed,
                failure_tag=tag,
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
                runtime_sec=runtime,
                timeout=False,
            )

        except subprocess.TimeoutExpired as e:
            runtime = time.time() - start
            return EvalVerdict(
                passed=False,
                failure_tag=FailureTag.TIMEOUT,
                stdout=e.stdout or "",
                stderr=e.stderr or "",
                exit_code=-1,
                runtime_sec=runtime,
                timeout=True,
            )
