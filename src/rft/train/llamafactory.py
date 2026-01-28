from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


def run_llamafactory_train(
    *,
    config_path: Path,
    env_name: str = "rft-train",
    extra_env: Optional[dict] = None,
) -> None:
    """
    Invoke LLaMA Factory training in a separate conda environment.
    """
    cmd = [
        "conda", "run",
        "-n", env_name,
        "llamafactory-cli", "train",
        str(config_path),
    ]

    env = None
    if extra_env:
        import os
        env = os.environ.copy()
        env.update(extra_env)

    print("[RFT] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
