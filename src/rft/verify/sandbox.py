from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def eval_sandbox(
    *,
    keep: bool = False,
) -> Iterator[Path]:
    """
    Create an isolated temp directory for evaluation.

    All eval scripts should be executed inside this directory.
    """
    root = Path(tempfile.mkdtemp(prefix="rft_eval_"))
    try:
        # Limit thread usage for safety / determinism
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

        yield root
    finally:
        if not keep:
            shutil.rmtree(root, ignore_errors=True)
