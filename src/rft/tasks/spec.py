from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class TaskSpec:
    """
    Production-grade TaskSpec for verifier-guided Rejective Fine-Tuning (RFT)
    on scientific and code-centric tasks.

    TaskSpec is a *runtime contract*:
    - Immutable
    - Declarative
    - Verifier-centric
    """

    # ============================================================
    # Identity
    # ============================================================
    task_id: str
    split: str
    version: str = "v1"

    # High-level categorization (not shown to model by default)
    domain: Optional[str] = None
    subtask_categories: Optional[List[str]] = None

    # ============================================================
    # Model-facing inputs
    # ============================================================
    instruction: str = ""
    dataset_preview: Optional[str] = None

    # Optional hints for IO / formatting (natural language, non-binding)
    input_format_hint: Optional[str] = None

    # ============================================================
    # Verification (hard authority)
    # ============================================================
    # Executable verifier entrypoint, e.g.:
    #   "python -m benchmark.eval_programs.eval_tox21_scaffold"
    eval_entrypoint: str = ""

    # Wall-clock timeout for verifier execution
    eval_timeout_sec: int = 1800

    # Optional hard memory limit for sandbox execution
    max_memory_mb: Optional[int] = 4096

    # ============================================================
    # Execution environment (declarative, non-enforced by default)
    # ============================================================
    # Example:
    # {
    #   "pip": ["rdkit", "scipy==1.10.1"],
    #   "conda_env": "sab-rdkit"
    # }
    execution_env: Dict[str, Any] = field(default_factory=dict)

    # ============================================================
    # Evaluation parameters (passed through to verifier)
    # ============================================================
    # Example:
    # {
    #   "numerical_precision": 1e-6,
    #   "relative_slack": 0.1
    # }
    evaluation_params: Dict[str, Any] = field(default_factory=dict)

    # ============================================================
    # Sampling hints (soft guidance, non-binding)
    # ============================================================
    # Example:
    # {
    #   "sample_count": 64,
    #   "temperature": 0.7
    # }
    sampling_hint: Dict[str, Any] = field(default_factory=dict)

    # ============================================================
    # Annotation & provenance metadata (never consumed by model)
    # ============================================================
    # This field absorbs annotation-time information such as:
    # - dataset_folder_tree
    # - src_file_or_path
    # - gold_program_name
    # - output_fname
    # - annotator notes
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ============================================================
    # Validation
    # ============================================================
    def validate(self) -> None:
        if not self.task_id:
            raise ValueError("TaskSpec.task_id must be non-empty")

        if not self.split:
            raise ValueError("TaskSpec.split must be non-empty")

        if not self.eval_entrypoint:
            raise ValueError(
                f"TaskSpec.eval_entrypoint missing for task '{self.task_id}'"
            )

        if self.subtask_categories is not None:
            if not isinstance(self.subtask_categories, list):
                raise TypeError(
                    "TaskSpec.subtask_categories must be a list of strings"
                )

    # ============================================================
    # Helpers
    # ============================================================
    def short_name(self) -> str:
        return f"{self.split}:{self.task_id}"

    def has_dataset_preview(self) -> bool:
        return bool(self.dataset_preview)

    def requires_special_env(self) -> bool:
        return bool(self.execution_env)
