from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from rft.tasks.spec import TaskSpec


class TaskRegistry:
    """
    TaskRegistry is the *single source of truth* that translates
    annotation-time metadata + filesystem conventions into runtime TaskSpec.

    Responsibilities:
    - Load annotation tables (CSV / JSON / JSONL)
    - Discover tasks by split (sab / sab-validation / autosdt-5k)
    - Construct validated, immutable TaskSpec objects
    - Encapsulate all benchmark-specific conventions
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        annotation_path: Path,
        benchmark_root: Path,
        split: str,
        eval_module_prefix: str = "benchmark.eval_programs",
        default_eval_timeout_sec: int = 1800,
    ):
        """
        Parameters
        ----------
        annotation_path:
            Path to annotation table (CSV / JSON / JSONL).
        benchmark_root:
            Root directory of benchmark repository.
        split:
            Dataset split name (e.g. "sab", "sab-validation", "autosdt-5k").
        eval_module_prefix:
            Python module prefix used to invoke eval scripts.
        default_eval_timeout_sec:
            Default timeout for eval execution (seconds).
        """
        self.annotation_path = annotation_path
        self.benchmark_root = benchmark_root
        self.split = split
        self.eval_module_prefix = eval_module_prefix
        self.default_eval_timeout_sec = default_eval_timeout_sec

        self._rows: List[Dict] = []
        self._tasks: Dict[str, TaskSpec] = {}

        self._load_annotations()
        self._build_tasks()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def list_tasks(self) -> List[TaskSpec]:
        """
        Return all TaskSpec objects in this registry.
        """
        return list(self._tasks.values())

    def get(self, task_id: str) -> TaskSpec:
        """
        Fetch a TaskSpec by task_id.
        """
        if task_id not in self._tasks:
            raise KeyError(f"Task '{task_id}' not found in registry")
        return self._tasks[task_id]

    def __iter__(self) -> Iterable[TaskSpec]:
        return iter(self._tasks.values())

    # ------------------------------------------------------------------
    # Internal: annotation loading
    # ------------------------------------------------------------------
    def _load_annotations(self) -> None:
        """
        Load annotation rows from CSV / JSON / JSONL.
        """
        path = self.annotation_path
        if not path.exists():
            raise FileNotFoundError(f"Annotation file not found: {path}")

        if path.suffix == ".csv":
            self._rows = self._load_csv(path)
        elif path.suffix in {".json", ".jsonl"}:
            self._rows = self._load_json(path)
        else:
            raise ValueError(
                f"Unsupported annotation format: {path.suffix}"
            )

        if not self._rows:
            raise RuntimeError(
                f"No annotation rows loaded from {path}"
            )

    def _load_csv(self, path: Path) -> List[Dict]:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]

    def _load_json(self, path: Path) -> List[Dict]:
        if path.suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        else:  # jsonl
            rows = []
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return rows

    # ------------------------------------------------------------------
    # Internal: TaskSpec construction
    # ------------------------------------------------------------------
    def _build_tasks(self) -> None:
        """
        Translate annotation rows into TaskSpec objects.
        """
        for row in self._rows:
            spec = self._row_to_taskspec(row)
            if spec.task_id in self._tasks:
                raise ValueError(
                    f"Duplicate task_id detected: {spec.task_id}"
                )
            spec.validate()
            self._tasks[spec.task_id] = spec

    def _row_to_taskspec(self, row: Dict) -> TaskSpec:
        """
        Core translation logic: annotation row -> TaskSpec.
        """

        # -------------------------
        # Required fields
        # -------------------------
        task_id = self._require(row, "task_id")
        instruction = self._require(row, "task_inst")
        eval_script_name = self._require(row, "eval_script_name")

        # -------------------------
        # Optional fields
        # -------------------------
        domain = row.get("domain")
        subtask_categories = self._parse_list(
            row.get("subtask_categories")
        )

        dataset_preview = row.get("dataset_preview")

        # -------------------------
        # Eval entrypoint resolution
        # -------------------------
        eval_entrypoint = self._build_eval_entrypoint(
            eval_script_name
        )

        # -------------------------
        # Metadata passthrough
        # -------------------------
        metadata = {}
        for key in [
            "dataset_folder_tree",
            "src_file_or_path",
            "gold_program_name",
            "output_fname",
        ]:
            if key in row and row[key] not in (None, ""):
                metadata[key] = row[key]

        # -------------------------
        # Construct TaskSpec
        # -------------------------
        return TaskSpec(
            task_id=task_id,
            split=self.split,
            domain=domain,
            subtask_categories=subtask_categories,
            instruction=instruction,
            dataset_preview=dataset_preview,
            eval_entrypoint=eval_entrypoint,
            benchmark_root=self.benchmark_root,
            eval_timeout_sec=self.default_eval_timeout_sec,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_eval_entrypoint(self, eval_script_name: str) -> str:
        """
        Convert eval_script_name into an executable Python module entrypoint.

        Example:
            eval_tox21_scaffold.py
            -> python -m benchmark.eval_programs.eval_tox21_scaffold
        """
        module = eval_script_name
        if module.endswith(".py"):
            module = module[:-3]

        return f"python -m {self.eval_module_prefix}.{module}"

    @staticmethod
    def _require(row: Dict, key: str) -> str:
        if key not in row or not row[key]:
            raise ValueError(
                f"Annotation row missing required field '{key}': {row}"
            )
        return row[key]

    @staticmethod
    def _parse_list(value: Optional[str]) -> Optional[List[str]]:
        """
        Parse list-like annotation fields.
        Accepts:
            - list
            - comma-separated string
        """
        if value is None or value == "":
            return None
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        raise TypeError(f"Cannot parse list from value: {value}")
