# scripts/build_annotation_table.py
from pathlib import Path
import csv

BENCHMARK_ROOT = Path("benchmark")
TASKS_DIR = BENCHMARK_ROOT / "tasks"
OUT = Path("annotations_initial.csv")

rows = []

for task_dir in TASKS_DIR.iterdir():
    if not task_dir.is_dir():
        continue

    task_id = task_dir.name
    inst = (task_dir / "instruction.txt").read_text().strip()
    preview_path = task_dir / "dataset_preview.txt"

    rows.append({
        "task_id": task_id,
        "task_inst": inst,
        "dataset_preview": preview_path.read_text().strip() if preview_path.exists() else "",
        "eval_script_name": f"eval_{task_id}.py",
    })

with OUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "task_id",
            "task_inst",
            "dataset_preview",
            "eval_script_name",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUT}")
