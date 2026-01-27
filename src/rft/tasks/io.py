from __future__ import annotations

from typing import Optional

from rft.tasks.spec import TaskSpec


# ----------------------------------------------------------------------
# Dataset preview tokens (SAB / AutoSDT convention)
# ----------------------------------------------------------------------
PREVIEW_START_TOKEN = "[START Preview of Dataset]"
PREVIEW_END_TOKEN = "[END Preview of Dataset]"


def get_task_instruction(task: TaskSpec) -> str:
    """
    Return the task instruction presented to the model.

    This function intentionally performs no formatting or decoration.
    """
    return task.instruction.strip()


def get_dataset_preview(task: TaskSpec) -> Optional[str]:
    """
    Return the dataset preview wrapped with START/END tokens,
    or None if the task has no dataset preview.
    """
    if not task.dataset_preview:
        return None

    preview = task.dataset_preview.strip()

    # Avoid double-wrapping if tokens already exist
    if preview.startswith(PREVIEW_START_TOKEN):
        return preview

    return (
        f"{PREVIEW_START_TOKEN}\n"
        f"{preview}\n"
        f"{PREVIEW_END_TOKEN}"
    )


def get_model_visible_input(task: TaskSpec) -> str:
    """
    Return the full model-visible input text:
    - instruction
    - optional dataset preview (with tokens)

    This function is the canonical source of truth for
    "what the model is allowed to see" about a task.
    """
    parts = []

    instruction = get_task_instruction(task)
    if instruction:
        parts.append(instruction)

    preview = get_dataset_preview(task)
    if preview:
        parts.append(preview)

    return "\n\n".join(parts)


def print_task_input(task: TaskSpec) -> None:
    """
    Print instruction + dataset preview to stdout.

    This function exists mainly for:
    - debugging
    - sanity checks
    - interactive inspection
    """
    text = get_model_visible_input(task)
    print(text)
