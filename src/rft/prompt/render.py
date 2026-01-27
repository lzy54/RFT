from __future__ import annotations

from typing import Dict, List

from rft.tasks.spec import TaskSpec
from rft.tasks.io import get_model_visible_input
from rft.prompt.templates import THINKING_CODE_TEMPLATE


def render(task: TaskSpec) -> List[Dict[str, str]]:
    """
    Render a TaskSpec into ShareGPT-style messages.

    This is the ONLY canonical way to construct prompts
    for thinking + code generation in the RFT pipeline.
    """

    # System message: output contract & behavior constraints
    system_message = {
        "role": "system",
        "content": THINKING_CODE_TEMPLATE,
    }

    # User message: task-visible factual input
    user_message = {
        "role": "user",
        "content": get_model_visible_input(task),
    }

    return [system_message, user_message]
