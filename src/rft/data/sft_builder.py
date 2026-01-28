from __future__ import annotations

from typing import Dict, List


def build_sharegpt_sft_example(rec: Dict) -> Dict:
    """
    Build one ShareGPT-style SFT example.

    Output schema:
    {
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
      ],
      "meta": {...}
    }
    """
    messages = rec["input_messages"]

    assistant_content = ""
    if rec.get("thinking"):
        assistant_content += f"<thinking>\n{rec['thinking']}\n</thinking>\n\n"
    assistant_content += f"```python\n{rec['program']}\n```"

    return {
        "messages": messages + [
            {
                "role": "assistant",
                "content": assistant_content,
            }
        ],
        "meta": {
            "task_id": rec["task_id"],
            "run_id": rec.get("run_id"),
            "sample_id": rec.get("sample_id"),
            "program_hash": rec.get("_program_hash"),
            "source_model": rec.get("model"),
        },
    }

def messages_to_conversations(messages):
    role_map = {
        "system": "system",
        "user": "human",
        "assistant": "assistant",
    }
    return [
        {
            "from": role_map[m["role"]],
            "value": m["content"],
        }
        for m in messages
    ]


def build_sft_dataset(records: List[Dict]) -> List[Dict]:
    """
    Build LLaMA-Factory-compatible SFT dataset.
    Each example contains:
      {
        "conversations": [...],
        "meta": {...}   # optional, LF will ignore it
      }
    """
    dataset = []
    for rec in records:
        sharegpt = build_sharegpt_sft_example(rec)

        dataset.append({
            "conversations": messages_to_conversations(sharegpt["messages"]),
            "meta": sharegpt.get("meta", {}),
        })

    return dataset

