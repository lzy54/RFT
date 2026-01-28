from __future__ import annotations

from enum import Enum
from typing import Optional


class FailureTag(str, Enum):
    # Infrastructure / environment
    TIMEOUT = "TIMEOUT"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    IMPORT_ERROR = "IMPORT_ERROR"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    PERMISSION_ERROR = "PERMISSION_ERROR"

    # Evaluation / logic
    METRIC_BELOW_THRESHOLD = "METRIC_BELOW_THRESHOLD"
    ASSERTION_FAILED = "ASSERTION_FAILED"
    WRONG_OUTPUT_FORMAT = "WRONG_OUTPUT_FORMAT"

    # RFT / pipeline
    FORMAT_ERROR = "FORMAT_ERROR"          # from parse.py
    EMPTY_PROGRAM = "EMPTY_PROGRAM"
    UNKNOWN = "UNKNOWN"


def classify_failure(
    *,
    exit_code: int,
    stdout: str,
    stderr: str,
    timeout: bool,
) -> FailureTag:
    if timeout:
        return FailureTag.TIMEOUT

    err = (stderr or "").lower()
    out = (stdout or "").lower()

    if "importerror" in err or "modulenotfounderror" in err:
        return FailureTag.IMPORT_ERROR
    if "filenotfounderror" in err or "no such file" in err:
        return FailureTag.FILE_NOT_FOUND
    if "permission denied" in err:
        return FailureTag.PERMISSION_ERROR
    if "assert" in err:
        return FailureTag.ASSERTION_FAILED
    if "metric" in err and "threshold" in err:
        return FailureTag.METRIC_BELOW_THRESHOLD

    if exit_code != 0:
        return FailureTag.RUNTIME_ERROR

    return FailureTag.UNKNOWN
