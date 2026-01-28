from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Dict, Any, Set, List

import pandas as pd


PRED_DIR = Path("./pred_results")
GOLD_DIR = Path("./benchmark/eval_programs/gold_results")

PRED_PARAMS = PRED_DIR / "estimated_parameters.tsv"
PRED_SUMMARY = PRED_DIR / "objective_summary.json"
PRED_HEAD = PRED_DIR / "sim_vs_data_head.csv"
PRED_PLOT = PRED_DIR / "fit_plot.png"
PRED_REMOVED_TXT = PRED_DIR / "removed_measurements_head.txt"

GOLD_PARAMS = GOLD_DIR / "pesto_conversion_estimated_parameters_gold.tsv"
GOLD_SUMMARY = GOLD_DIR / "pesto_conversion_objective_summary_gold.json"
GOLD_REMOVED = GOLD_DIR / "pesto_conversion_removed_measurements.tsv"  

def _fail(msg: str) -> Tuple[bool, str]:
    return False, msg


def _ok(msg: str) -> Tuple[bool, str]:
    return True, msg


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _nonempty_file(path: Path) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size > 0
    except Exception:
        return False


def _require_files(paths: List[Path]) -> Tuple[bool, str]:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        return _fail(f"Missing required output files: {missing}")
    return _ok("All required output files exist.")


def _check_params_schema_and_ids() -> Tuple[bool, str]:
    if not GOLD_PARAMS.exists():
        return _fail(
            f"Missing gold params file: {GOLD_PARAMS}. "
            "You must copy pred_results into gold_results after running gold program."
        )

    pred = pd.read_csv(PRED_PARAMS, sep="\t")
    gold = pd.read_csv(GOLD_PARAMS, sep="\t")

    for col in ["parameterId", "estimate"]:
        if col not in pred.columns:
            return _fail(f"{PRED_PARAMS} missing required column '{col}'")

    if "parameterId" not in gold.columns:
        return _fail(f"{GOLD_PARAMS} missing required column 'parameterId'")

    pred_ids: Set[str] = set(pred["parameterId"].astype(str).tolist())
    gold_ids: Set[str] = set(gold["parameterId"].astype(str).tolist())

    if pred_ids != gold_ids:
        missing = sorted(list(gold_ids - pred_ids))
        extra = sorted(list(pred_ids - gold_ids))
        return _fail(f"parameterId set mismatch vs gold. missing={missing[:10]} extra={extra[:10]}")

    if pred["estimate"].isna().any():
        return _fail("Some parameter estimates are NaN.")

    return _ok(f"estimated_parameters.tsv schema OK; matched {len(pred_ids)} parameterIds with gold.")


def _check_objective_threshold() -> Tuple[bool, str]:
    if not GOLD_SUMMARY.exists():
        return _fail(
            f"Missing gold summary file: {GOLD_SUMMARY}. "
            "You must copy pred_results into gold_results after running gold program."
        )

    pred_sum = _read_json(PRED_SUMMARY)
    gold_sum = _read_json(GOLD_SUMMARY)

    if "best_fval" not in pred_sum or pred_sum["best_fval"] is None:
        return _fail("objective_summary.json missing 'best_fval' or it is null.")
    if "best_fval" not in gold_sum or gold_sum["best_fval"] is None:
        return _fail("Gold objective summary missing 'best_fval' or it is null.")

    pred_fval = float(pred_sum["best_fval"])
    gold_fval = float(gold_sum["best_fval"])

    abs_tol = 1e-4
    rel_tol = 0.05  
    threshold = gold_fval + abs_tol + rel_tol * abs(gold_fval)

    if pred_fval > threshold:
        return _fail(f"Objective too high: pred best_fval={pred_fval:.6g} > allowed={threshold:.6g} (gold={gold_fval:.6g})")

    return _ok(f"Objective OK: pred best_fval={pred_fval:.6g}, gold={gold_fval:.6g}, allowed<= {threshold:.6g}.")


def _check_head_csv_schema() -> Tuple[bool, str]:
    df = pd.read_csv(PRED_HEAD)

    required = ["observableId", "conditionId", "time", "measurement", "simulation"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return _fail(f"{PRED_HEAD} missing required columns: {missing}")

    if len(df) == 0:
        return _fail(f"{PRED_HEAD} is empty.")

    if df["time"].isna().any():
        return _fail("sim_vs_data_head.csv contains NaN time values.")
    if df["measurement"].isna().any():
        return _fail("sim_vs_data_head.csv contains NaN measurement values.")

    return _ok(f"sim_vs_data_head.csv schema OK; rows={len(df)}.")


def _contamination_check_removed_rows() -> Tuple[bool, str]:
    if not GOLD_REMOVED.exists():
        return _ok(f"Removed-measurements gold file not found ({GOLD_REMOVED}). Skipping contamination check.")

    removed = pd.read_csv(GOLD_REMOVED, sep="\t")
    pred_head = pd.read_csv(PRED_HEAD)

    candidate_cols = ["observableId", "conditionId", "time", "measurement"]
    common = [c for c in candidate_cols if c in removed.columns and c in pred_head.columns]

    if not common:
        return _ok(f"No common columns for contamination check between {GOLD_REMOVED.name} and {PRED_HEAD.name}. Skipping.")

    def _tuples(df: pd.DataFrame, cols: List[str]) -> Set[tuple]:
        out: Set[tuple] = set()
        sub = df[cols].copy()
        for c in cols:
            sub[c] = sub[c].astype(str)
        for row in sub.itertuples(index=False, name=None):
            out.add(tuple(row))
        return out

    removed_keys = _tuples(removed, common)
    pred_keys = _tuples(pred_head, common)

    overlap = removed_keys.intersection(pred_keys)
    if overlap:
        sample = list(overlap)[:3]
        return _fail(f"Contamination check failed: removed rows appear in sim_vs_data_head.csv. Sample overlap keys={sample}")

    return _ok(f"Contamination check passed using columns={common}.")


def eval() -> Tuple[bool, str]:
    ok, msg = _require_files([PRED_PARAMS, PRED_SUMMARY, PRED_HEAD, PRED_PLOT])
    if not ok:
        return ok, msg

    if not PRED_REMOVED_TXT.exists():
        pass

    if not _nonempty_file(PRED_PLOT):
        return _fail(f"{PRED_PLOT} is missing or empty.")

    ok, msg = _check_params_schema_and_ids()
    if not ok:
        return ok, msg

    ok, msg = _check_objective_threshold()
    if not ok:
        return ok, msg

    ok, msg = _check_head_csv_schema()
    if not ok:
        return ok, msg

    ok, msg = _contamination_check_removed_rows()
    if not ok:
        return ok, msg

    return _ok("Passed: all checks succeeded (files, schema, objective threshold, contamination).")


if __name__ == "__main__":
    passed, info = eval()
    print({"passed": passed, "info": info})