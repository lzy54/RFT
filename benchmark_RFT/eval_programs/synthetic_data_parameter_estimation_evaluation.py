#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np


PRED_PARAMS = "./pred_results/pyPESTO_synthetic_params.csv"
PRED_DETAILED = "./pred_results/pyPESTO_synthetic_parameter_estimates_detailed.csv"
PRED_SUMMARY = "./pred_results/pyPESTO_synthetic_optimization_summary.csv"
PRED_PLOT = "./pred_results/pyPESTO_synthetic_comparison.png"

GOLD_TRUE = "./benchmark/eval_programs/gold_results/pyPESTO_true_params.csv"


def check_files():
    required_pred = [PRED_PARAMS]
    required_gold = [GOLD_TRUE]
    optional_pred = [PRED_DETAILED, PRED_SUMMARY, PRED_PLOT]

    missing_required = [f for f in required_pred + required_gold if not os.path.exists(f)]
    missing_optional = [f for f in optional_pred if not os.path.exists(f)]

    if missing_required:
        return False, f"Missing required files: {missing_required}"

    msg = ""
    if missing_optional:
        msg = f"Optional files missing (not fatal): {missing_optional}"

    return True, msg


def validate_parameters_structure():
    try:
        pred_df = pd.read_csv(PRED_PARAMS)
        gold_df = pd.read_csv(GOLD_TRUE)
    except Exception as e:
        return False, f"Failed to read CSV files: {e}"

    required_pred_cols = {"param_id", "est_value"}
    required_gold_cols = {"param_id", "true_value"}

    if not required_pred_cols.issubset(pred_df.columns):
        return False, f"Prediction file must contain columns {required_pred_cols}, got {list(pred_df.columns)}"

    if not required_gold_cols.issubset(gold_df.columns):
        return False, f"Gold file must contain columns {required_gold_cols}, got {list(gold_df.columns)}"

    pred_params = set(pred_df["param_id"])
    gold_params = set(gold_df["param_id"])

    if pred_params != gold_params:
        return False, f"Parameter id mismatch between prediction and gold: pred={pred_params}, gold={gold_params}"

    if pred_df["est_value"].isna().any():
        return False, "Prediction file contains NaN in est_value"

    if not np.isfinite(pred_df["est_value"]).all():
        return False, "Prediction file contains non-finite est_value"

    return True, ""


def validate_summary():
    if not os.path.exists(PRED_SUMMARY):
        return True, "Summary file not found, skipping summary checks"

    try:
        df = pd.read_csv(PRED_SUMMARY)
    except Exception as e:
        return False, f"Failed to read summary file: {e}"

    required_cols = {"method", "best_objective", "convergence_success"}
    if not required_cols.issubset(df.columns):
        return False, f"Summary file must contain columns {required_cols}, got {list(df.columns)}"

    if len(df) < 1:
        return False, "Summary file has no rows"

    for _, row in df.iterrows():
        if not np.isfinite(row["best_objective"]) or row["best_objective"] < 0:
            return False, f"Invalid best_objective for method {row['method']}: {row['best_objective']}"
        if not bool(row["convergence_success"]):
            return False, f"Optimization did not converge for method {row['method']}"

    return True, ""


def validate_plot():
    if not os.path.exists(PRED_PLOT):
        return True, "Plot file not found, skipping plot checks"

    try:
        size = os.path.getsize(PRED_PLOT)
    except Exception as e:
        return False, f"Failed to stat plot file: {e}"

    if size < 5000:
        return False, f"Plot file too small ({size} bytes), may be invalid"

    return True, ""


def check_contamination():
    try:
        pred_df = pd.read_csv(PRED_PARAMS)
        gold_df = pd.read_csv(GOLD_TRUE)
    except Exception as e:
        return False, f"Failed to read files for contamination check: {e}"

    merged = gold_df.merge(pred_df, on="param_id", how="inner")
    if len(merged) == 0:
        return False, "No overlapping param_id between prediction and gold in contamination check"

    true_vals = merged["true_value"].to_numpy(dtype=float)
    est_vals = merged["est_value"].to_numpy(dtype=float)

    if np.array_equal(true_vals, est_vals):
        return False, "Predicted parameters are exactly equal to gold parameters for all entries (possible contamination)"

    return True, ""


def compare_with_gold():
    """
    Compare predicted parameters with gold, using per-parameter relative error thresholds.
    This evaluates all points (K1, K2, K3, K5) but with different acceptable error ranges.
    """
    try:
        pred_df = pd.read_csv(PRED_PARAMS)
        gold_df = pd.read_csv(GOLD_TRUE)
    except Exception as e:
        return False, f"Failed to read files for gold comparison: {e}"

    merged = gold_df.merge(pred_df, on="param_id", how="inner")
    if len(merged) == 0:
        return False, "No overlapping param_id between prediction and gold for comparison"

    # Per-parameter relative error thresholds
    thresholds = {
        "K1": 0.1,   # 10%
        "K2": 1.0,   # 100%
        "K3": 1.0,   # 100%
        "K5": 5.0,   # 500%
    }

    details = []
    per_param_pass = []

    rel_errors = []

    for _, row in merged.iterrows():
        pid = row["param_id"]
        t = float(row["true_value"])
        e = float(row["est_value"])
        rel_err = abs(e - t) / max(abs(t), 1e-8)
        rel_errors.append(rel_err)

        thr = thresholds.get(pid, 1.0)
        ok = rel_err <= thr
        per_param_pass.append(ok)

        details.append(
            f"{pid}: est={e:.6g}, true={t:.6g}, rel_err={rel_err:.3f}, "
            f"threshold={thr:.3f}, passed={ok}"
        )

    max_rel_err = float(np.max(rel_errors))
    mean_rel_err = float(np.mean(rel_errors))

    passed = all(per_param_pass)
    info = (
        f"max_relative_error={max_rel_err:.4f}, "
        f"mean_relative_error={mean_rel_err:.4f}; "
        "per-parameter details: " + " | ".join(details)
    )

    if not passed:
        return False, info

    return True, info


def eval():
    print("🔬 Evaluating pyPESTO Synthetic Parameter Estimation")
    print("=" * 60)

    checks = [
        ("Files", check_files),
        ("Parameter structure", validate_parameters_structure),
        ("Summary", validate_summary),
        ("Plot", validate_plot),
        ("Contamination", check_contamination),
        ("Gold comparison", compare_with_gold),
    ]

    results = []
    for name, func in checks:
        try:
            success, message = func()
            results.append((name, success, message))
            status = "PASSED" if success else "FAILED"
            print(f"{'✅' if success else '❌'} {name}: {status}")
            if message:
                print(f"   {message}")
        except Exception as e:
            results.append((name, False, f"Exception: {e}"))
            print(f"💥 {name}: ERROR - {e}")

    passed = sum(1 for _, s, _ in results if s)
    total = len(results)
    print(f"\n📊 Summary: {passed}/{total} checks passed")

    if passed == total:
        return True, f"All checks passed: {passed}/{total}"
    else:
        failed = [name for name, s, _ in results if not s]
        return False, f"Failed checks: {failed}"


if __name__ == "__main__":
    success, message = eval()
    raise SystemExit(0 if success else 1)
