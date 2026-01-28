#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np


def check_output_files():
    expected_files = [
        "./pred_results/pypesto_best_parameters.csv",
        "./pred_results/pypesto_summary.csv",
        "./pred_results/pypesto_results.png",
        "./pred_results/pypesto_predictions.csv",
    ]
    missing = [f for f in expected_files if not os.path.exists(f)]
    if missing:
        return False, f"Missing files: {[os.path.basename(f) for f in missing]}"
    return True, ""


def check_best_parameters():
    path = "./pred_results/pypesto_best_parameters.csv"
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return False, f"Failed to read best_parameters: {e}"

    required_cols = ["parameter", "estimate", "log_estimate", "objective_value"]
    if any(c not in df.columns for c in required_cols):
        return False, f"Missing required columns in best_parameters: {required_cols}"

    if len(df) != 2:
        return False, f"Expected 2 parameter rows (k1,k2), got {len(df)}"

    params = set(df["parameter"])
    if params != {"k1", "k2"}:
        return False, f"Expected parameters {{'k1','k2'}}, got {params}"

    estimates = df["estimate"].values
    if not np.all(np.isfinite(estimates)):
        return False, "Non-finite parameter estimates"
    if np.any(estimates <= 0):
        return False, "Parameter estimates must be > 0"
    if np.any(estimates > 1e6):
        return False, "Parameter estimates too large (>1e6)"

    obj_vals = df["objective_value"].values
    if not np.all(np.isfinite(obj_vals)):
        return False, "Non-finite objective_value"
    if np.any(obj_vals > 1e6):
        return False, "objective_value too large (>1e6)"

    return True, ""


def check_summary():
    path = "./pred_results/pypesto_summary.csv"
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return False, f"Failed to read summary: {e}"

    if "metric" not in df.columns or "value" not in df.columns:
        return False, "Summary must have columns: metric, value"

    summary = dict(zip(df["metric"], df["value"]))

    required_metrics = ["best_objective", "k1_estimate", "k2_estimate"]
    missing = [m for m in required_metrics if m not in summary]
    if missing:
        return False, f"Missing required metrics in summary: {missing}"

    vals = np.array([summary["best_objective"], summary["k1_estimate"], summary["k2_estimate"]])
    if not np.all(np.isfinite(vals)):
        return False, "Non-finite values in summary"

    if summary["k1_estimate"] <= 0 or summary["k2_estimate"] <= 0:
        return False, "k1_estimate and k2_estimate must be > 0"

    if summary["best_objective"] > 1e6:
        return False, "best_objective too large (>1e6)"

    return True, ""


def check_predictions():
    path = "./pred_results/pypesto_predictions.csv"
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return False, f"Failed to read predictions: {e}"

    required_cols = ["time", "measurement", "prediction"]
    if any(c not in df.columns for c in required_cols):
        return False, f"Missing required columns in predictions: {required_cols}"

    time_vals = df["time"].values
    meas_vals = df["measurement"].values
    pred_vals = df["prediction"].values

    if not np.all(np.isfinite(time_vals)):
        return False, "Non-finite time values in predictions"
    if not np.all(np.isfinite(meas_vals)):
        return False, "Non-finite measurement values in predictions"
    if not np.all(np.isfinite(pred_vals)):
        return False, "Non-finite prediction values"

    return True, ""


def check_png():
    path = "./pred_results/pypesto_results.png"
    if not os.path.exists(path):
        return False, "Visualization file not found"
    size = os.path.getsize(path)
    if size < 2000:
        return False, f"Visualization file too small: {size} bytes"
    return True, ""


def eval():
    checks = [
        check_output_files,
        check_best_parameters,
        check_summary,
        check_predictions,
        check_png,
    ]

    for fn in checks:
        passed, msg = fn()
        if not passed:
            return False, msg

    return True, "All checks passed"


if __name__ == "__main__":
    ok, msg = eval()
    print("Result:", ok, "-", msg)
    raise SystemExit(0 if ok else 1)
