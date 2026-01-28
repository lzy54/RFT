from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

import petab.v1 as petab
import pypesto
import pypesto.engine
import pypesto.objective.roadrunner as pypesto_rr
import pypesto.optimize as optimize
import pypesto.petab


DATASET_DIR = Path("./benchmark/datasets/PESTO_conver")
PRED_DIR = Path("./pred_results")

PETAB_YAML = DATASET_DIR / "conversion_reaction.yaml"

REMOVED_GOLD_FILE = Path("./benchmark/eval_programs/gold_results/pesto_conversion_removed_measurements.tsv")

N_STARTS = 5
SEED = 1912

RR_REL_TOL = 1e-6
RR_ABS_TOL = 1e-12
RR_MAX_STEPS = 10000


def _ensure_dirs() -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)


def _safe_read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, sep="\t")


def _write_removed_measurements_head() -> None:
    out_path = PRED_DIR / "removed_measurements_head.txt"

    if REMOVED_GOLD_FILE.exists():
        df_removed = _safe_read_tsv(REMOVED_GOLD_FILE)
        head_txt = df_removed.head(10).to_csv(sep="\t", index=False)
        out_path.write_text(
            "Source: benchmark/eval_programs/gold_results/pesto_conversion_removed_measurements.tsv\n\n"
            + head_txt
        )
        return
    
    meas_path = DATASET_DIR / "measurements.tsv"
    df_meas = _safe_read_tsv(meas_path)
    head_txt = df_meas.head(10).to_csv(sep="\t", index=False)
    out_path.write_text(
        "WARNING: gold removed-measurements file not found.\n"
        "This is a fallback snippet from measurements.tsv. You should remove 5 rows during preprocessing\n"
        "and save them to benchmark/eval_programs/gold_results/pesto_conversion_removed_measurements.tsv\n\n"
        + head_txt
    )


def _extract_best_parameters(
    result: pypesto.Result, parameter_ids: List[str]
) -> pd.DataFrame:
    if result.optimize_result is None or len(result.optimize_result.list) == 0:
        raise RuntimeError("No optimization results found. Optimization likely failed.")

    best = min(result.optimize_result.list, key=lambda r: float(r.fval) if r.fval is not None else float("inf"))
    if best.x is None:
        raise RuntimeError("Best result has no parameter vector (best.x is None).")

    x = np.asarray(best.x, dtype=float)
    if len(x) != len(parameter_ids):
        raise RuntimeError(
            f"Length mismatch: best.x has length {len(x)} but parameter_ids has length {len(parameter_ids)}."
        )

    df = pd.DataFrame({"parameterId": parameter_ids, "estimate": x})
    return df


def _save_sim_vs_data_head(
    petab_problem: petab.Problem,
    objective: Any,
    parameter_ids: List[str],
    x_best: np.ndarray,
) -> None:
    meas_df = _safe_read_tsv(DATASET_DIR / "measurements.tsv")

    base_required = {"observableId", "time", "measurement"}
    missing_base = base_required - set(meas_df.columns)
    if missing_base:
        raise RuntimeError(f"measurements.tsv is missing columns: {sorted(missing_base)}")

    cond_col = None
    for candidate in ["conditionId", "simulationConditionId"]:
        if candidate in meas_df.columns:
            cond_col = candidate
            break

    if cond_col is None:
        cond_series = pd.Series(["NA"] * len(meas_df), index=meas_df.index)
    else:
        cond_series = meas_df[cond_col].astype(str)

    head_df = meas_df.head(50).copy()
    head_df["conditionId"] = cond_series.head(50).values

    rr = getattr(objective, "roadrunner_instance", None)
    if rr is None:
        head_df["simulation"] = np.nan
        out_csv = PRED_DIR / "sim_vs_data_head.csv"
        head_df[["observableId", "conditionId", "time", "measurement", "simulation"]].to_csv(out_csv, index=False)
        return

    param_map = dict(zip(parameter_ids, map(float, x_best)))
    for pid, val in param_map.items():
        try:
            rr.setValue(pid, val)  
        except Exception:
            try:
                rr[pid] = val  
            except Exception:
                pass

    sim_values: List[float] = []
    for _, row in head_df.iterrows():
        t = float(row["time"])
        obs = str(row["observableId"])
        try:
            sim_res = rr.simulate(times=[t])  
            sim_val = np.nan

            if hasattr(sim_res, "colnames"):
                colnames = list(sim_res.colnames)
                arr = np.asarray(sim_res)
                if obs in colnames:
                    j = colnames.index(obs)
                    sim_val = float(arr[-1, j])
                else:
                    if "time" in colnames and len(colnames) == 2:
                        j = 1 if colnames[0] == "time" else 0
                        sim_val = float(arr[-1, j])
            sim_values.append(sim_val)
        except Exception:
            sim_values.append(np.nan)

    head_df["simulation"] = sim_values
    out_csv = PRED_DIR / "sim_vs_data_head.csv"
    head_df[["observableId", "conditionId", "time", "measurement", "simulation"]].to_csv(out_csv, index=False)


def _save_fit_plot(
    petab_problem: petab.Problem,
    objective: Any,
) -> None:
    rr = getattr(objective, "roadrunner_instance", None)
    if rr is None:
        raise RuntimeError("Objective does not expose roadrunner_instance; cannot plot.")

    out_png = PRED_DIR / "fit_plot.png"

    try:
        fig = rr.plot()  
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close("all")
        return
    except Exception:
        pass

    try:
        times = np.linspace(0, 50, 101).tolist()
        sim = rr.simulate(times=times)  
        plt.figure(figsize=(10, 6))
        if hasattr(sim, "colnames"):
            colnames = list(sim.colnames)
            arr = np.asarray(sim)
            time_idx = colnames.index("time") if "time" in colnames else 0
            t = arr[:, time_idx]
            for j, name in enumerate(colnames):
                if name == "time":
                    continue
                plt.plot(t, arr[:, j], label=name)
            plt.xlabel("time")
            plt.ylabel("value")
            plt.legend(loc="best", fontsize=8)
            plt.grid(True, alpha=0.3)
        else:
            plt.plot(np.asarray(sim))
            plt.grid(True, alpha=0.3)

        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close("all")
    except Exception as e:
        out_png.write_bytes(b"")  
        raise RuntimeError(f"Failed to generate fit_plot.png: {e}")


def main() -> None:
    _ensure_dirs()
    np.random.seed(SEED)

    _write_removed_measurements_head()

    if not PETAB_YAML.exists():
        raise FileNotFoundError(f"PEtab YAML not found at: {PETAB_YAML}")

    petab_problem = petab.Problem.from_yaml(str(PETAB_YAML))

    importer = pypesto.petab.PetabImporter(petab_problem, simulator_type="roadrunner")
    problem = importer.create_problem()

    solver_options = pypesto_rr.SolverOptions(
        relative_tolerance=RR_REL_TOL,
        absolute_tolerance=RR_ABS_TOL,
        maximum_num_steps=RR_MAX_STEPS,
    )
    problem.objective.solver_options = solver_options

    optimizer = optimize.ScipyOptimizer()
    engine = pypesto.engine.SingleCoreEngine()

    result = optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=N_STARTS,
        engine=engine,
    )

    parameter_ids: List[str] = []
    try:
        parameter_ids = list(petab_problem.x_ids)  
    except Exception:
        try:
            parameter_ids = list(problem.x_names)  
        except Exception:
            parameter_ids = [f"param_{i}" for i in range(problem.dim_full)]  

    best = min(result.optimize_result.list, key=lambda r: float(r.fval) if r.fval is not None else float("inf"))
    if best.x is None:
        raise RuntimeError("Optimization produced no best.x vector.")
    x_best = np.asarray(best.x, dtype=float)

    df_params = _extract_best_parameters(result, parameter_ids)
    df_params.to_csv(PRED_DIR / "estimated_parameters.tsv", sep="\t", index=False)

    summary: Dict[str, Any] = {
        "seed": SEED,
        "n_starts": N_STARTS,
        "optimizer": "ScipyOptimizer",
        "best_fval": float(best.fval) if best.fval is not None else None,
        "best_start_index": int(best.id) if hasattr(best, "id") and best.id is not None else None,
        "rr_solver_options": {
            "relative_tolerance": RR_REL_TOL,
            "absolute_tolerance": RR_ABS_TOL,
            "maximum_num_steps": RR_MAX_STEPS,
        },
        "success_flags": {
            "any_success": any(getattr(r, "exitflag", 1) == 0 for r in result.optimize_result.list),
        },
    }
    (PRED_DIR / "objective_summary.json").write_text(json.dumps(summary, indent=2))

    _save_sim_vs_data_head(
        petab_problem=petab_problem,
        objective=problem.objective,
        parameter_ids=parameter_ids,
        x_best=x_best,
    )

    _save_fit_plot(petab_problem=petab_problem, objective=problem.objective)

    print("Done. Wrote outputs to ./pred_results/.")


if __name__ == "__main__":
    main()