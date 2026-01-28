#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def load_data():
    print("📂 Loading data from ./benchmark/datasets/pyPESTO/ ...")
    data_folder = "./benchmark/datasets/pyPESTO/"

    measurements_path = os.path.join(
        data_folder, "measurementData_example_semiquantitative.tsv"
    )
    parameters_path = os.path.join(
        data_folder, "parameters_example_semiquantitative.tsv"
    )

    measurements_df = pd.read_csv(measurements_path, sep="\t")
    parameters_df = pd.read_csv(parameters_path, sep="\t")

    print(
        f"✅ Loaded {len(measurements_df)} measurements, "
        f"{len(parameters_df)} parameters"
    )
    return measurements_df, parameters_df


def simple_model(params, conditions):

    K1, K2, K3, K5 = params  # K1, K2 kept for completeness

    inhibitor_concs = []
    for condition in conditions:
        if "Inhibitor_" in condition:
            conc_str = condition.split("_", 1)[1]
            conc = float(conc_str)
            inhibitor_concs.append(conc)
        else:
            inhibitor_concs.append(0.0)

    inhibitor_concs = np.array(inhibitor_concs, dtype=float)
    return K3 / (1.0 + inhibitor_concs / K5)


def objective(params_opt, measurements_df):
    # Transform parameters: K1, K2 linear; K3, K5 on log10 scale
    K1, K2 = params_opt[:2]
    K3, K5 = 10 ** params_opt[2:4]
    params = [K1, K2, K3, K5]

    conditions = measurements_df["simulationConditionId"].values
    predicted = simple_model(params, conditions)
    measured = measurements_df["measurement"].values

    return np.sum((measured - predicted) ** 2)


def run_optimization(measurements_df, n_starts=3):
    print("🚀 Running multi-start optimization...")

    # Parameter bounds in optimization space
    bounds = [
        (-5.0, 5.0),                       # K1
        (-5.0, 5.0),                       # K2
        (np.log10(0.1), np.log10(1e5)),    # log10(K3)
        (np.log10(1e-5), np.log10(1e5)),   # log10(K5)
    ]

    results = []
    rng = np.random.default_rng(42)

    for i in range(n_starts):
        if i == 0:
            # Nominal initial guess (in optimization space)
            x0 = [0.04, 20.0, np.log10(4000.0), np.log10(0.1)]
        else:
            x0 = [rng.uniform(lb, ub) for lb, ub in bounds]

        result = minimize(
            objective,
            x0,
            args=(measurements_df,),
            method="L-BFGS-B",
            bounds=bounds,
        )
        if result.success:
            results.append(result)

    if not results:
        raise RuntimeError("All optimization starts failed.")

    results.sort(key=lambda x: x.fun)
    print(f"✅ Best objective: {results[0].fun:.6f}")
    return results


def generate_synthetic_data(measurements_df):
    print("🎲 Generating synthetic data...")

    true_params = [0.04, 20.0, 4000.0, 0.1]  # K1, K2, K3, K5

    conditions = measurements_df["simulationConditionId"].values
    synthetic_measurements = simple_model(true_params, conditions)

    # Add a small amount of Gaussian noise
    rng = np.random.default_rng(123)
    noise_scale = 0.1 * float(np.std(synthetic_measurements))
    noise = noise_scale * rng.normal(size=len(synthetic_measurements))

    synthetic_measurements_noisy = synthetic_measurements + noise

    synthetic_df = measurements_df.copy()
    synthetic_df["measurement"] = synthetic_measurements_noisy

    print(f"✅ Generated {len(synthetic_measurements_noisy)} synthetic measurements")
    return synthetic_df, true_params


def create_plots(original_results, synthetic_results, true_params):
    print("📈 Creating comparison plots...")
    os.makedirs("./pred_results", exist_ok=True)

    # Extract best parameters in linear space
    best_orig = original_results[0].x
    best_synth = synthetic_results[0].x

    orig_params = [
        best_orig[0],
        best_orig[1],
        10 ** best_orig[2],
        10 ** best_orig[3],
    ]
    synth_params = [
        best_synth[0],
        best_synth[1],
        10 ** best_synth[2],
        10 ** best_synth[3],
    ]

    param_names = ["K1", "K2", "K3", "K5"]
    x_pos = np.arange(len(param_names))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Bar plot of parameter estimates
    axes[0].bar(x_pos - 0.2, orig_params, 0.4, label="Original", alpha=0.7)
    axes[0].bar(x_pos + 0.2, synth_params, 0.4, label="Synthetic", alpha=0.7)
    axes[0].set_xlabel("Parameters")
    axes[0].set_ylabel("Estimated values")
    axes[0].set_title("Parameter estimates (linear scale)")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(param_names)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Objective values per start
    orig_objs = [r.fun for r in original_results]
    synth_objs = [r.fun for r in synthetic_results]

    axes[1].plot(orig_objs, "o-", label="Original data")
    axes[1].plot(synth_objs, "s-", label="Synthetic data")
    axes[1].set_xlabel("Optimization start index")
    axes[1].set_ylabel("Objective value")
    axes[1].set_title("Optimization results")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = "./pred_results/pyPESTO_synthetic_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Plot saved to {plot_path}")
    return orig_params, synth_params


def save_results(orig_params, synth_params, true_params, original_results, synthetic_results):
    print("💾 Saving results to ./pred_results/ ...")
    os.makedirs("./pred_results", exist_ok=True)

    param_names = ["K1", "K2", "K3", "K5"]

    # 1) Canonical file for SAB evaluation: ONLY param_id + est_value (synthetic fit)
    eval_df = pd.DataFrame(
        {
            "param_id": param_names,
            "est_value": synth_params,
        }
    )
    eval_path = "./pred_results/pyPESTO_synthetic_params.csv"
    eval_df.to_csv(eval_path, index=False)
    print(f"✅ Evaluation-ready params saved: {eval_path}")

    # 2) Detailed parameter comparison (for humans, not needed by eval script)
    detailed_df = pd.DataFrame(
        {
            "param_id": param_names,
            "original_estimate": orig_params,
            "synthetic_estimate": synth_params,
            "true_value": true_params,
        }
    )
    detailed_path = "./pred_results/pyPESTO_synthetic_parameter_estimates_detailed.csv"
    detailed_df.to_csv(detailed_path, index=False)

    # 3) Summary of objective values
    summary_df = pd.DataFrame(
        {
            "method": ["Original data", "Synthetic data"],
            "best_objective": [original_results[0].fun, synthetic_results[0].fun],
            "convergence_success": [
                bool(original_results[0].success),
                bool(synthetic_results[0].success),
            ],
        }
    )
    summary_path = "./pred_results/pyPESTO_synthetic_optimization_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(
        "✅ Detailed results saved:\n"
        f"   {detailed_path}\n"
        f"   {summary_path}"
    )


def main():
    print("🔬 Synthetic Data Parameter Estimation (pyPESTO example)")
    print("=" * 60)

    # Load data
    measurements_df, parameters_df = load_data()

    # Optimization on original data
    print("\n🎯 Optimization on original measurements")
    original_results = run_optimization(measurements_df, n_starts=3)

    # Generate synthetic data and optimize
    print("\n🎲 Optimization on synthetic measurements")
    synthetic_df, true_params = generate_synthetic_data(measurements_df)
    synthetic_results = run_optimization(synthetic_df, n_starts=3)

    # Create plots and save results
    orig_params, synth_params = create_plots(
        original_results, synthetic_results, true_params
    )
    save_results(orig_params, synth_params, true_params, original_results, synthetic_results)

    # Optional: print a brief numerical summary (no pass/fail logic here)
    print("\n" + "=" * 60)
    print("✅ Analysis completed.")
    print(f"📊 Best objective (original data): {original_results[0].fun:.6f}")
    print(f"🎲 Best objective (synthetic data): {synthetic_results[0].fun:.6f}")

    # Parameter recovery report (informative only, thresholding is in eval script)
    print("\n🔎 Parameter recovery (synthetic fit vs. true values):")
    for name, est, true in zip(["K1", "K2", "K3", "K5"], synth_params, true_params):
        rel_err = abs(est - true) / max(abs(true), 1e-8)
        print(f"   {name}: est={est:.6g}, true={true:.6g}, rel_error={rel_err:.3e}")


if __name__ == "__main__":
    main()
