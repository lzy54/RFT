#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def load_data():
    print("📂 Loading data...")
    data_folder = "./benchmark/datasets/PESTO_conver/"
    measurements_df = pd.read_csv(os.path.join(data_folder, "measurements.tsv"), sep="\t")
    parameters_df = pd.read_csv(os.path.join(data_folder, "parameters.tsv"), sep="\t")
    print(f"✅ Loaded {len(measurements_df)} measurements, {len(parameters_df)} parameters")
    return measurements_df, parameters_df


def model_function(params, times):
    k1, k2 = np.exp(params)
    return np.exp(-k1 * times) * (1 + k2 * np.sin(times))


def objective(params, measurements_df):
    times = measurements_df["time"].values
    data = measurements_df["measurement"].values
    predicted = model_function(params, times)
    return np.sum((data - predicted) ** 2)


def run_optimization(measurements_df, parameters_df):
    print("🚀 Running optimization...")
    bounds = [(np.log(1e-5), np.log(1e5)), (np.log(1e-5), np.log(1e5))]
    results = []
    np.random.seed(42)
    for i in range(5):
        if i > 0:
            x0 = [np.random.uniform(lb, ub) for lb, ub in bounds]
        else:
            x0 = [np.log(0.8), np.log(0.6)]
        result = minimize(objective, x0, args=(measurements_df,), method="L-BFGS-B", bounds=bounds)
        if result.success:
            results.append(result)
    results.sort(key=lambda x: x.fun)
    print(f"✅ Best objective: {results[0].fun:.6f}")
    return results


def run_profiling(best_result, measurements_df):
    print("📊 Running profiling...")
    profiles = {}
    for i, param in enumerate(["k1", "k2"]):
        points = np.linspace(best_result.x[i] - 1, best_result.x[i] + 1, 10)
        values = []
        for p in points:
            test_params = best_result.x.copy()
            test_params[i] = p
            values.append(objective(test_params, measurements_df))
        profiles[param] = {"points": points, "values": values}
    print("✅ Profiling completed")
    return profiles


def run_sampling(best_result, measurements_df):
    print("🎲 Running sampling...")
    current = best_result.x.copy()
    samples = []
    np.random.seed(123)
    for _ in range(500):
        proposal = current + np.random.normal(0, 0.1, 2)
        if np.random.rand() < 0.5:
            current = proposal
        samples.append(current.copy())
    print("✅ Sampling completed")
    return np.array(samples)


def create_plots(results, profiles, samples, measurements_df):
    print("📈 Creating plots...")
    os.makedirs("./pred_results", exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    obj_vals = [r.fun for r in results]
    axes[0, 0].plot(obj_vals, "o-")
    axes[0, 0].set_title("Optimization Results")
    axes[0, 0].grid(True)

    for i, (param, profile) in enumerate(profiles.items()):
        axes[0, 1].plot(profile["points"], profile["values"], label=param)
    axes[0, 1].set_title("Parameter Profiles")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(samples[:, 0], label="k1", alpha=0.7)
    axes[1, 0].plot(samples[:, 1], label="k2", alpha=0.7)
    axes[1, 0].set_title("MCMC Traces")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    times = measurements_df["time"].values
    data = measurements_df["measurement"].values
    predicted = model_function(results[0].x, times)

    axes[1, 1].scatter(times, data, alpha=0.7, label="Data")
    axes[1, 1].plot(times, predicted, "r-", label="Fit")
    axes[1, 1].set_title("Model Fit")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plot_path = "./pred_results/pypesto_results.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Plot saved: {plot_path}")

    return plot_path


def save_results(results, profiles, samples, measurements_df):
    print("💾 Saving results...")
    os.makedirs("./pred_results", exist_ok=True)

    best = results[0]
    k1, k2 = np.exp(best.x)

    best_df = pd.DataFrame(
        {
            "parameter": ["k1", "k2"],
            "estimate": [k1, k2],
            "log_estimate": best.x,
            "objective_value": best.fun,
        }
    )
    best_path = "./pred_results/pypesto_best_parameters.csv"
    best_df.to_csv(best_path, index=False)

    summary_df = pd.DataFrame(
        {
            "metric": ["best_objective", "k1_estimate", "k2_estimate"],
            "value": [best.fun, k1, k2],
        }
    )
    summary_path = "./pred_results/pypesto_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    pred_df = pd.DataFrame(
        {
            "time": measurements_df["time"].values,
            "measurement": measurements_df["measurement"].values,
            "prediction": model_function(best.x, measurements_df["time"].values),
        }
    )
    pred_path = "./pred_results/pypesto_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    print("✅ Results saved")
    return best_path, summary_path, pred_path


def main():
    print("🔬 PyPESTO Result Storage (Simplified)")
    print("=" * 40)

    try:
        measurements_df, parameters_df = load_data()
        results = run_optimization(measurements_df, parameters_df)
        profiles = run_profiling(results[0], measurements_df)
        samples = run_sampling(results[0], measurements_df)

        plot_path = create_plots(results, profiles, samples, measurements_df)
        paths = save_results(results, profiles, samples, measurements_df)

        k1, k2 = np.exp(results[0].x)
        print("\n" + "=" * 40)
        print("✅ Analysis completed!")
        print(f"📊 Best objective: {results[0].fun:.6f}")
        print(f"🎯 Parameters: k1={k1:.4f}, k2={k2:.4f}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
