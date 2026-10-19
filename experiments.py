"""Command-line experiment runner (no Streamlit needed).

Usage:
    python experiments.py                     # default settings
    python experiments.py --seeds 10 --deliveries 15 --dataset-seed 42

It writes real results (nothing is hard-coded) to the reports/ folder:
    results_runs.csv       one row per (method, seed)
    results_summary.csv    mean and standard deviation per method
    results_improvement.csv improvement % versus the two baselines
    results_small_instance.csv ACO versus brute-force optimum on a small instance

METHODOLOGY NOTE (avoiding leakage): the objective weights (0.5 / 0.5) and the ACO
parameters are the fixed defaults chosen BEFORE any result was looked at. They are not
tuned on the evaluation dataset. A separate "tuning dataset seed" is only used by the
optional --sensitivity run, never for the headline comparison.
"""

import argparse
from pathlib import Path

import pandas as pd

from src.aco import ACOParams
from src.baselines import ACO_HYBRID, BRUTE_FORCE, brute_force
from src.data_generator import generate_dataset
from src.data_validation import clean_dataframe
from src.fuzzy_system import FuzzyPenaltySystem
from src.metrics import compare_methods, route_metrics, run_repeated
from src.route_evaluator import ObjectiveConfig, RoutingProblem

REPORT_DIR = Path(__file__).parent / "reports"


def build_problem(n_deliveries, dataset_seed, speed):
    df = clean_dataframe(generate_dataset(n_deliveries, dataset_seed, speed))
    return RoutingProblem.from_dataframe(df, speed)


SENSITIVITY_DATASET_SEED = 7   # different from the evaluation dataset (leakage avoidance)


def run_weight_sensitivity(args, fuzzy, params):
    """Show how the distance weight w_d moves the trade-off between distance and lateness.

    Uses a separate dataset so that nothing is tuned on the evaluation dataset. Objective
    values are NOT compared across weights (J changes meaning); raw metrics are reported.
    """
    problem = build_problem(args.deliveries, SENSITIVITY_DATASET_SEED, args.speed)
    rows = []
    for w_distance in (0.2, 0.35, 0.5, 0.65, 0.8, 1.0):
        config = ObjectiveConfig(w_distance, round(1.0 - w_distance, 10)).validate()
        table, _ = run_repeated(problem, ACO_HYBRID, params, fuzzy, config, range(3))
        rows.append({"w_distance": w_distance, "w_penalty": round(1.0 - w_distance, 2),
                     "total_distance_mean": table["total_distance"].mean(),
                     "total_lateness_mean": table["total_lateness"].mean(),
                     "n_late_mean": table["n_late"].mean(),
                     "total_penalty_mean": table["total_penalty"].mean()})
    result = pd.DataFrame(rows)
    result.to_csv(REPORT_DIR / "results_weight_sensitivity.csv", index=False)
    print(f"\n=== Weight sensitivity (separate dataset, seed {SENSITIVITY_DATASET_SEED}, 3 ACO seeds) ===")
    print(result.round(3).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Run the QuickRoute experiments.")
    parser.add_argument("--deliveries", type=int, default=15)
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, default=10, help="number of ACO random seeds")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--w-distance", type=float, default=0.5)
    parser.add_argument("--ants", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--sensitivity", action="store_true",
                        help="also run the weight-sensitivity study on a SEPARATE dataset (seed 7)")
    args = parser.parse_args()

    REPORT_DIR.mkdir(exist_ok=True)
    fuzzy = FuzzyPenaltySystem()
    config = ObjectiveConfig(args.w_distance, 1.0 - args.w_distance).validate()
    params = ACOParams(n_ants=args.ants, n_iterations=args.iterations).validate()
    seeds = list(range(args.seeds))

    # ---- main comparison -----------------------------------------------------------
    problem = build_problem(args.deliveries, args.dataset_seed, args.speed)
    outcome = compare_methods(problem, params, fuzzy, config, seeds)
    outcome["runs"].to_csv(REPORT_DIR / "results_runs.csv", index=False)
    outcome["summary"].to_csv(REPORT_DIR / "results_summary.csv", index=False)
    outcome["improvements"].to_csv(REPORT_DIR / "results_improvement.csv", index=False)

    pd.set_option("display.width", 220)
    columns = ["method", "n_runs", "total_distance_mean", "total_distance_std",
               "total_lateness_mean", "n_late_mean", "total_penalty_mean",
               "objective_mean", "objective_std", "runtime_seconds_mean"]
    print(f"\n=== Main comparison: {args.deliveries} deliveries, dataset seed {args.dataset_seed}, "
          f"{args.seeds} ACO seeds ===")
    print(outcome["summary"][columns].round(3).to_string(index=False))
    print("\nImprovement (%), positive = better than the baseline:")
    print(outcome["improvements"].round(2).to_string(index=False))

    # ---- small instance: compare with the brute-force optimum -------------------------
    small = build_problem(8, args.dataset_seed + 1, args.speed)
    optimum = brute_force(small, fuzzy, config)
    optimum_row = route_metrics(small, optimum, config, fuzzy)
    table, _ = run_repeated(small, ACO_HYBRID, params, fuzzy, config, seeds)
    hits = int((table["objective"] <= optimum_row["objective"] + 1e-9).sum())
    small_summary = pd.DataFrame([
        {"method": BRUTE_FORCE, "objective_mean": optimum_row["objective"], "objective_std": 0.0,
         "runs": 1, "runs_matching_optimum": ""},
        {"method": ACO_HYBRID, "objective_mean": table["objective"].mean(),
         "objective_std": table["objective"].std(ddof=0), "runs": len(table),
         "runs_matching_optimum": hits},
    ])
    small_summary.to_csv(REPORT_DIR / "results_small_instance.csv", index=False)
    print("\n=== Small instance (8 deliveries): hybrid ACO versus brute-force optimum ===")
    print(small_summary.round(4).to_string(index=False))
    if args.sensitivity:
        run_weight_sensitivity(args, fuzzy, params)
    print(f"\nFiles written to {REPORT_DIR}")


if __name__ == "__main__":
    main()
