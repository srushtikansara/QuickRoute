"""Evaluation metrics, improvement formulas and repeated-run experiments.

IMPROVEMENT FORMULAS (percentages)
  For a metric where LOWER is better (distance, lateness, penalty, objective):
        improvement % = (baseline - method) / baseline * 100       (positive = better)
  Baselines: Nearest Neighbor and ACO Distance-Only.
  If the baseline value is 0 the improvement is reported as NaN (undefined).
"""

import numpy as np
import pandas as pd

from .baselines import (ACO_DISTANCE, ACO_HYBRID, BRUTE_FORCE, MAX_BRUTE_FORCE_DELIVERIES,
                        NEAREST_NEIGHBOR, SolverOutput, solve, with_seed)
from .route_evaluator import evaluate_route

# metric column -> (label, lower_is_better)
METRIC_INFO = {
    "total_distance": ("Total distance (su)", True),
    "total_travel_time": ("Travel time (min)", True),
    "total_service_time": ("Service time (min)", True),
    "total_lateness": ("Total lateness (min)", True),
    "n_late": ("Late deliveries (count)", True),
    "pct_late": ("Late deliveries (%)", True),
    "priority_weighted_lateness": ("Priority-weighted lateness (priority x min)", True),
    "total_penalty": ("Fuzzy penalty (sum, 0-100 each)", True),
    "objective": ("Hybrid objective J", True),
    "runtime_seconds": ("Execution time (s)", True),
}
METRIC_COLUMNS = list(METRIC_INFO)


def route_metrics(problem, solver_output, config, fuzzy_system):
    """Evaluate a solver's route with the SAME hybrid objective and return a flat dict."""
    result = evaluate_route(problem, solver_output.delivery_order, config, fuzzy_system)
    row = {
        "method": solver_output.method,
        "total_distance": result.total_distance,
        "total_travel_time": result.total_travel_time,
        "total_service_time": result.total_service_time,
        "total_lateness": result.total_lateness,
        "n_late": result.n_late,
        "pct_late": result.pct_late,
        "priority_weighted_lateness": result.priority_weighted_lateness,
        "total_penalty": result.total_penalty,
        "normalized_distance": result.normalized_distance,
        "normalized_penalty": result.normalized_penalty,
        "objective": result.objective,
        "runtime_seconds": solver_output.runtime_seconds,
    }
    return row


def improvement_percent(baseline_value, method_value, lower_is_better=True):
    """(baseline - method) / baseline * 100 for lower-is-better metrics (positive = better)."""
    if baseline_value is None or not np.isfinite(baseline_value) or abs(baseline_value) < 1e-12:
        return float("nan")
    change = (baseline_value - method_value) if lower_is_better else (method_value - baseline_value)
    return 100.0 * change / abs(baseline_value)


def run_repeated(problem, method, params, fuzzy_system, config, seeds):
    """Run one method for several seeds. Returns (metrics DataFrame, list of SolverOutput)."""
    rows, outputs = [], []
    for seed in seeds:
        output = solve(method, problem, with_seed(params, seed), fuzzy_system, config)
        row = route_metrics(problem, output, config, fuzzy_system)
        row["seed"] = int(seed)
        rows.append(row)
        outputs.append(output)
    return pd.DataFrame(rows), outputs


def compare_methods(problem, params, fuzzy_system, config, seeds, progress_callback=None,
                    include_brute_force=False):
    """Compare Nearest Neighbor (deterministic) with the two ACO variants (repeated runs).

    Returns a dict with
      runs     : one row per (method, seed)            -> the raw repeated-run table
      summary  : mean and standard deviation per method -> the comparison table
      outputs  : {method: [SolverOutput, ...]}          -> routes and ACO histories
      improvements : improvement % of each method versus the two baselines
    """
    seeds = list(seeds)
    if not seeds:
        raise ValueError("At least one random seed is required.")

    all_rows, outputs = [], {}

    nn_output = solve(NEAREST_NEIGHBOR, problem, params, fuzzy_system, config)
    row = route_metrics(problem, nn_output, config, fuzzy_system)
    row["seed"] = -1                      # deterministic, no seed
    all_rows.append(row)
    outputs[NEAREST_NEIGHBOR] = [nn_output]

    if include_brute_force and problem.n_deliveries <= MAX_BRUTE_FORCE_DELIVERIES:
        bf_output = solve(BRUTE_FORCE, problem, params, fuzzy_system, config)
        row = route_metrics(problem, bf_output, config, fuzzy_system)
        row["seed"] = -1
        all_rows.append(row)
        outputs[BRUTE_FORCE] = [bf_output]

    total_steps = 2 * len(seeds)
    step = 0
    for method in (ACO_DISTANCE, ACO_HYBRID):
        table, outs = run_repeated(problem, method, params, fuzzy_system, config, seeds)
        all_rows.extend(table.to_dict("records"))
        outputs[method] = outs
        step += len(seeds)
        if progress_callback:
            progress_callback(step, total_steps)

    runs = pd.DataFrame(all_rows)
    summary = summarize_runs(runs)
    return {
        "runs": runs,
        "summary": summary,
        "outputs": outputs,
        "improvements": improvement_table(summary),
        "n_seeds": len(seeds),
    }


def summarize_runs(runs):
    """Mean and standard deviation of every metric for each method."""
    order = [m for m in (NEAREST_NEIGHBOR, ACO_DISTANCE, ACO_HYBRID, BRUTE_FORCE)
             if m in set(runs["method"])]
    rows = []
    for method in order:
        group = runs[runs["method"] == method]
        row = {"method": method, "n_runs": len(group)}
        for column in METRIC_COLUMNS:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=0)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def improvement_table(summary):
    """Improvement (%) of every method over Nearest Neighbor and over ACO Distance-Only.

    Uses the mean metric values. Positive = the method is better than the baseline.
    """
    means = summary.set_index("method")
    rows = []
    for baseline in (NEAREST_NEIGHBOR, ACO_DISTANCE):
        if baseline not in means.index:
            continue
        for method in means.index:
            if method == baseline:
                continue
            row = {"method": method, "baseline": baseline}
            for column in ["total_distance", "total_lateness", "n_late",
                           "total_penalty", "objective"]:
                row[f"{column}_improvement_pct"] = improvement_percent(
                    means.loc[baseline, f"{column}_mean"], means.loc[method, f"{column}_mean"],
                    METRIC_INFO[column][1])
            rows.append(row)
    return pd.DataFrame(rows)


def delivery_level_table(problem, result):
    """One row per delivery in route order: arrival, deadline, lateness and fuzzy penalty."""
    rows = []
    for position, node in enumerate(result.delivery_order, start=1):
        rows.append({
            "stop": position,
            "delivery_id": problem.ids[node],
            "location_name": problem.names[node],
            "priority": int(problem.priorities[node]),
            "arrival_minutes": round(result.arrivals[node], 2),
            "deadline_minutes": float(problem.deadlines[node]),
            "arrival_minus_deadline": round(result.signed_lateness[node], 2),
            "lateness_minutes": round(result.lateness[node], 2),
            "late": result.lateness[node] > 1e-9,
            "fuzzy_penalty": round(result.penalties[node], 2),
        })
    return pd.DataFrame(rows)


def best_method_by_objective(summary):
    return summary.loc[summary["objective_mean"].idxmin(), "method"]


def format_summary_table(summary):
    """Human-readable table: 'mean +- std' for every metric (one row per method)."""
    rows = []
    for _, row in summary.iterrows():
        shown = {"Method": row["method"], "Runs": int(row["n_runs"])}
        for column, (label, _) in METRIC_INFO.items():
            decimals = 4 if column in ("objective", "runtime_seconds") else 2
            mean, std = row[f"{column}_mean"], row[f"{column}_std"]
            shown[label] = f"{mean:.{decimals}f} +- {std:.{decimals}f}" if row["n_runs"] > 1 else f"{mean:.{decimals}f}"
        rows.append(shown)
    return pd.DataFrame(rows)
