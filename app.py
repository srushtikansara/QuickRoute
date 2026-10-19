"""QuickRoute - Streamlit dashboard.

Run with:  streamlit run app.py
All algorithm logic lives in the src/ package; this file only builds the user interface.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.aco import ACOParams
from src.baselines import (ACO_DISTANCE, ACO_HYBRID, BRUTE_FORCE, MAX_BRUTE_FORCE_DELIVERIES,
                           NEAREST_NEIGHBOR, solve)
from src.data_generator import generate_dataset
from src.data_validation import (DataValidationError, clean_dataframe, load_and_validate_csv,
                                 validate_dataframe)
from src.distance import DISTANCE_UNIT_LABEL
from src.fuzzy_system import (LATENESS_RANGE, PRIORITY_RANGE, RULES, FuzzyPenaltySystem,
                              rule_text)
from src.metrics import (METRIC_INFO, compare_methods, delivery_level_table, format_summary_table,
                         route_metrics)
from src.route_evaluator import (ObjectiveConfig, ParameterError, RoutingProblem, evaluate_route)
from src.visualization import (comparison_bars_figure, convergence_figure, fuzzy_output_figure,
                               locations_figure, membership_figure, route_figure)

st.set_page_config(page_title="QuickRoute", page_icon="🚚", layout="wide")

SAMPLE_CSV = Path(__file__).parent / "data" / "sample_deliveries.csv"
PAGES = ["1. Project Overview", "2. Dataset Management", "3. Route Optimization",
         "4. Fuzzy Logic Explorer", "5. Algorithm Comparison", "6. Results and Export"]


# ------------------------------------------------------------------------------------
# Shared state helpers
# ------------------------------------------------------------------------------------
@st.cache_resource
def get_fuzzy_system():
    """One fuzzy system for the whole session (it caches its own results)."""
    return FuzzyPenaltySystem()


def init_state():
    if "df" not in st.session_state:
        st.session_state["df"] = load_and_validate_csv(SAMPLE_CSV)
        st.session_state["data_source"] = "Default sample dataset (data/sample_deliveries.csv)"
    st.session_state.setdefault("last_run", None)
    st.session_state.setdefault("comparison", None)


def set_dataset(df, source):
    """Store a new dataset and forget results computed on the old one."""
    st.session_state["df"] = df
    st.session_state["data_source"] = source
    st.session_state["last_run"] = None
    st.session_state["comparison"] = None


def sidebar_settings():
    """Settings used by pages 3, 5 and 6. Always rendered so the values persist."""
    st.sidebar.header("Optimization settings")
    st.sidebar.caption("Used by the Route Optimization and Algorithm Comparison pages.")
    speed = st.sidebar.number_input("Vehicle speed (distance units / minute)", 0.1, 20.0, 1.0, 0.1,
                                    key="speed",
                                    help="Simulated speed. Distances are simulated units, not km.")
    w_distance = st.sidebar.slider("Distance weight w_d", 0.0, 1.0, 0.5, 0.05, key="w_distance",
                                   help="J = w_d * normalised distance + w_p * normalised fuzzy penalty")
    st.sidebar.write(f"Penalty weight w_p = 1 - w_d = **{1 - w_distance:.2f}**")
    with st.sidebar.expander("ACO parameters", expanded=False):
        n_ants = st.number_input("Number of ants", 1, 500, 20, 1, key="n_ants")
        n_iterations = st.number_input("Number of iterations", 1, 2000, 100, 10, key="n_iterations")
        alpha = st.number_input("alpha (pheromone influence)", 0.0, 10.0, 1.0, 0.1, key="alpha")
        beta = st.number_input("beta (distance-heuristic influence)", 0.0, 10.0, 3.0, 0.1, key="beta")
        rho = st.number_input("Evaporation rate rho", 0.01, 1.0, 0.3, 0.01, key="rho")
        tau0 = st.number_input("Initial pheromone", 0.001, 100.0, 1.0, 0.1, key="tau0")
        seed = st.number_input("Random seed", 0, 1_000_000, 0, 1, key="seed")
    return {
        "speed": float(speed),
        "config": ObjectiveConfig(float(w_distance), float(round(1.0 - w_distance, 10))),
        "params": ACOParams(n_ants=int(n_ants), n_iterations=int(n_iterations), alpha=float(alpha),
                            beta=float(beta), evaporation_rate=float(rho),
                            initial_pheromone=float(tau0), seed=int(seed)),
    }


def build_problem(settings):
    return RoutingProblem.from_dataframe(st.session_state["df"], settings["speed"])


# ------------------------------------------------------------------------------------
# Page 1 - overview
# ------------------------------------------------------------------------------------
def page_overview():
    st.title("🚚 QuickRoute")
    st.subheader("Intelligent Delivery Route Optimization Using Ant Colony Optimization and Fuzzy Logic")
    st.write(
        "QuickRoute plans the delivery order of **one vehicle** that starts at a depot, visits every "
        "customer exactly once and returns to the depot. It combines two Soft Computing techniques: "
        "**Ant Colony Optimization (ACO)** searches for short routes, and a **Mamdani Fuzzy Logic** "
        "system turns delivery priority and deadline tightness into a lateness penalty. "
        "This is an **academic simulation**: coordinates are synthetic and there is no GPS, live traffic "
        "or mapping API.")

    st.markdown("### Objectives")
    st.markdown(
        "- Generate or load deliveries with priorities and deadlines, and compute distances.\n"
        "- Optimise the delivery order with ACO implemented from scratch.\n"
        "- Estimate arrival times and delays, and convert them to penalties with Fuzzy Logic.\n"
        "- Minimise a combined objective that balances distance and weighted delays.\n"
        "- Compare against Nearest Neighbor and distance-only ACO (and brute force on tiny instances).")

    left, right = st.columns(2)
    with left:
        st.markdown("### Ant Colony Optimization in short")
        st.write(
            "Artificial ants build routes one step at a time. At location *i* an ant picks an unvisited "
            "location *j* with probability proportional to **τ(i,j)^α · η(i,j)^β**, where τ is the "
            "pheromone level and η = 1/distance. After each iteration pheromone evaporates "
            "(τ ← (1−ρ)τ) and better routes deposit more pheromone (Δτ = Q / cost), so good "
            "edges become more attractive over time.")
    with right:
        st.markdown("### Fuzzy Logic in short")
        st.write(
            "A crisp rule such as 'late = bad' ignores how important and how late a delivery is. The "
            "fuzzy system fuzzifies **priority** (Low/Medium/High) and **deadline tightness** "
            "(Comfortable/Tight/Critical), applies 9 IF–THEN rules, and defuzzifies with the "
            "centroid method into a **penalty from 0 to 100**. The same delay costs more when priority is higher.")

    st.markdown("### Workflow")
    st.graphviz_chart("""
    digraph G {
      rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontsize=11];
      data  [label="Delivery data\\n(synthetic / CSV)"];
      dist  [label="Distance matrix"];
      aco   [label="ACO: ants build routes\\n(pheromone + 1/distance)"];
      sim   [label="Route simulation\\narrival times, lateness"];
      fuzzy [label="Fuzzy system\\npriority + tightness -> penalty"];
      obj   [label="Objective J\\n= w_d*D + w_p*P"];
      upd   [label="Pheromone update\\nevaporate + deposit"];
      out   [label="Best route\\n+ metrics + export"];
      data -> dist -> aco -> sim -> fuzzy -> obj -> upd -> aco;
      upd -> out [label=" after last iteration", fontsize=9];
    }
    """)

    st.markdown("### Assumptions and limitations")
    st.markdown(
        f"- Distances are Euclidean and measured in **{DISTANCE_UNIT_LABEL}**; speed is in units per minute.\n"
        "- **Single vehicle, closed tour** (depot → all customers → depot); no capacity limits.\n"
        "- The vehicle leaves the depot at time 0; arrival time is compared with the deadline "
        "(waiting for early arrivals is not modelled).\n"
        "- ACO is a heuristic: it does **not** guarantee the globally optimal route.\n"
        "- Penalty values depend on the membership functions and rules documented on page 4.")


# ------------------------------------------------------------------------------------
# Page 2 - dataset management
# ------------------------------------------------------------------------------------
def page_dataset():
    st.header("Dataset Management")
    st.caption(f"Current data: {st.session_state['data_source']}")

    with st.expander("Generate a synthetic dataset", expanded=False):
        c1, c2, c3 = st.columns(3)
        n_deliveries = c1.slider("Number of deliveries", 3, 30, 15, key="gen_n")
        data_seed = c2.number_input("Random seed", 0, 1_000_000, 42, 1, key="gen_seed")
        gen_speed = c3.number_input("Speed used to size deadlines", 0.1, 20.0, 1.0, 0.1, key="gen_speed")
        if st.button("Generate dataset"):
            df = clean_dataframe(generate_dataset(int(n_deliveries), int(data_seed), float(gen_speed)))
            set_dataset(df, f"Synthetic: {n_deliveries} deliveries, seed {data_seed}")
            st.success("New dataset generated. Earlier results were cleared.")

    with st.expander("Upload a CSV file", expanded=False):
        st.write("Required columns: delivery_id, location_name, x_coordinate, y_coordinate, priority, "
                 "service_time_minutes, delivery_deadline_minutes, is_depot (package_size is optional). "
                 "Exactly one row must have is_depot = True.")
        uploaded = st.file_uploader("CSV file", type=["csv"])
        if st.button("Validate and load uploaded file"):
            if uploaded is None:
                st.warning("Please choose a CSV file first.")
            else:
                try:
                    df = load_and_validate_csv(uploaded)
                    set_dataset(df, f"Uploaded file: {uploaded.name}")
                    st.success(f"Loaded {len(df) - 1} deliveries from {uploaded.name}.")
                except DataValidationError as error:
                    st.error(str(error))

    if st.button("Reload default sample dataset"):
        set_dataset(load_and_validate_csv(SAMPLE_CSV), "Default sample dataset (data/sample_deliveries.csv)")
        st.success("Default dataset restored.")

    df = st.session_state["df"]
    c1, c2 = st.columns([3, 1])
    with c1:
        st.subheader("Delivery records")
        st.dataframe(df, width="stretch", hide_index=True)
    with c2:
        st.download_button("⬇ Download dataset as CSV", df.to_csv(index=False).encode("utf-8"),
                           file_name="quickroute_deliveries.csv", mime="text/csv")
        if st.button("Validate current data"):
            problems = validate_dataframe(df)
            if problems:
                st.error("\n".join(problems))
            else:
                st.success("Data is valid: no missing values, duplicate IDs, invalid coordinates or priorities.")
        st.metric("Deliveries", len(df) - 1)

    st.subheader("Locations")
    st.plotly_chart(locations_figure(df), width="stretch")
    st.caption(f"Coordinates are in {DISTANCE_UNIT_LABEL}. Marker colour shows priority (darker = more urgent).")


# ------------------------------------------------------------------------------------
# Page 3 - route optimisation
# ------------------------------------------------------------------------------------
def show_metrics_row(result):
    c = st.columns(5)
    c[0].metric("Distance (su)", f"{result.total_distance:.1f}")
    c[1].metric("Total lateness (min)", f"{result.total_lateness:.1f}")
    c[2].metric("Late deliveries", f"{result.n_late} ({result.pct_late:.0f}%)")
    c[3].metric("Fuzzy penalty (sum)", f"{result.total_penalty:.1f}")
    c[4].metric("Objective J", f"{result.objective:.4f}")


def page_optimization(settings):
    st.header("Route Optimization")
    st.info("Assumption: one vehicle, closed tour (depot → every customer once → depot). "
            "Change the weights, speed and ACO parameters in the sidebar.")
    problem = build_problem(settings)
    config, params = settings["config"], settings["params"]
    fuzzy = get_fuzzy_system()

    methods = [NEAREST_NEIGHBOR, ACO_DISTANCE, ACO_HYBRID]
    if problem.n_deliveries <= MAX_BRUTE_FORCE_DELIVERIES:
        methods.append(BRUTE_FORCE)
    else:
        st.caption(f"Brute force is only offered for up to {MAX_BRUTE_FORCE_DELIVERIES} deliveries "
                   f"(this dataset has {problem.n_deliveries}).")
    method = st.selectbox("Algorithm", methods, index=2)

    if st.button("▶ Run optimization", type="primary"):
        try:
            config.validate()
            params.validate()
        except ParameterError as error:
            st.error(f"Invalid settings: {error}")
        else:
            progress = st.progress(0.0, text="Starting...")

            def on_progress(iteration, total, best_cost):
                progress.progress(iteration / total, text=f"Iteration {iteration}/{total} - best cost {best_cost:.4f}")

            with st.spinner("Optimizing..."):
                output = solve(method, problem, params, fuzzy, config, on_progress)
            progress.empty()
            st.session_state["last_run"] = {
                "method": method, "output": output, "problem": problem, "config": config,
                "params": params, "dataset": st.session_state["data_source"],
            }

    run = st.session_state["last_run"]
    if run is None:
        st.write("Choose an algorithm and press **Run optimization**.")
        return
    if run["problem"].n_deliveries != problem.n_deliveries or run["problem"].speed != problem.speed:
        st.warning("Settings changed since the last run; the result below is from the earlier settings.")

    result = evaluate_route(run["problem"], run["output"].delivery_order, run["config"], fuzzy)
    st.subheader(f"Result: {run['method']}")
    show_metrics_row(result)
    names = [run["problem"].names[n] for n in result.full_route]
    st.markdown("**Best delivery order:** " + " → ".join(names))
    st.caption(f"Execution time {run['output'].runtime_seconds:.3f} s. This is a single run "
               f"(seed {run['params'].seed}); stochastic methods vary between seeds - see the Comparison page.")
    st.plotly_chart(route_figure(run["problem"], result, f"{run['method']} route"), width="stretch")
    if run["output"].history is not None:
        label = "Best objective J so far" if run["method"] == ACO_HYBRID else \
            "Best-so-far route evaluated with hybrid objective J"
        st.pyplot(convergence_figure({run["method"]: [run["output"].history]}, label))


# ------------------------------------------------------------------------------------
# Page 4 - fuzzy explorer
# ------------------------------------------------------------------------------------
PRESETS = {"Custom (use the slider)": None, "Comfortable (45 min early)": -45.0,
           "Tight (exactly on the deadline)": 0.0, "Critical (60 min late)": 60.0}


def apply_preset():
    value = PRESETS[st.session_state["fz_preset"]]
    if value is not None:
        st.session_state["fz_late"] = value


def page_fuzzy():
    st.header("Fuzzy Logic Explorer")
    fuzzy = get_fuzzy_system()
    st.write("Inputs: **priority** and **lateness = arrival time − deadline** (minutes; "
             "**positive means the deadline was missed**). Output: a **penalty from 0 to 100**.")
    st.pyplot(membership_figure())

    st.subheader("IF–THEN rule base (9 rules, one for every input combination)")
    rules = pd.DataFrame([{"Rule": f"R{i}", "Definition": rule_text(p, t)}
                          for i, ((p, t), _) in enumerate(RULES.items(), start=1)])
    st.dataframe(rules, width="stretch", hide_index=True)

    st.subheader("Try it")
    c1, c2 = st.columns(2)
    priority = c1.slider("Delivery priority", float(PRIORITY_RANGE[0]), float(PRIORITY_RANGE[1]), 4.0, 0.1)
    st.session_state.setdefault("fz_late", 10.0)
    c2.selectbox("Deadline tightness preset", list(PRESETS), key="fz_preset", on_change=apply_preset)
    lateness = c2.slider("Lateness = arrival − deadline (minutes)", float(LATENESS_RANGE[0]),
                         float(LATENESS_RANGE[1]), step=1.0, key="fz_late")

    result = fuzzy.infer(priority, lateness)
    m1, m2, m3 = st.columns(3)
    m1.markdown("**Priority memberships**")
    m1.dataframe(pd.DataFrame({"term": list(result.priority_memberships),
                               "degree": [round(v, 3) for v in result.priority_memberships.values()]}),
                 hide_index=True, width="stretch")
    m2.markdown("**Deadline-tightness memberships**")
    m2.dataframe(pd.DataFrame({"term": list(result.tightness_memberships),
                               "degree": [round(v, 3) for v in result.tightness_memberships.values()]}),
                 hide_index=True, width="stretch")
    m3.metric("Defuzzified penalty (centroid)", f"{result.penalty:.1f} / 100")

    fired = [{"Rule": rule_text(*rule), "Firing strength": round(strength, 3)}
             for rule, strength in sorted(result.rule_strengths.items(), key=lambda kv: -kv[1])
             if strength > 0]
    st.markdown("**Activated rules**")
    st.dataframe(pd.DataFrame(fired), width="stretch", hide_index=True)
    st.pyplot(fuzzy_output_figure(result))

    st.markdown("**Why this penalty?**")
    for line in fuzzy.explain(result):
        st.write("• " + line)

    st.subheader("How priority changes the penalty")
    import matplotlib.pyplot as plt
    grid = np.linspace(LATENESS_RANGE[0], LATENESS_RANGE[1], 91)
    figure, axis = plt.subplots(figsize=(7, 3.4))
    for level in (1, 3, 5):
        axis.plot(grid, [fuzzy.penalty(level, x) for x in grid], label=f"priority {level}", linewidth=2)
    axis.set_xlabel("Arrival − deadline (min)   [> 0 = late]")
    axis.set_ylabel("Penalty (0-100)")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    st.pyplot(figure)
    st.caption("The penalty rises with lateness, and rises faster for higher-priority deliveries. "
               "Note: even an on-time delivery has a small floor (~11) because the centroid of the 'Low' "
               "shape is not 0; this adds the same constant to every route and does not change the ranking.")


# ------------------------------------------------------------------------------------
# Page 5 - comparison
# ------------------------------------------------------------------------------------
def page_comparison(settings):
    st.header("Algorithm Comparison")
    st.write("All methods use the same dataset, the same closed-tour assumptions and are **scored with "
             "the same hybrid objective J**. Nearest Neighbor is deterministic (1 run). The two ACO "
             "variants are run once per random seed; tables report **mean ± standard deviation over "
             "those runs**.")
    problem = build_problem(settings)
    config, params = settings["config"], settings["params"]
    fuzzy = get_fuzzy_system()

    c1, c2 = st.columns(2)
    n_seeds = c1.slider("Number of random seeds (repeated runs)", 1, 30, 5)
    include_bf = c2.checkbox(f"Add brute-force optimum (only if ≤ {MAX_BRUTE_FORCE_DELIVERIES} deliveries)",
                             value=False, disabled=problem.n_deliveries > MAX_BRUTE_FORCE_DELIVERIES)

    if st.button("▶ Run comparison", type="primary"):
        try:
            config.validate()
            params.validate()
        except ParameterError as error:
            st.error(f"Invalid settings: {error}")
        else:
            progress = st.progress(0.0, text="Running repeated experiments...")
            with st.spinner("Running ACO for every seed (this can take a while)..."):
                outcome = compare_methods(
                    problem, params, fuzzy, config, range(int(n_seeds)),
                    progress_callback=lambda done, total: progress.progress(done / total),
                    include_brute_force=include_bf)
            progress.empty()
            outcome.update({"problem": problem, "config": config, "params": params,
                            "dataset": st.session_state["data_source"]})
            st.session_state["comparison"] = outcome

    outcome = st.session_state["comparison"]
    if outcome is None:
        st.write("Press **Run comparison** to generate results.")
        return

    summary = outcome["summary"]
    st.subheader(f"Comparison table (n = {outcome['n_seeds']} seeds for ACO methods)")
    st.dataframe(format_summary_table(summary), width="stretch", hide_index=True)
    st.caption("Lower is better for every column. 'mean +- std' appears only for methods with more than one run.")

    st.subheader("Improvement (%)")
    st.caption("improvement % = (baseline − method) / baseline × 100, computed from mean values; "
               "positive = the method is better than the baseline, negative = worse. "
               "All metrics shown are lower-is-better.")
    improvements = outcome["improvements"].rename(columns=lambda c: c.replace("_improvement_pct", " (%)"))
    st.dataframe(improvements.round(2), width="stretch", hide_index=True)

    st.subheader("Distance and delay")
    st.pyplot(comparison_bars_figure(summary))
    runs = outcome["runs"]
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(7, 3.6))
    for method, group in runs.groupby("method"):
        axis.scatter(group["total_distance"], group["total_lateness"], label=method, s=45, alpha=0.8)
    axis.set_xlabel("Route distance (su)")
    axis.set_ylabel("Total lateness (min)")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    axis.set_title("Each point = one run (distance vs delay trade-off)")
    figure.tight_layout()
    st.pyplot(figure)

    st.subheader("Optimization progress")
    histories = {m: [o.history for o in outs] for m, outs in outcome["outputs"].items()
                 if outs and outs[0].history is not None}
    st.pyplot(convergence_figure(histories, "Best-so-far route scored with hybrid objective J"))
    st.caption("Both ACO variants are plotted on the SAME hybrid objective J. The distance-only ACO "
               "does not optimise J directly, so its curve may flatten above the hybrid curve.")

    with st.expander("All individual runs"):
        st.dataframe(runs.round(4), width="stretch", hide_index=True)


# ------------------------------------------------------------------------------------
# Page 6 - results and export
# ------------------------------------------------------------------------------------
def collect_available_results():
    """Routes that can be shown on the export page: {label: (problem, config, SolverOutput)}."""
    available = {}
    run = st.session_state["last_run"]
    if run is not None:
        available[f"Single run (page 3): {run['method']}"] = (run["problem"], run["config"], run["output"])
    outcome = st.session_state["comparison"]
    if outcome is not None:
        fuzzy = get_fuzzy_system()
        for method, outputs in outcome["outputs"].items():
            best = min(outputs, key=lambda o: evaluate_route(outcome["problem"], o.delivery_order,
                                                             outcome["config"], fuzzy).objective)
            suffix = "" if len(outputs) == 1 else f" - best of {len(outputs)} runs"
            available[f"Comparison (page 5): {method}{suffix}"] = (outcome["problem"], outcome["config"], best)
    return available


def build_report_text(problem, config, label, result, outcome):
    lines = [
        "# QuickRoute - Project Report Summary",
        "",
        "## Problem",
        f"Single-vehicle closed-tour delivery routing with {problem.n_deliveries} deliveries "
        f"(dataset: {st.session_state['data_source']}). Distances are simulated units (su); "
        f"vehicle speed {problem.speed:g} su/min.",
        "",
        "## Method",
        "Ant Colony Optimization (implemented from scratch) searches delivery orders. A Mamdani fuzzy "
        "system (priority x deadline tightness -> penalty 0-100, centroid defuzzification) scores delays. "
        f"Objective: J = {config.w_distance:.2f} x normalised distance + {config.w_penalty:.2f} x normalised fuzzy penalty.",
        "",
        f"## Selected route: {label}",
        "Order: " + " -> ".join(problem.names[n] for n in result.full_route),
        "",
        f"- Distance: {result.total_distance:.2f} su (travel time {result.total_travel_time:.1f} min, "
        f"service time {result.total_service_time:.1f} min)",
        f"- Total lateness: {result.total_lateness:.1f} min; late deliveries: {result.n_late} "
        f"({result.pct_late:.0f}%)",
        f"- Priority-weighted lateness: {result.priority_weighted_lateness:.1f}",
        f"- Fuzzy penalty (sum): {result.total_penalty:.1f}; weighted penalty term: {result.weighted_penalty_term:.4f}",
        f"- Weighted distance term: {result.weighted_distance_term:.4f}; objective J = {result.objective:.4f}",
    ]
    if outcome is not None:
        lines += ["", f"## Repeated-run comparison ({outcome['n_seeds']} ACO seeds; mean over runs)"]
        for _, row in outcome["summary"].iterrows():
            lines.append(f"- {row['method']}: distance {row['total_distance_mean']:.1f}, "
                         f"lateness {row['total_lateness_mean']:.1f} min, late deliveries {row['n_late_mean']:.1f}, "
                         f"J {row['objective_mean']:.4f} (std {row['objective_std']:.4f})")
        lines.append("Positive improvement values on page 5 mean better than the baseline; negative mean worse.")
    lines += ["", "## Limitations",
              "Synthetic data and simulated distances; one vehicle; no capacities, time windows with waiting, "
              "or traffic; ACO is heuristic and does not guarantee the global optimum; results depend on the "
              "dataset, seeds and chosen weights."]
    return "\n".join(lines)


def page_results(settings):
    st.header("Results and Export")
    available = collect_available_results()
    if not available:
        st.info("No results yet. Run an algorithm on page 3 or a comparison on page 5 first.")
        return
    label = st.selectbox("Which result do you want to inspect?", list(available))
    problem, config, output = available[label]
    fuzzy = get_fuzzy_system()
    result = evaluate_route(problem, output.delivery_order, config, fuzzy)

    st.subheader("Final route sequence")
    st.markdown(" → ".join(problem.names[n] for n in result.full_route))
    show_metrics_row(result)
    st.caption(f"J = {result.weighted_distance_term:.4f} (distance term) + "
               f"{result.weighted_penalty_term:.4f} (penalty term). "
               f"Travel time {result.total_travel_time:.1f} min, service time {result.total_service_time:.1f} min, "
               f"total time {result.total_time:.1f} min.")

    table = delivery_level_table(problem, result)
    st.subheader("Delivery-level arrival and delay")
    st.dataframe(table, width="stretch", hide_index=True)

    metrics_row = route_metrics(problem, output, config, fuzzy)
    single_summary = pd.DataFrame([{**{"result": label}, **metrics_row}])
    outcome = st.session_state["comparison"]

    c1, c2, c3 = st.columns(3)
    c1.download_button("⬇ Route results (CSV)", table.to_csv(index=False).encode("utf-8"),
                       file_name="quickroute_route_results.csv", mime="text/csv")
    if outcome is not None:
        summary_csv = outcome["summary"].to_csv(index=False)
        summary_name = "quickroute_evaluation_summary.csv"
    else:
        summary_csv = single_summary.to_csv(index=False)
        summary_name = "quickroute_single_run_summary.csv"
    c2.download_button("⬇ Evaluation summary (CSV)", summary_csv.encode("utf-8"),
                       file_name=summary_name, mime="text/csv")
    report = build_report_text(problem, config, label, result, outcome)
    c3.download_button("⬇ Report summary (Markdown)", report.encode("utf-8"),
                       file_name="quickroute_report_summary.md", mime="text/markdown")

    st.subheader("Project report summary")
    st.markdown(report)


# ------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------
def main():
    init_state()
    st.sidebar.title("🚚 QuickRoute")
    page = st.sidebar.radio("Navigate", PAGES)
    settings = sidebar_settings()
    try:
        if page == PAGES[0]:
            page_overview()
        elif page == PAGES[1]:
            page_dataset()
        elif page == PAGES[2]:
            page_optimization(settings)
        elif page == PAGES[3]:
            page_fuzzy()
        elif page == PAGES[4]:
            page_comparison(settings)
        else:
            page_results(settings)
    except (ParameterError, DataValidationError, ValueError) as error:
        st.error(f"Something went wrong: {error}")


main()
