"""Plotting helpers (Plotly for routes, Matplotlib for the other charts)."""

import matplotlib

matplotlib.use("Agg")   # no GUI window; Streamlit displays the figure
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from .fuzzy_system import (LATENESS_RANGE, PENALTY_RANGE, PENALTY_TERMS, PRIORITY_RANGE,
                           PRIORITY_TERMS, TIGHTNESS_TERMS)

METHOD_COLORS = {
    "Nearest Neighbor": "#7f7f7f",
    "ACO Distance-Only": "#1f77b4",
    "Hybrid ACO + Fuzzy": "#d62728",
}


def locations_figure(df):
    """Scatter plot of the depot and the customers (marker colour = priority)."""
    depot = df[df["is_depot"]]
    customers = df[~df["is_depot"]]
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=customers["x_coordinate"], y=customers["y_coordinate"], mode="markers+text",
        text=customers["delivery_id"].astype(str), textposition="top center",
        marker=dict(size=14, color=customers["priority"], colorscale="YlOrRd", cmin=1, cmax=5,
                    showscale=True, colorbar=dict(title="Priority"), line=dict(width=1, color="black")),
        customdata=np.c_[customers["location_name"], customers["priority"],
                         customers["delivery_deadline_minutes"]],
        hovertemplate="%{customdata[0]}<br>Priority %{customdata[1]}<br>"
                      "Deadline %{customdata[2]} min<extra></extra>",
        name="Customers"))
    figure.add_trace(go.Scatter(
        x=depot["x_coordinate"], y=depot["y_coordinate"], mode="markers+text", text=["Depot"],
        textposition="bottom center",
        marker=dict(size=22, symbol="star", color="royalblue", line=dict(width=1, color="black")),
        name="Depot"))
    figure.update_layout(xaxis_title="x (simulated distance units)",
                         yaxis_title="y (simulated distance units)",
                         height=520, margin=dict(l=10, r=10, t=30, b=10),
                         yaxis=dict(scaleanchor="x", scaleratio=1))
    return figure


def route_figure(problem, result, title=""):
    """Interactive route plot: arrows show direction, labels show the delivery order."""
    coords = problem.coordinates
    figure = go.Figure()

    # Route legs with arrowheads.
    for start, end in zip(result.full_route[:-1], result.full_route[1:]):
        figure.add_annotation(x=coords[end, 0], y=coords[end, 1], ax=coords[start, 0], ay=coords[start, 1],
                              xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                              arrowhead=3, arrowsize=1.2, arrowwidth=1.6, arrowcolor="#555555")

    stops = result.delivery_order
    late_flags = [result.lateness[n] > 1e-9 for n in stops]
    figure.add_trace(go.Scatter(
        x=[coords[n, 0] for n in stops], y=[coords[n, 1] for n in stops], mode="markers+text",
        text=[str(i + 1) for i in range(len(stops))], textposition="middle center",
        textfont=dict(color="white", size=11),
        marker=dict(size=24, color=["#d62728" if late else "#2ca02c" for late in late_flags],
                    line=dict(width=1, color="black")),
        customdata=np.c_[[problem.names[n] for n in stops], [int(problem.priorities[n]) for n in stops],
                         [round(result.arrivals[n], 1) for n in stops],
                         [problem.deadlines[n] for n in stops],
                         [round(result.penalties[n], 1) for n in stops]],
        hovertemplate="Stop %{text}: %{customdata[0]}<br>Priority %{customdata[1]}<br>"
                      "Arrival %{customdata[2]} min / deadline %{customdata[3]} min<br>"
                      "Fuzzy penalty %{customdata[4]}<extra></extra>",
        name="Customers (green = on time, red = late)"))
    figure.add_trace(go.Scatter(
        x=[coords[0, 0]], y=[coords[0, 1]], mode="markers+text", text=["Depot"],
        textposition="bottom center",
        marker=dict(size=26, symbol="star", color="royalblue", line=dict(width=1, color="black")),
        name="Depot (start and end)"))
    figure.update_layout(title=title, height=560, margin=dict(l=10, r=10, t=50, b=10),
                         xaxis_title="x (simulated distance units)",
                         yaxis_title="y (simulated distance units)",
                         yaxis=dict(scaleanchor="x", scaleratio=1),
                         legend=dict(orientation="h", y=-0.15))
    return figure


def membership_figure():
    """Membership functions of the two inputs and the output."""
    figure, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    specs = [
        (axes[0], PRIORITY_TERMS, np.linspace(*PRIORITY_RANGE, 401), "Priority (1-5)"),
        (axes[1], TIGHTNESS_TERMS, np.linspace(*LATENESS_RANGE, 721),
         "Arrival - deadline (min)  [> 0 = late]"),
        (axes[2], PENALTY_TERMS, np.linspace(*PENALTY_RANGE, 501), "Penalty (0-100)"),
    ]
    for axis, terms, grid, label in specs:
        for name, function in terms.items():
            axis.plot(grid, function(grid), label=name, linewidth=2)
        axis.set_xlabel(label)
        axis.set_ylabel("Membership degree")
        axis.set_ylim(-0.02, 1.05)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    axes[0].set_title("Input 1: priority")
    axes[1].set_title("Input 2: deadline tightness")
    axes[2].set_title("Output: delivery penalty")
    figure.tight_layout()
    return figure


def fuzzy_output_figure(result):
    """Aggregated output shape with the centroid marked."""
    figure, axis = plt.subplots(figsize=(7, 3.4))
    for name, function in PENALTY_TERMS.items():
        axis.plot(result.output_grid, function(result.output_grid), linestyle="--", alpha=0.5, label=name)
    axis.fill_between(result.output_grid, result.aggregated_membership, color="orange", alpha=0.6,
                      label="Aggregated output")
    axis.axvline(result.penalty, color="red", linewidth=2, label=f"Centroid = {result.penalty:.1f}")
    axis.set_xlabel("Penalty (0-100)")
    axis.set_ylabel("Membership degree")
    axis.set_ylim(-0.02, 1.05)
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    return figure


def convergence_figure(histories, ylabel="Best objective J so far"):
    """Best-so-far objective against ACO iteration. `histories` = {label: [DataFrame, ...]}.

    Several DataFrames (several seeds) are averaged; the band shows +-1 standard deviation.
    """
    figure, axis = plt.subplots(figsize=(8, 4))
    for label, frames in histories.items():
        if not frames:
            continue
        values = np.vstack([frame["best_reported"].to_numpy() for frame in frames])
        iterations = frames[0]["iteration"].to_numpy()
        mean, std = values.mean(axis=0), values.std(axis=0)
        color = METHOD_COLORS.get(label)
        axis.plot(iterations, mean, label=f"{label} (mean of {len(frames)} run(s))", color=color, linewidth=2)
        if len(frames) > 1:
            axis.fill_between(iterations, mean - std, mean + std, color=color, alpha=0.2)
    axis.set_xlabel("ACO iteration")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    return figure


def comparison_bars_figure(summary):
    """Bar charts (mean +- std) for distance, total lateness and the hybrid objective."""
    panels = [("total_distance", "Route distance (su)"),
              ("total_lateness", "Total lateness (min)"),
              ("objective", "Hybrid objective J")]
    figure, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    labels = summary["method"].tolist()
    colors = [METHOD_COLORS.get(m, "#999999") for m in labels]
    for axis, (column, title) in zip(axes, panels):
        axis.bar(range(len(labels)), summary[f"{column}_mean"], yerr=summary[f"{column}_std"],
                 color=colors, capsize=4)
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels([label.replace(" ", "\n", 1) for label in labels], fontsize=8)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    return figure
