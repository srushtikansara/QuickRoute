"""Mamdani fuzzy inference system for the delivery penalty.

Implemented directly in Python/NumPy (no scikit-fuzzy). Reasons:
  * the ACO calls the fuzzy system thousands of times, so a small, cached,
    dependency-free implementation is much faster;
  * scikit-fuzzy is not reliably installable on newer Python/NumPy versions;
  * every step (fuzzification, rule firing, aggregation, centroid) is visible,
    which is useful for a viva.

INPUTS
  priority        universe [1, 5]       terms: Low, Medium, High
  lateness_signed universe [-60, 120]   terms: Comfortable, Tight, Critical
        lateness_signed = arrival_time - deadline   (minutes)
        NEGATIVE = arrives early (slack), POSITIVE = deadline missed.
OUTPUT
  penalty         universe [0, 100]     terms: Low, Medium, High, Very high

Operators: AND = min, rule implication = min (clipping), aggregation = max,
defuzzification = centroid of the aggregated output shape.

KNOWN PROPERTY (not a bug): centroid defuzzification of clipped shapes is not perfectly
monotone. Measured on a fine grid, the penalty can dip by at most about 3.5 points along
the lateness axis and about 6.3 points along the priority axis (scale 0-100) before it
continues to rise. The overall trends (later -> larger penalty, higher priority -> larger
penalty) hold; the tests check this with an explicit tolerance. Also, the smallest possible
output is about 10.8 (the centroid of the 'Low' shape), not 0.
"""

from dataclasses import dataclass

import numpy as np

PRIORITY_RANGE = (1.0, 5.0)
LATENESS_RANGE = (-60.0, 120.0)
PENALTY_RANGE = (0.0, 100.0)
PENALTY_MAX = PENALTY_RANGE[1]


@dataclass(frozen=True)
class Trapezoid:
    """Trapezoidal membership function with corner points a <= b <= c <= d.

    mu = 0 outside (a, d), rises on [a, b], equals 1 on [b, c], falls on [c, d].
    A triangle is a trapezoid with b == c. Shoulders use a == b or c == d.
    """
    a: float
    b: float
    c: float
    d: float

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        rising = np.where(self.b > self.a, (x - self.a) / max(self.b - self.a, 1e-12),
                          (x >= self.b).astype(float))
        falling = np.where(self.d > self.c, (self.d - x) / max(self.d - self.c, 1e-12),
                           (x <= self.c).astype(float))
        return np.clip(np.minimum(rising, falling), 0.0, 1.0)

    def value(self, x):
        """Scalar (pure Python) membership degree - used for fast fuzzification."""
        if x < self.a or x > self.d:
            return 0.0
        if x < self.b:
            return (x - self.a) / (self.b - self.a)
        if x <= self.c:
            return 1.0
        if x < self.d:
            return (self.d - x) / (self.d - self.c)
        return 1.0 if self.c == self.d else 0.0


# ----- membership functions (documented design) --------------------------------
PRIORITY_TERMS = {
    "Low": Trapezoid(1, 1, 2, 3),
    "Medium": Trapezoid(2, 3, 3, 4),
    "High": Trapezoid(3, 4, 5, 5),
}
TIGHTNESS_TERMS = {
    "Comfortable": Trapezoid(-60, -60, -30, 0),
    "Tight": Trapezoid(-30, 0, 0, 60),
    "Critical": Trapezoid(0, 60, 120, 120),
}
PENALTY_TERMS = {
    "Low": Trapezoid(0, 0, 10, 30),
    "Medium": Trapezoid(15, 40, 40, 65),
    "High": Trapezoid(45, 70, 70, 90),
    "Very high": Trapezoid(70, 90, 100, 100),
}

# ----- rule base: every (priority, tightness) pair has exactly one rule ----------
# (priority term, tightness term) -> penalty term
RULES = {
    ("Low", "Comfortable"): "Low",
    ("Low", "Tight"): "Low",
    ("Low", "Critical"): "Medium",
    ("Medium", "Comfortable"): "Low",
    ("Medium", "Tight"): "Medium",
    ("Medium", "Critical"): "High",
    ("High", "Comfortable"): "Low",
    ("High", "Tight"): "High",
    ("High", "Critical"): "Very high",
}


def rule_text(priority_term, tightness_term):
    output = RULES[(priority_term, tightness_term)]
    return (f"IF priority is {priority_term} AND deadline is {tightness_term} "
            f"THEN penalty is {output}")


@dataclass
class FuzzyResult:
    """Everything the Fuzzy Logic Explorer needs to explain one inference."""
    priority: float
    lateness_signed: float
    priority_memberships: dict
    tightness_memberships: dict
    rule_strengths: dict          # (priority term, tightness term) -> firing strength
    output_strengths: dict        # penalty term -> max firing strength (after aggregation)
    output_grid: np.ndarray
    aggregated_membership: np.ndarray
    penalty: float


class FuzzyPenaltySystem:
    """Mamdani FIS: (priority, signed lateness) -> penalty in [0, 100]."""

    def __init__(self, resolution=501):
        self.output_grid = np.linspace(PENALTY_RANGE[0], PENALTY_RANGE[1], resolution)
        self.output_curves = {name: mf(self.output_grid) for name, mf in PENALTY_TERMS.items()}
        self._cache = {}

    # ---- full, explainable inference --------------------------------------------
    def infer(self, priority, lateness_signed):
        # 1) Clip inputs into their universes (values outside saturate).
        priority = float(np.clip(priority, *PRIORITY_RANGE))
        lateness_signed = float(np.clip(lateness_signed, *LATENESS_RANGE))

        # 2) Fuzzification: crisp value -> membership degree of every term.
        priority_mu = {name: mf.value(priority) for name, mf in PRIORITY_TERMS.items()}
        tightness_mu = {name: mf.value(lateness_signed) for name, mf in TIGHTNESS_TERMS.items()}

        # 3) Rule evaluation: AND = min.
        rule_strengths = {}
        output_strengths = {name: 0.0 for name in PENALTY_TERMS}
        for (p_term, t_term), out_term in RULES.items():
            strength = min(priority_mu[p_term], tightness_mu[t_term])
            rule_strengths[(p_term, t_term)] = strength
            # 4) Aggregation of rules sharing the same output term: max.
            output_strengths[out_term] = max(output_strengths[out_term], strength)

        # 5) Implication (clip each output term at its strength) + aggregation (max).
        aggregated = np.zeros_like(self.output_grid)
        for out_term, strength in output_strengths.items():
            if strength > 0:
                aggregated = np.maximum(aggregated, np.minimum(self.output_curves[out_term], strength))

        # 6) Centroid defuzzification.
        area = aggregated.sum()
        if area <= 0:       # cannot happen with the partitions above, but stay safe
            penalty = 0.5 * (PENALTY_RANGE[0] + PENALTY_RANGE[1])
        else:
            penalty = float((self.output_grid * aggregated).sum() / area)

        return FuzzyResult(priority, lateness_signed, priority_mu, tightness_mu,
                           rule_strengths, output_strengths, self.output_grid,
                           aggregated, penalty)

    # ---- fast version used inside the optimiser ---------------------------------
    def penalty(self, priority, lateness_signed):
        key = (round(float(priority), 3), round(float(lateness_signed), 1))
        cached = self._cache.get(key)
        if cached is None:
            cached = self.infer(priority, lateness_signed).penalty
            self._cache[key] = cached
        return cached

    def explain(self, result):
        """Plain-English explanation of one inference result."""
        active = [(rule, s) for rule, s in result.rule_strengths.items() if s > 0]
        active.sort(key=lambda item: -item[1])
        lines = []
        state = "late" if result.lateness_signed > 0 else "early / on time"
        lines.append(
            f"Priority {result.priority:.1f} with arrival − deadline = "
            f"{result.lateness_signed:+.1f} min ({state}).")
        for rule, strength in active:
            lines.append(f"Rule fired at {strength:.2f}: {rule_text(*rule)}.")
        lines.append(
            "The fired rules clip their output shapes; the shapes are combined with max, "
            f"and the centroid of the combined shape is {result.penalty:.1f} (scale 0–100).")
        lines.append("Higher priority moves the inputs toward the 'High' priority term, so "
                     "the same lateness triggers rules with a larger output (e.g. High + Critical → Very high).")
        return lines
