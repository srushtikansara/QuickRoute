# QuickRoute - Methodology notes

## 1. Design decisions

| Decision | Reason |
|---|---|
| Single vehicle, closed tour | Required default; keeps the problem a TSP with deadlines so the effect of the fuzzy penalty is easy to explain. |
| Distances in *simulated units* | Coordinates are synthetic; no real-world scale is claimed. Speed is "units per minute". |
| Own Mamdani implementation (no scikit-fuzzy) | The fuzzy system is called tens of thousands of times per experiment (results are cached); scikit-fuzzy is not reliably installable on recent Python/NumPy; each step can be shown in a viva. |
| Same ACO code for both ACO variants | Only the cost function differs (distance vs. hybrid J), so any difference comes from the objective, not from the algorithm. |
| Heuristic = 1/distance in both variants | Matches the specification; the delay information reaches the ants only through pheromone (a deadline-aware heuristic is future work). |
| Arrival earlier than the deadline is not delayed | No waiting is modelled; lateness is `max(0, arrival - deadline)`. |

## 2. Fixed reference scales

* `D_ref = 2 * sum_i d(depot, i)` - the length of the "star tour" (depot -> each customer -> depot). It is determined by the dataset alone.
* `P_ref = 100 * n` - the largest possible sum of penalties.

Because they do not depend on the candidate route, `J` values are comparable between routes and between methods.

## 3. Fuzzy system definition

Inputs: priority 1-5, and `arrival - deadline` in minutes clipped to [-60, 120] (positive = late).

| Variable | Term | (a, b, c, d) |
|---|---|---|
| Priority | Low | 1, 1, 2, 3 |
| | Medium | 2, 3, 3, 4 |
| | High | 3, 4, 5, 5 |
| Deadline tightness | Comfortable | -60, -60, -30, 0 |
| | Tight | -30, 0, 0, 60 |
| | Critical | 0, 60, 120, 120 |
| Penalty | Low | 0, 0, 10, 30 |
| | Medium | 15, 40, 40, 65 |
| | High | 45, 70, 70, 90 |
| | Very high | 70, 90, 100, 100 |

The input terms form a partition of unity (memberships sum to 1 everywhere), so at least one rule always fires.

| Priority \ Deadline | Comfortable | Tight | Critical |
|---|---|---|---|
| Low | Low | Low | Medium |
| Medium | Low | Medium | High |
| High | Low | High | Very high |

Operators: AND = min, implication = min (clip), aggregation = max, defuzzification = centroid on a 501-point grid.

### Known properties

* **Floor:** the smallest output is about 10.8 (centroid of the `Low` shape), so even an on-time delivery has a small penalty. It adds a near-constant to every route and does not change the ranking.
* **Saturation:** lateness beyond 60 minutes is treated as fully "Critical"; beyond that the penalty only changes through priority.
* **Ripple:** the output is not perfectly monotone. On a fine grid the largest peak-to-valley dip is about 3.5 points along the lateness axis and 6.3 points along the priority axis. A search over monotone rule tables with the same terms reduced the dip only to 5.3-6.3 points for rule bases that keep "Comfortable -> Low"; a table without that property reached 1.4 points but gives on-time Medium and High priority deliveries a Medium-level penalty, so the interpretable table above was kept.
* **Weights matter:** with `w_d = w_p = 0.5` the distance term can dominate small penalty differences (see `tests/test_route_evaluator.py`, where a weight of 0.8 on the penalty is needed to prefer the longer on-time route in a constructed example).

## 4. Experimental protocol

1. Build the dataset (`generate_dataset(n, seed)`), validate and clean it.
2. Fix weights (0.5 / 0.5), speed (1 su/min) and ACO parameters (20 ants, 100 iterations, alpha 1, beta 3, rho 0.3, tau0 1, Q 1, elitist weight 2) **before** looking at results.
3. Run Nearest Neighbor once and each ACO variant for seeds 0-9.
4. Score every final route with the same hybrid objective `J`; report mean and standard deviation (population std) per method.
5. Improvement % = `(baseline - method) / baseline * 100` on mean values (lower is better for every metric).
6. Small instance (8 deliveries, different dataset seed): compare hybrid ACO with the brute-force optimum.
7. Optional sensitivity study of `w_d` on a **separate** dataset (seed 7); it is descriptive only and not used to choose the evaluation settings.

## 5. Threats to validity

* One synthetic dataset per experiment; results may differ on other seeds, sizes and weights.
* The hybrid method optimises the same J that is used for scoring, so J favours it by construction; the distance/lateness trade-off and the late-delivery counts are the more informative figures.
* The fuzzy design (membership functions, rules) is hand-made and was not tuned on data.
* ACO parameters are generic defaults, not tuned per method.

## 6. Reproducing the numbers

```
python experiments.py --sensitivity
```

writes `results_runs.csv`, `results_summary.csv`, `results_improvement.csv`, `results_small_instance.csv` and `results_weight_sensitivity.csv` into this folder.
