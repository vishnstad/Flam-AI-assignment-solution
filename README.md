# AI R&D Assignment — Parametric Curve Parameter Estimation

**Problem:** Find unknowns θ, M, X in a parametric curve given only a scatter of (x, y) points.

---

## 1. The Problem

Given model:

```
x(t) = t*cos(θ) - e^(M|t|)*sin(0.3t)*sin(θ) + X
y(t) = 42 + t*sin(θ) + e^(M|t|)*sin(0.3t)*cos(θ)
```

Unknowns and their allowed ranges:

```
0° < θ < 50°
-0.05 < M < 0.05
0 < X < 100
```

Parameter `t` (unknown per data point) ranges over `6 < t < 60`.

Given: `xy_data.csv` — 1500 `(x, y)` points that lie on this curve, with no `t` value recorded per point.

**Task:** recover θ, M, X. Scored by the L1 distance between the reconstructed curve and the true one.

---

## 2. Conceptual Foundation — What "parametric curve" means

`x(t)` and `y(t)` are **not two curves** — they're the two coordinates needed to locate **one** point at each instant `t`. Like a pen tracing a squiggle: `x(t)` is how far right the pen tip is at "time" t, `y(t)` is how far up. Sweep t from 6 to 60 and the ink trail is the one curve. Neither equation alone draws anything; together they trace a single path.

---

## 3. The Key Trick — Reformulating as a Rotation

The two equations can be rearranged:

```
x - X  = t·cos(θ) - v·sin(θ)
y - 42 = t·sin(θ) + v·cos(θ)      where v = e^(M|t|)·sin(0.3t)
```

This is exactly a 2D **rotation matrix** applied to the vector `(t, v)`:

```
[x-X ]   [cos θ   -sin θ] [t]
[y-42] = [sin θ    cos θ] [v]
```

### Why this can be inverted for free

Rotation matrices are **orthogonal**: `R(θ)⁻¹ = R(θ)ᵀ` (transpose = inverse — no computation needed, just flip the sign on the off-diagonal terms). Geometrically: rotating forward by θ and then by −θ returns you to where you started.

So for **any candidate (θ, X)**, every data point directly tells you its own implied `t` and `v`:

```
t = (x-X)·cos(θ) + (y-42)·sin(θ)
v = -(x-X)·sin(θ) + (y-42)·cos(θ)
```

### Why this matters

- Turns a hard "match point to nearest spot on curve" search into simple 1D algebra.
- Gives a free sanity check: correct (θ, X) ⟹ recovered `t` lands inside (6, 60).
- The recovered `v` must equal the model's own prediction `e^(M·t)·sin(0.3t)` — the gap between them **is** the residual to minimize.

**Worked numeric example** (row 1: x=88.3645, y=57.7844, using the final θ=30°, X=55):
```
t ≈ 36.7867,  v ≈ -3.01256
model-predicted v (M=0.03) ≈ -3.01254   →  essentially matches
```

---

## 4. Optimization Problem Formulation

Residual per point:

```
r_i(θ,M,X) = v_obs,i(θ,X) - e^(M·|t_i(θ,X)|)·sin(0.3·t_i(θ,X))
```

**Nonlinear least-squares, box-constrained, non-convex:**

```
minimize   F(θ,M,X) = Σ r_i(θ,M,X)²
subject to θ ∈ (0°,50°), M ∈ (-0.05,0.05), X ∈ (0,100)
```

- **Least-squares** (not generic minimization) → lets solvers approximate the Hessian as `JᵀJ` from the Jacobian alone.
- **Nonlinear**: θ inside sin/cos, M inside an exponent → no single matrix solve finishes the job.
- **Box-constrained**: independent min/max per variable, no coupling between them.
- **Non-convex**: `sin(0.3t)` and `e^(M|t|)` create plateaus/local minima — confirmed empirically (optimizer got stuck for iterations 3–10 in one run before breaking free).

---

## 5. Algorithm

**Stage A — Global search** (avoid local minima/plateaus): grid of starting guesses across the allowed ranges (e.g. 9×5×5 = 225 combinations).

**Stage B — Local refinement** (Levenberg-Marquardt / Trust Region Reflective):

```
repeat:
    r ← residual(p)                      # vector of n residuals
    J ← Jacobian(r, p)                   # via finite differences
    solve (JᵀJ + λI)Δp = -Jᵀr            # the "Ax = b" step
    Δp ← clip so p+Δp stays within bounds
    if cost(p+Δp) < cost(p):
        accept: p ← p+Δp, λ ← λ/2
    else:
        reject: λ ← λ×3
until convergence
```

**Stage C — Validation:** recovered `t` should span ≈(6,60); cost should be ≈0.

### Why Trust Region Reflective over plain Levenberg-Marquardt

Plain LM has no notion of bounds — in one demonstration run it let `M` wander to `-0.356`, well outside the allowed `(-0.05, 0.05)`, before self-correcting. TRF (`scipy.optimize.least_squares` with `bounds=...`) reflects steps back into the feasible box at every iteration, so it never reports an out-of-range intermediate or final value.

### `Ax = b`, precisely where it lives

| Step | Linear or not? |
|---|---|
| Recover (t, v) from (x, y) given a guess (θ, X) | **Linear** — literally `R(θ)ᵀ · b` |
| Check v against model prediction | **Nonlinear** — θ, M inside sin/cos/exp |
| Find the (θ,M,X) minimizing total mismatch | **Locally linearized every iteration**: `(JᵀJ + λI)·Δp = -Jᵀr` is solved once per step, and the loop repeats on the updated point |

---

## 6. Iteration Trace (real run, starting from θ=20°, M=0, X=40)

| iter | θ (deg) | M | X | cost |
|---|---|---|---|---|
| 0 | 20.00 | 0.000 | 40.00 | 2.01×10⁴ |
| 1 | 20.95 | -0.059 | 38.84 | 1.60×10⁴ |
| 2 | 26.71 | -0.356 *(out of bounds — demo only)* | 54.21 | 1.13×10⁴ |
| 3–10 | 26.71 | -0.356 | 54.21 | 1.13×10⁴ *(plateau)* |
| 20 | 28.71 | -0.056 | 54.07 | 7.95×10³ |
| 34 | **30.00** | **0.030** | **55.00** | 3.19×10⁻⁸ |

---

## 7. Result

```
θ = 30°   (0.5236 rad)
M = 0.03
X = 55
```

**Verification (forward pass):** plugging t=36.7867 and the fitted parameters back into the original x(t), y(t) equations gives `(88.364446, 57.784394)`, matching the real data point `(88.364456, 57.784378)` to 4 decimal places.

**Fit quality across all 1500 points:**
- Sum of squared residuals: 1.8×10⁻⁸ (essentially exact)
- Recovered t range: [6.05, 59.995] — matches expected (6, 60)
- Mean L1 distance (data point → nearest reconstructed curve point): ≈ 0.01 (limited by curve-sampling grid resolution, not model error)

**Desmos / LaTeX submission string:**
```
\left(t*\cos(0.5236)-e^{0.03\left|t\right|}\cdot\sin(0.3t)\sin(0.5236)+55,42+\
t*\sin(0.5236)+e^{0.03\left|t\right|}\cdot\sin(0.3t)\cos(0.5236)\right)
```

### Visualization

![Parametric Curve Fit](curve_fit_plot.png)

---

## 8. Python Implementation

See [`solve_curve_params.py`](solve_curve_params.py). Structure:

1. `recover_t_v(x, y, theta, X)` — the closed-form rotation inverse
2. `residuals(params, x, y)` — the objective function fed to the solver
3. `forward(t, theta, M, X)` — the original equations, for validation
4. `fit_parameters(...)` — global multi-start grid + `scipy.optimize.least_squares` (method="trf", bounded)
5. `validate(...)` — sanity checks (t range) + L1 distance metric matching the assignment's own scoring criterion
6. `plot_results(...)` — matplotlib visualization of the fit and residuals

### Requirements

```
numpy
pandas
scipy
matplotlib
```

### Usage

```bash
pip install numpy pandas scipy matplotlib
python solve_curve_params.py
```

Place `xy_data.csv` in the same directory before running.
