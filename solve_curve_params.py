import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.optimize import least_squares

DATA_PATH = "xy_data.csv"

# Fixed constants from the problem statement
Y_OFFSET = 42.0
OMEGA = 0.3  # the 0.3 in sin(0.3t)

# Parameter bounds from the problem statement
THETA_BOUNDS = (np.deg2rad(1e-4), np.deg2rad(50 - 1e-4))
M_BOUNDS = (-0.05 + 1e-6, 0.05 - 1e-6)
X_BOUNDS = (1e-4, 100 - 1e-4)
T_RANGE = (6, 60)


def recover_t_v(x, y, theta, X):
    """Invert the rotation to get the (t, v) implied by a candidate (theta, X)."""
    xp = x - X
    yp = y - Y_OFFSET
    t = xp * np.cos(theta) + yp * np.sin(theta)
    v_obs = -xp * np.sin(theta) + yp * np.cos(theta)
    return t, v_obs


def residuals(params, x, y):
    """Residual vector: observed v (from data) minus predicted v (from model)."""
    theta, M, X = params
    t, v_obs = recover_t_v(x, y, theta, X)
    v_pred = np.exp(M * np.abs(t)) * np.sin(OMEGA * t)
    return v_obs - v_pred


def forward(t, theta, M, X):
    """Forward model: compute (x, y) for given t and parameters."""
    v = np.exp(M * np.abs(t)) * np.sin(OMEGA * t)
    x = t * np.cos(theta) - v * np.sin(theta) + X
    y = Y_OFFSET + t * np.sin(theta) + v * np.cos(theta)
    return x, y


def fit_parameters(x, y, n_theta=9, n_M=5, n_X=5, verbose=True):
    """Global multi-start + local bounded least-squares refinement."""
    lb = [THETA_BOUNDS[0], M_BOUNDS[0], X_BOUNDS[0]]
    ub = [THETA_BOUNDS[1], M_BOUNDS[1], X_BOUNDS[1]]

    theta_grid = np.linspace(THETA_BOUNDS[0], THETA_BOUNDS[1], n_theta)
    M_grid = np.linspace(M_BOUNDS[0], M_BOUNDS[1], n_M)
    X_grid = np.linspace(X_BOUNDS[0], X_BOUNDS[1], n_X)

    best_cost, best_result = np.inf, None
    for theta0 in theta_grid:
        for M0 in M_grid:
            for X0 in X_grid:
                p0 = [theta0, M0, X0]
                result = least_squares(
                    residuals, p0, args=(x, y),
                    bounds=(lb, ub), method="trf",
                    xtol=1e-14, ftol=1e-14,
                )
                cost = np.sum(result.fun ** 2)
                if cost < best_cost:
                    best_cost, best_result = cost, result

    theta_hat, M_hat, X_hat = best_result.x
    if verbose:
        print(f"Best cost (sum sq residual): {best_cost:.6e}")
        print(f"theta = {np.rad2deg(theta_hat):.6f} deg  ({theta_hat:.6f} rad)")
        print(f"M     = {M_hat:.6f}")
        print(f"X     = {X_hat:.6f}")

    return theta_hat, M_hat, X_hat, best_cost


def validate(x, y, theta, M, X, n_curve_samples=2000, verbose=True):
    """Sanity checks: recovered t should lie in (6,60); L1 distance to reconstructed curve."""
    t_rec, _ = recover_t_v(x, y, theta, X)

    t_uniform = np.linspace(T_RANGE[0], T_RANGE[1], n_curve_samples)
    x_curve, y_curve = forward(t_uniform, theta, M, X)

    l1_total = 0.0
    for xi, yi in zip(x, y):
        d = np.abs(x_curve - xi) + np.abs(y_curve - yi)
        l1_total += d.min()
    mean_l1 = l1_total / len(x)

    if verbose:
        print(f"\nRecovered t range: [{t_rec.min():.4f}, {t_rec.max():.4f}]  (expected within (6,60))")
        print(f"Mean L1 distance (data point -> reconstructed curve): {mean_l1:.6e}")

    return t_rec, mean_l1


def plot_results(x, y, theta, M, X, t_rec, mean_l1):
    """Visualize data points vs. the fitted parametric curve."""
    theta_deg = np.rad2deg(theta)

    # Dense curve for smooth rendering
    t_dense = np.linspace(T_RANGE[0], T_RANGE[1], 4000)
    x_curve, y_curve = forward(t_dense, theta, M, X)

    # Sparse curve to draw tick marks along the curve
    t_ticks = np.arange(10, 61, 10, dtype=float)
    x_ticks, y_ticks = forward(t_ticks, theta, M, X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0f1117")

    # ── Left panel: XY plane ──────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#1a1d27")

    # Fitted curve
    ax.plot(x_curve, y_curve, color="#7c83fd", linewidth=2.2,
            label="Fitted curve", zorder=2)

    # Data points
    sc = ax.scatter(x, y, c=t_rec, cmap="plasma", s=40, zorder=3,
                    edgecolors="white", linewidths=0.4, label="Data points")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Recovered  t", color="#cccccc", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#cccccc")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#cccccc")

    # t-tick markers along the curve
    ax.scatter(x_ticks, y_ticks, color="#f4c542", s=60, zorder=4,
               marker="D", label="t = 10,20,…,60")
    for ti, xi, yi in zip(t_ticks, x_ticks, y_ticks):
        ax.annotate(f"t={int(ti)}", (xi, yi),
                    textcoords="offset points", xytext=(6, 5),
                    color="#f4c542", fontsize=7.5,
                    path_effects=[pe.withStroke(linewidth=2,
                                                foreground="#1a1d27")])

    ax.set_title("Parametric Curve Fit", color="white", fontsize=13, pad=10)
    ax.set_xlabel("x", color="#aaaaaa")
    ax.set_ylabel("y", color="#aaaaaa")
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444455")
    ax.legend(facecolor="#252838", edgecolor="#444455",
              labelcolor="white", fontsize=9)

    # Parameter box
    info = (
        f"θ  = {theta_deg:.4f}°\n"
        f"M  = {M:.4f}\n"
        f"X  = {X:.4f}\n"
        f"SSR = {mean_l1:.2e}"
    )
    ax.text(0.02, 0.97, info, transform=ax.transAxes, va="top",
            color="#e0e0ff", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#252838",
                      edgecolor="#7c83fd", alpha=0.85))

    # ── Right panel: residuals along the curve ────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#1a1d27")

    res = residuals([theta, M, X], x, y)   # v_obs - v_pred per point
    ax2.scatter(t_rec, res, c=np.abs(res), cmap="RdYlGn_r",
                s=35, zorder=3, edgecolors="white", linewidths=0.3)
    ax2.axhline(0, color="#7c83fd", linewidth=1.2, linestyle="--", alpha=0.7)
    ax2.set_title("Residuals  (v_obs − v_pred)", color="white",
                  fontsize=13, pad=10)
    ax2.set_xlabel("Recovered  t", color="#aaaaaa")
    ax2.set_ylabel("Residual", color="#aaaaaa")
    ax2.tick_params(colors="#aaaaaa")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#444455")

    fig.suptitle(
        f"Parametric Curve Parameter Estimation  |  θ={theta_deg:.2f}°  M={M:.4f}  X={X:.4f}",
        color="white", fontsize=11, y=1.01
    )
    plt.tight_layout()
    out_path = "curve_fit_plot.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nPlot saved → {out_path}")
    plt.show()


def main():
    df = pd.read_csv(DATA_PATH)
    x, y = df["x"].values, df["y"].values

    theta_hat, M_hat, X_hat, cost = fit_parameters(x, y)
    t_rec, mean_l1 = validate(x, y, theta_hat, M_hat, X_hat)

    theta_deg = np.rad2deg(theta_hat)
    print("\n=== Final Answer ===")
    print(f"theta = {theta_deg:.4f} deg")
    print(f"M     = {M_hat:.4f}")
    print(f"X     = {X_hat:.4f}")

    print("\n=== Desmos / LaTeX submission string ===")
    print(
        f'\\left(t*\\cos({theta_hat:.4f})-e^{{{M_hat:.4f}\\left|t\\right|}}\\cdot'
        f'\\sin(0.3t)\\sin({theta_hat:.4f})+{X_hat:.4f},'
        f'42+t*\\sin({theta_hat:.4f})+e^{{{M_hat:.4f}\\left|t\\right|}}\\cdot'
        f'\\sin(0.3t)\\cos({theta_hat:.4f})\\right)'
    )

    plot_results(x, y, theta_hat, M_hat, X_hat, t_rec, mean_l1)


if __name__ == "__main__":
    main()
