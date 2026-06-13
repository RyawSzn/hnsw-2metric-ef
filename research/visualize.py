import argparse
import os
import sys
import warnings

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")  # non-interactive; swap to "TkAgg"/"Qt5Agg" for live windows
import matplotlib.pyplot as plt
from scipy import stats

CANDIDATE_DISTS = [
    ("Normal", stats.norm),
    ("Skew-norm", stats.skewnorm),
    ("Beta", stats.beta),
    ("Chi-squared", stats.chi2),
    ("Weibull", stats.weibull_min),
    ("Log-normal", stats.lognorm),
    ("Maxwell-B", stats.maxwell),
]

COLORS = ["crimson", "darkorange", "forestgreen", "mediumpurple", "saddlebrown", "teal"]


def load_hdf5(path: str):
    with h5py.File(path, "r") as f:
        train = f["train"][:]
        test = f["test"][:]
        neighbors = f["neighbors"][:]
        distance = f.attrs.get("distance", "angular")
        if isinstance(distance, bytes):
            distance = distance.decode()
    return train, test, neighbors, distance


def l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms


def cosine_sim_estimator(data: np.ndarray):
    mu = data.mean(axis=0)
    diff = data - mu
    Sigma = (diff.T @ diff) / (len(data) - 1)
    return mu, Sigma


def theoretical_distribution(q, mu, Sigma):
    th_mean = float(q @ mu)
    th_var = float(q @ Sigma @ q)
    return th_mean, th_var


def compute_scores(train: np.ndarray, q: np.ndarray, distance: str) -> np.ndarray:
    if distance == "angular":
        return (train @ q).astype(np.float64)
    diff = train - q
    return (diff * diff).sum(axis=1).astype(np.float64)


def fit_distributions(sim: np.ndarray, candidates: list) -> list:
    results = []
    for name, dist in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                params = dist.fit(sim)
            log_l = dist.logpdf(sim, *params).sum()
            k = len(params)
            aic = 2 * k - 2 * log_l
            results.append((name, dist, params, aic))
        except Exception:
            pass
    results.sort(key=lambda r: r[3])
    return results


def plot_histogram(
    sim: np.ndarray,
    fits: list,
    th_mean: float | None,
    th_var: float | None,
    q_idx: int,
    metric_label: str,
    out_dir: str | None,
    stem: str,
):
    emp_mean = sim.mean()
    emp_std = sim.std(ddof=1)
    margin = 0.05 * (sim.max() - sim.min())
    x = np.linspace(sim.min() - margin, sim.max() + margin, 600)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(
        sim, bins=80, density=True, alpha=0.45, color="steelblue", label="Empirical"
    )

    for (name, dist, params, aic), color in zip(fits, COLORS):
        ax.plot(
            x,
            dist.pdf(x, *params),
            color=color,
            linewidth=1.8,
            label=f"{name}  AIC={aic:.0f}",
        )

    if th_mean is not None:
        ax.axvline(
            th_mean,
            color="black",
            linestyle=":",
            linewidth=1.4,
            label=f"Ada-ef μ={th_mean:.4f}",
        )
    ax.axvline(
        emp_mean,
        color="steelblue",
        linestyle=":",
        linewidth=1.0,
        label=f"Empirical μ={emp_mean:.4f}",
    )

    ax.set_title(
        f"{metric_label} — Query {q_idx}  |  "
        f"empirical μ={emp_mean:.4f}, σ={emp_std:.4f}"
    )
    ax.set_xlabel(metric_label)
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, f"{stem}_histogram_q{q_idx}.png")


def plot_qq_grid(
    sim: np.ndarray,
    fits: list,
    q_idx: int,
    metric_label: str,
    out_dir: str | None,
    stem: str,
):
    n = len(fits)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False
    )

    for idx, (name, dist, params, aic) in enumerate(fits):
        ax = axes[idx // ncols][idx % ncols]
        (osm, osr), (slope, intercept, r) = stats.probplot(
            sim, dist=dist, sparams=params, fit=True
        )
        lo, hi = osm[0], osm[-1]
        ax.scatter(osm, osr, s=3, alpha=0.35, color="steelblue")
        ax.plot([lo, hi], [lo, hi], color="crimson", linewidth=1.3)
        ax.plot(
            [lo, hi],
            [slope * lo + intercept, slope * hi + intercept],
            color="darkorange",
            linewidth=1.3,
            linestyle="--",
        )
        ax.set_title(f"{name}\nAIC={aic:.0f}   R²={r**2:.4f}", fontsize=9)
        ax.set_xlabel("Theoretical quantiles", fontsize=8)
        ax.set_ylabel("Empirical quantiles", fontsize=8)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f"Q-Q Grid — {metric_label} — Query {q_idx}", fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir, f"{stem}_qq_q{q_idx}.png")


def _save(fig, out_dir: str | None, filename: str):
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    default_data_dir = "/home/ryawszn/experiments/data"
    default_hdf5 = os.path.join(default_data_dir, "deep-image-96-angular.hdf5")

    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", default=default_hdf5)
    parser.add_argument("--query", type=int, default=0)
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save PNGs to research/figures/ instead of displaying",
    )
    args = parser.parse_args()

    if not os.path.exists(args.hdf5):
        sys.exit(f"Dataset not found: {args.hdf5}")

    print(f"Loading {args.hdf5} …")
    train, test, _, distance = load_hdf5(args.hdf5)
    train = train.astype(np.float32)
    test = test.astype(np.float32)

    stem = os.path.splitext(os.path.basename(args.hdf5))[0]
    print(
        f"Distance metric: {distance}   Database: {train.shape}   Queries: {test.shape}"
    )

    candidates = CANDIDATE_DISTS
    if distance == "angular":
        train = l2_normalize(train)
        test = l2_normalize(test)
        metric_label = "Cosine similarity"
    else:
        metric_label = "Squared L2 distance"

    q = test[args.query]

    if distance == "angular":
        print("Computing covariance matrix …")
        mu, Sigma = cosine_sim_estimator(train)
        th_mean, th_var = theoretical_distribution(q, mu, Sigma)
    else:
        th_mean, th_var = None, None

    print(f"Computing exact {metric_label.lower()} values …")
    sim = compute_scores(train, q, distance)

    emp_std = sim.std(ddof=1)
    emp_skewness = float(stats.skew(sim))
    emp_kurtosis = float(stats.kurtosis(sim))

    print(f"\n{'=' * 50}")
    if th_mean is not None:
        print(f"[Theoretical (Ada-ef) — Query {args.query}]")
        print(f"  Mean : {th_mean:.6f}")
        print(f"  Var  : {th_var:.6f}")
        print(f"  Std  : {np.sqrt(th_var):.6f}")
    print(f"\n[Empirical — Query {args.query}]")
    print(f"  Mean           : {sim.mean():.6f}")
    print(f"  Var            : {emp_std**2:.6f}")
    print(f"  Std            : {emp_std:.6f}")
    print(f"  Skewness       : {emp_skewness:.4f}  (Normal = 0)")
    print(f"  Excess kurtosis: {emp_kurtosis:.4f}  (Normal = 0)")

    print(f"\nFitting {len(candidates)} distributions …")
    fits = fit_distributions(sim, candidates)
    print(f"\n  {'Distribution':<14}  {'AIC':>14}")
    print("  " + "-" * 32)
    for rank, (name, _, _, aic) in enumerate(fits, 1):
        marker = "  ← best" if rank == 1 else ""
        print(f"  {name:<14}  {aic:>14.0f}{marker}")
    print(f"{'=' * 50}\n")

    out_dir = (
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
        if args.save
        else None
    )

    print("Plotting histogram …")
    plot_histogram(sim, fits, th_mean, th_var, args.query, metric_label, out_dir, stem)

    print("Plotting Q-Q grid …")
    plot_qq_grid(sim, fits, args.query, metric_label, out_dir, stem)


if __name__ == "__main__":
    main()
