import json
import os

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

DATASET = "glove-100-angular"
BINS = "32x32"
EXPERIMENTS = "/home/ryawszn/experiments"

LOOKUP_CSV = f"{EXPERIMENTS}/2metric/lookup/lookup_table_{DATASET}_{BINS}.csv"
CMP_CSV = f"{EXPERIMENTS}/2metric/compare/comparison_{DATASET}.csv"
STATS_JSON = f"{EXPERIMENTS}/2metric/lookup/run_stats_{DATASET}_{BINS}.json"
OUT_P1 = f"{EXPERIMENTS}/2metric/lookup/performance_{DATASET}_{BINS}.png"
OUT_P2 = f"{EXPERIMENTS}/2metric/lookup/performance_{DATASET}_{BINS}_compare.png"

try:
    with open(STATS_JSON, "r") as f:
        run_stats = json.load(f)
        RUNTIME = run_stats
        EF_DIST = {int(k): v for k, v in run_stats.get("ef_dist", {}).items()}
except Exception as e:
    print(f"Warning: Could not load {STATS_JSON}: {e}")
    RUNTIME = {
        "search_ms": 0,
        "avg_recall": 0.0,
        "avg_ef": 0.0,
        "pct5_recall": 0.0,
        "pct1_recall": 0.0,
        "target_recall": 0.95,
        "max_ef": 5000,
        "nq": 10000,
    }
    EF_DIST = {}


def extract_lower(s):
    try:
        return float(str(s).split(",")[0].replace("(", "").strip())
    except Exception:
        return -1.0


def dark_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#1a1d27")
    ax.tick_params(colors="#cccccc")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444455")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color="#cccccc")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color="#cccccc")
    if title:
        ax.set_title(title, fontsize=11, color="white", pad=7)
    ax.xaxis.label.set_color("#cccccc")
    ax.yaxis.label.set_color("#cccccc")


sns.set_theme(style="darkgrid", font_scale=1.05)


def extract_from_curve(curve_str, target_recall=0.95):
    if pd.isna(curve_str):
        return 50, 0.0
    pts = str(curve_str).strip('"').split(",")
    for pt in pts:
        if not pt:
            continue
        try:
            ef_str, rec_str = pt.split(":")
            ef, rec = int(ef_str), float(rec_str)
            if rec >= target_recall:
                return ef, rec
        except:
            continue
    if pts and pts[-1]:
        try:
            ef_str, rec_str = pts[-1].split(":")
            return int(ef_str), float(rec_str)
        except:
            pass
    return 50, 0.0


ldf = pd.read_csv(LOOKUP_CSV)
if "curve" in ldf.columns:
    parsed = ldf["curve"].apply(
        lambda c: pd.Series(extract_from_curve(c, RUNTIME.get("target_recall", 0.95)))
    )
    ldf["ef"] = parsed[0].astype(int)
    ldf["actual_recall"] = parsed[1].astype(float)
elif "curves" in ldf.columns:
    parsed = ldf["curves"].apply(
        lambda c: pd.Series(
            extract_from_curve(c.split("|")[0], RUNTIME.get("target_recall", 0.95))
        )
    )
    ldf["ef"] = parsed[0].astype(int)
    ldf["actual_recall"] = parsed[1].astype(float)

ldf["d_ep_sort"] = ldf["d_ep_bin"].apply(extract_lower)
ldf["RV_sort"] = ldf["RV_bin"].apply(extract_lower)
ldf = ldf.sort_values(["d_ep_sort", "RV_sort"])
sorted_ep = sorted(ldf["d_ep_bin"].unique(), key=extract_lower)
sorted_rv = sorted(ldf["RV_bin"].unique(), key=extract_lower)
pivot_ef = ldf.pivot(index="d_ep_bin", columns="RV_bin", values="ef").reindex(
    index=sorted_ep, columns=sorted_rv
)
pivot_recall = ldf.pivot(
    index="d_ep_bin", columns="RV_bin", values="actual_recall"
).reindex(index=sorted_ep, columns=sorted_rv)


cdf = pd.read_csv(CMP_CSV)
cdf["ratio"] = cdf["2metric_ef"] / cdf["true_ef"].clip(lower=1)
cdf["error"] = cdf["2metric_ef"] - cdf["true_ef"]
cdf["under"] = cdf["2metric_ef"] < cdf["true_ef"]

ef_vals = np.array(list(EF_DIST.keys()))
ef_cnts = np.array(list(EF_DIST.values()))

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Lookup table + runtime overview
# ═══════════════════════════════════════════════════════════════════════════════
fig1 = plt.figure(figsize=(24, 16), facecolor="#0f1117")
fig1.suptitle(
    f"2-Metric Adaptive EF — Performance Dashboard\n"
    f"{DATASET}  |  {BINS} grid  |  target recall {RUNTIME['target_recall']:.2f}",
    fontsize=20,
    fontweight="bold",
    color="white",
    y=0.96,
)
gs1 = gridspec.GridSpec(2, 3, figure=fig1, hspace=0.35, wspace=0.3)

ax_he = fig1.add_subplot(gs1[0, :2])
ax_hr = fig1.add_subplot(gs1[1, :2])
ax_efd = fig1.add_subplot(gs1[0, 2])
ax_met = fig1.add_subplot(gs1[1, 2])

# heatmap – ef
sns.heatmap(
    pivot_ef,
    ax=ax_he,
    annot=True,
    fmt=".0f",
    cmap="YlOrRd",
    annot_kws={"size": 7},
    cbar_kws={"label": "Assigned ef", "shrink": 0.8},
    linewidths=0.3,
    linecolor="#333344",
)
dark_ax(
    ax_he,
    "Lookup Table — Assigned EF per (RC, RV) Bin",
    "RV_Rank Interval",
    "d_ep Interval",
)
ax_he.set_xticklabels(ax_he.get_xticklabels(), rotation=45, ha="right", fontsize=6)
ax_he.set_yticklabels(ax_he.get_yticklabels(), rotation=0, fontsize=6)
ax_he.collections[0].colorbar.ax.tick_params(colors="white")
ax_he.collections[0].colorbar.set_label("Assigned ef", color="white")

# heatmap – recall; highlight bins below target
min_r = ldf["actual_recall"].min() - 0.003
sns.heatmap(
    pivot_recall,
    ax=ax_hr,
    annot=True,
    fmt=".3f",
    cmap="crest",
    vmin=min_r,
    vmax=1.0,
    annot_kws={"size": 7},
    cbar_kws={"label": "Achieved Recall", "shrink": 0.8},
    linewidths=0.3,
    linecolor="#333344",
)
dark_ax(
    ax_hr,
    "Lookup Table — Actual Recall Achieved per Bin",
    "RV_Rank Interval",
    "d_ep Interval",
)
ax_hr.set_xticklabels(ax_hr.get_xticklabels(), rotation=45, ha="right", fontsize=6)
ax_hr.set_yticklabels(ax_hr.get_yticklabels(), rotation=0, fontsize=6)
for text in ax_hr.texts:
    try:
        val = float(text.get_text())
        if val < RUNTIME["target_recall"]:
            text.set_color("#ff4444")
            text.set_fontweight("bold")
    except ValueError:
        pass
ax_hr.collections[0].colorbar.ax.tick_params(colors="white")
ax_hr.collections[0].colorbar.set_label("Achieved Recall", color="white")

# ef distribution histogram
ax_efd.bar(ef_vals, ef_cnts, width=1.8, color="#5c9cf5", alpha=0.85, edgecolor="none")
ax_efd.axvline(
    RUNTIME["avg_ef"],
    color="#ff6b6b",
    linewidth=2,
    linestyle="--",
    label=f"avg ef = {RUNTIME['avg_ef']:.0f}",
)
dark_ax(
    ax_efd, "EF Distribution (runtime)", "Assigned EF (with avg-ef floor)", "# Queries"
)
ax_efd.legend(fontsize=9, facecolor="#222233", labelcolor="white", edgecolor="#555566")
ax_efd.tick_params(axis="x", rotation=45)

# metrics table
ax_met.axis("off")
ax_met.set_facecolor("#12151f")
for sp in ax_met.spines.values():
    sp.set_visible(False)
rows = [
    ("Avg Recall", f"{RUNTIME['avg_recall']:.4f}", "#a8e6cf"),
    ("Avg EF Used", f"{RUNTIME['avg_ef']:.1f}", "#5c9cf5"),
    ("5th %ile Recall", f"{RUNTIME['pct5_recall']:.3f}", "#a8e6cf"),
    ("1st %ile Recall", f"{RUNTIME['pct1_recall']:.3f}", "#ffd3b6"),
    ("Search Time", f"{RUNTIME['search_ms'] / 1000:.1f}s", "#d4a5ff"),
    ("# Queries", f"{RUNTIME['nq']:,}", "#cccccc"),
    ("Max EF", f"{RUNTIME['max_ef']:,}", "#cccccc"),
    ("Target Recall", f"{RUNTIME['target_recall']:.2f}", "#ffd3b6"),
]
for i, (label, value, color) in enumerate(rows):
    y = 0.93 - i * 0.115
    ax_met.text(
        0.05,
        y,
        label,
        transform=ax_met.transAxes,
        fontsize=12,
        color="#aaaaaa",
        va="top",
    )
    ax_met.text(
        0.95,
        y,
        value,
        transform=ax_met.transAxes,
        fontsize=13,
        fontweight="bold",
        color=color,
        va="top",
        ha="right",
    )
ax_met.set_title("Runtime Metrics", fontsize=12, color="white", pad=6)

# ═══════════════════════════════════════════════════════════════════════════════
fig1.savefig(OUT_P1, dpi=180, bbox_inches="tight", facecolor=fig1.get_facecolor())
print(f"Page 1 saved: {OUT_P1}")
plt.close(fig1)

# PAGE 2 — Prediction accuracy vs ground-truth ef
# ═══════════════════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(24, 18), facecolor="#0f1117")
fig2.suptitle(
    f"2-Metric Prediction Accuracy vs Ground-Truth EF\n"
    f"{DATASET}  |  n={len(cdf)} queries  |  target recall {RUNTIME['target_recall']:.2f}",
    fontsize=18,
    fontweight="bold",
    color="white",
    y=0.98,
)
gs2 = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.42, wspace=0.36)

ax_pv = fig2.add_subplot(gs2[0, 0])
ax_err = fig2.add_subplot(gs2[0, 1])
ax_rat = fig2.add_subplot(gs2[0, 2])
ax_rv = fig2.add_subplot(gs2[1, 0])
ax_rc = fig2.add_subplot(gs2[1, 1])
ax_sum = fig2.add_subplot(gs2[1, 2])

# predicted vs true scatter
under_mask = cdf["under"]
ax_pv.scatter(
    cdf.loc[~under_mask, "true_ef"],
    cdf.loc[~under_mask, "2metric_ef"],
    c="#5c9cf5",
    s=14,
    alpha=0.55,
    rasterized=True,
    label="over-allocated",
)
ax_pv.scatter(
    cdf.loc[under_mask, "true_ef"],
    cdf.loc[under_mask, "2metric_ef"],
    c="#ff6b6b",
    s=18,
    alpha=0.85,
    rasterized=True,
    label=f"under-predicted ({under_mask.sum()})",
)
mx = max(cdf["true_ef"].max(), cdf["2metric_ef"].max())
ax_pv.plot(
    [0, mx],
    [0, mx],
    color="#ffd166",
    lw=1.5,
    linestyle="--",
    label="perfect prediction",
)
dark_ax(
    ax_pv,
    "2-Metric Predicted EF vs True EF",
    "true_ef (ground truth)",
    "2metric_ef (predicted)",
)
ax_pv.legend(fontsize=9, facecolor="#222233", labelcolor="white", edgecolor="#555566")

# error histogram
bins_e = np.linspace(cdf["error"].min(), cdf["error"].max(), 60)
ax_err.hist(cdf["error"], bins=bins_e, color="#5c9cf5", alpha=0.8, edgecolor="none")
ax_err.axvline(0, color="#ffd166", lw=1.5, linestyle="--")
ax_err.axvline(
    cdf["error"].mean(),
    color="#ff8b94",
    lw=2,
    linestyle="-",
    label=f"mean error = {cdf['error'].mean():+.0f}",
)
ax_err.axvline(
    np.median(cdf["error"]),
    color="#a8e6cf",
    lw=2,
    linestyle="-.",
    label=f"median error = {np.median(cdf['error']):+.0f}",
)
dark_ax(
    ax_err,
    "Prediction Error Distribution\n(2metric_ef − true_ef)",
    "Error (ef units)",
    "# Queries",
)
ax_err.legend(fontsize=9, facecolor="#222233", labelcolor="white", edgecolor="#555566")

# ratio histogram (log scale x)
ratio_clip = cdf["ratio"].clip(upper=20)
ax_rat.hist(ratio_clip, bins=40, color="#a8e6cf", alpha=0.8, edgecolor="none")
ax_rat.axvline(1.0, color="#ff6b6b", lw=2, linestyle="--", label="ratio=1 (perfect)")
ax_rat.axvline(
    cdf["ratio"].median(),
    color="#ffd166",
    lw=2,
    linestyle="-.",
    label=f"median ratio = {cdf['ratio'].median():.1f}×",
)
dark_ax(
    ax_rat,
    "Over-allocation Ratio\n(2metric_ef / true_ef, clipped @20)",
    "2metric_ef / true_ef",
    "# Queries",
)
ax_rat.legend(fontsize=9, facecolor="#222233", labelcolor="white", edgecolor="#555566")

# RV_rank vs error scatter
sc2 = ax_rv.scatter(
    cdf["RV_rank"],
    cdf["error"],
    c=np.log(cdf["true_ef"].clip(lower=1)),
    cmap="plasma",
    s=12,
    alpha=0.6,
    rasterized=True,
)
ax_rv.axhline(0, color="#ffd166", lw=1, linestyle="--")
cb2 = plt.colorbar(sc2, ax=ax_rv)
cb2.set_label("log(true_ef)", color="white")
cb2.ax.tick_params(colors="white")
dark_ax(
    ax_rv,
    "Prediction Error vs RV_Rank\n(color = log true_ef)",
    "RV_Rank",
    "error (2metric − true)",
)

# RC vs error scatter
sc3 = ax_rc.scatter(
    cdf["RC"],
    cdf["error"],
    c=np.log(cdf["true_ef"].clip(lower=1)),
    cmap="plasma",
    s=12,
    alpha=0.6,
    rasterized=True,
)
ax_rc.axhline(0, color="#ffd166", lw=1, linestyle="--")
cb3 = plt.colorbar(sc3, ax=ax_rc)
cb3.set_label("log(true_ef)", color="white")
cb3.ax.tick_params(colors="white")
dark_ax(
    ax_rc,
    "Prediction Error vs RC\n(color = log true_ef)",
    "RC (relative contrast)",
    "error (2metric − true)",
)

# summary stats panel
ax_sum.axis("off")
ax_sum.set_facecolor("#12151f")
for sp in ax_sum.spines.values():
    sp.set_visible(False)
n_under = under_mask.sum()
n_over = (~under_mask & (cdf["2metric_ef"] != cdf["true_ef"])).sum()
n_exact = (cdf["2metric_ef"] == cdf["true_ef"]).sum()
rows2 = [
    ("Sample Size", f"{len(cdf)}", "#cccccc"),
    ("MAE", f"{np.abs(cdf['error']).mean():.1f}", "#ffd3b6"),
    ("Median Error", f"{np.median(cdf['error']):+.0f}", "#ffd3b6"),
    ("Under-predictions", f"{n_under}  ({100 * n_under / len(cdf):.1f}%)", "#ff6b6b"),
    ("Over-predictions", f"{n_over}  ({100 * n_over / len(cdf):.1f}%)", "#5c9cf5"),
    ("Exact hits", f"{n_exact}  ({100 * n_exact / len(cdf):.1f}%)", "#a8e6cf"),
    ("Median ratio", f"{cdf['ratio'].median():.1f}×", "#ffd166"),
    ("95th %ile ratio", f"{np.percentile(cdf['ratio'], 95):.1f}×", "#ffd166"),
    ("Under-est. risk", f"{100 * n_under / len(cdf):.1f}%", "#ff6b6b"),
]
for i, (label, value, color) in enumerate(rows2):
    y = 0.96 - i * 0.105
    ax_sum.text(
        0.05,
        y,
        label,
        transform=ax_sum.transAxes,
        fontsize=12,
        color="#aaaaaa",
        va="top",
    )
    ax_sum.text(
        0.95,
        y,
        value,
        transform=ax_sum.transAxes,
        fontsize=12,
        fontweight="bold",
        color=color,
        va="top",
        ha="right",
    )
ax_sum.set_title("Prediction Accuracy Summary", fontsize=12, color="white", pad=6)

fig2.savefig(OUT_P2, dpi=180, bbox_inches="tight", facecolor=fig2.get_facecolor())
print(f"Page 2 saved: {OUT_P2}")
plt.close(fig2)
