"""
Experiment 4: Slot Duration Sweep

K=5 builders, 24 GCP regions (REGIONS_EXP1), 10 sources (5 high-value, 5
peripheral). Per-source value ratio fixed at 10x (alpha ~ 0.91). Sweeps slot
duration delta on a log scale from 10ms to 12s.

For each delta we draw N_INSTANCES random source layouts and run
N_SEEDS_PER_INSTANCE ABR runs from random initial placements. Reported
curves are median + IQR over all (instance, seed) runs.

Figure layout (2 rows, 3 cols), matching exp1 panels A-F:
Row 1: (A) welfare bars | (B) welfare ratio | (C) geographic HHI
Row 2: (D) utility HHI | (E) high-value coverage | (F) peripheral coverage
"""
import sys
import warnings
import json
import pickle
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from math import comb
from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.exp_helpers import (
    REGIONS_DEFAULT,
    FIGURES_DIR,
    GCP_REGION_COORDS,
    load_propagation_model,
    build_two_cluster_sources,
    make_sliced_prop,
    compute_opt_sliced,
    run_abr_full,
    geo_hhi,
    mean_pairwise_distance_km,
    cluster_coverage_fraction,
)
from sim.simulator import compute_all_builder_utilities
from sim.metrics import hhi as _hhi

REGIONS_EXP1 = list(REGIONS_DEFAULT) + ["europe-west2", "asia-northeast2", "asia-south2", "us-west2"]

# Experiment parameters
MASTER_SEED = 1234
K = 5
TOTAL_VALUE = 10.0

# Fixed value concentration: per-source ratio of 10x.
# With n_high = n_peri = 5, alpha = (10 * 5) / (10 * 5 + 5) = 50/55 ~ 0.909.
VALUE_RATIO = 10.0
# VALUE_RATIO = 7.0 / 3.0

DELTA_GRID_MS = [10, 25, 50, 100, 250, 500, 1000, 3000, 6000, 12000]
DELTA_GRID = np.array([d / 1000.0 for d in DELTA_GRID_MS])
DELTA_ANCHOR = 0.050   # 50 ms (from Constellation MCP proposal)

N_INSTANCES = 5  # random source-layout instances per delta
N_SEEDS_PER_INSTANCE = 3  # random ABR initialisations per instance
N_T = 100
N_T_FINAL = 200
MAX_ROUNDS = 6000
N_HIGH = 5
N_PERI = 5

# Plausible high-value pool: regions near major financial / tech centers
# (same as exp1)
HIGH_VALUE_POOL = [
    "us-east1", "us-east4", "us-central1",
    "europe-west1", "europe-west2", "europe-west3", "europe-west4",
    "europe-north1",
    "asia-northeast1", "asia-northeast2",
    "asia-southeast1",
]

# Plausible peripheral pool: regions geographically distant from the above
# (same as exp1)
PERIPHERAL_POOL = [
    "southamerica-east1", "southamerica-west1",
    "africa-south1",
    "australia-southeast1", "australia-southeast2",
    "asia-south1", "asia-south2",
    "us-west1", "us-west2",
]


def alpha_from_value_ratio(ratio, n_high=N_HIGH, n_peri=N_PERI):
    """Convert per-source high/low value ratio into cluster-level alpha."""
    return float((ratio * n_high) / (ratio * n_high + n_peri))


ALPHA = alpha_from_value_ratio(VALUE_RATIO)

OPT_METHOD = "brute"
BRUTE_FORCE_MAX_PROFILES = 10_000_000  # used only if OPT_METHOD == "auto"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ABR_COLOR = "#1f77b4"
PLAN_COLOR = "#2ca02c"

def sample_source_layout(rng):
    """Return (high_value_regions, peripheral_regions) drawn from the pools."""
    high = list(rng.choice(HIGH_VALUE_POOL, size=N_HIGH, replace=False))
    peri = list(rng.choice(PERIPHERAL_POOL, size=N_PERI, replace=False))
    return high, peri

def _worker(args):
    delta_idx, delta, instance_idx, seed_within_instance = args
    inst_rng = np.random.default_rng(MASTER_SEED + instance_idx)
    high_regions, peri_regions = sample_source_layout(inst_rng)

    regions, prop, region_index_map = load_propagation_model(REGIONS_EXP1)
    sources = build_two_cluster_sources(
        ALPHA, TOTAL_VALUE, region_index_map,
        high_value_regions=high_regions,
        distant_regions=peri_regions,
    )
    sliced_prop = make_sliced_prop(sources, prop)
    n_regions = len(regions)

    # ABR initialisation: deterministic per (instance, seed), not delta dependent
    # so the same initial placements are used across the sweep.
    init_rng = np.random.default_rng(
        MASTER_SEED + 1_000_000 + instance_idx * 10_000 + seed_within_instance
    )
    init_regions = [int(init_rng.integers(0, n_regions)) for _ in range(K)]

    # Seed for ABR's internal randomness (candidate region shuffling)
    abr_seed = (
        MASTER_SEED + 2_000_000 + delta_idx * 100_000
        + instance_idx * 10_000 + seed_within_instance
    )

    result = run_abr_full(
        K, sources, sliced_prop, regions, delta, init_regions, abr_seed,
        n_t=N_T, max_rounds=MAX_ROUNDS, n_t_final=N_T_FINAL,
        n_high_sources=N_HIGH,
    )
    result["delta_idx"] = delta_idx
    result["delta"] = float(delta)
    result["instance_idx"] = instance_idx
    result["seed_within_instance"] = seed_within_instance
    result["high_regions"] = high_regions
    result["peri_regions"] = peri_regions
    return result

def _opt_method_for(K, n_regions):
    if OPT_METHOD != "auto":
        return OPT_METHOD
    n_profiles = comb(n_regions + K - 1, K)
    return "brute" if n_profiles <= BRUTE_FORCE_MAX_PROFILES else "greedy"


def compute_planner_metrics_one(args):
    delta_idx, delta, instance_idx, opt_method = args

    inst_rng = np.random.default_rng(MASTER_SEED + instance_idx)
    high_regions, peri_regions = sample_source_layout(inst_rng)

    regions, prop, region_index_map = load_propagation_model(REGIONS_EXP1)
    sources = build_two_cluster_sources(
        ALPHA, TOTAL_VALUE, region_index_map,
        high_value_regions=high_regions,
        distant_regions=peri_regions,
    )
    sliced_prop = make_sliced_prop(sources, prop)
    n_regions = len(regions)

    w_opt, opt_profile = compute_opt_sliced(
        K, sources, sliced_prop, n_regions, delta,
        n_t=N_T_FINAL, method=opt_method,
    )
    high_ids = list(range(N_HIGH))
    peri_ids = list(range(N_HIGH, N_HIGH + N_PERI))

    planner_utilities = compute_all_builder_utilities(
        opt_profile, sources, sliced_prop, delta, N_T_FINAL,
    )
    planner_util_hhi = float(_hhi(planner_utilities))

    return {
        "delta_idx": delta_idx,
        "delta": float(delta),
        "instance_idx": instance_idx,
        "w_opt": w_opt,
        "opt_profile": opt_profile,
        "geo_hhi_opt": geo_hhi(opt_profile, n_regions),
        "mean_pairwise_km_opt": mean_pairwise_distance_km(opt_profile, list(regions)),
        "utility_hhi_opt": planner_util_hhi,
        "cov_high_opt": cluster_coverage_fraction(
            opt_profile, sources, sliced_prop, delta, high_ids, n_t=N_T_FINAL),
        "cov_peripheral_opt": cluster_coverage_fraction(
            opt_profile, sources, sliced_prop, delta, peri_ids, n_t=N_T_FINAL),
        "high_regions": high_regions,
        "peri_regions": peri_regions,
    }

def _summarize(values):
    """Median, 25th percentile, 75th percentile."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.median(arr)), float(np.percentile(arr, 25)), float(np.percentile(arr, 75))


def save_results(delta_grid_ms, delta_grid, abr_runs_by_delta, planner_runs_by_delta,
                 region_names, n_runs_per_delta, opt_method):
    """Save raw experiment outputs so figures can be inspected/replotted later."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = FIGURES_DIR / f"exp4_slot_duration_results_{ts}"

    payload = {
        "metadata": {
            "experiment": "exp4_slot_duration",
            "timestamp": ts,
            "MASTER_SEED": MASTER_SEED,
            "K": K,
            "VALUE_RATIO": VALUE_RATIO,
            "ALPHA": ALPHA,
            "TOTAL_VALUE": TOTAL_VALUE,
            "DELTA_GRID_MS": list(delta_grid_ms),
            "DELTA_GRID": delta_grid.tolist(),
            "DELTA_ANCHOR": DELTA_ANCHOR,
            "N_INSTANCES": N_INSTANCES,
            "N_SEEDS_PER_INSTANCE": N_SEEDS_PER_INSTANCE,
            "N_T": N_T,
            "N_T_FINAL": N_T_FINAL,
            "MAX_ROUNDS": MAX_ROUNDS,
            "N_HIGH": N_HIGH,
            "N_PERI": N_PERI,
            "OPT_METHOD": OPT_METHOD,
            "opt_method_used": opt_method,
            "n_runs_per_delta": n_runs_per_delta,
            "region_names": list(region_names),
            "high_value_pool": HIGH_VALUE_POOL,
            "peripheral_pool": PERIPHERAL_POOL,
        },
        "abr_runs_by_delta": {
            f"{d:.4f}": abr_runs_by_delta[d] for d in delta_grid
        },
        "planner_runs_by_delta": {
            f"{d:.4f}": planner_runs_by_delta[d] for d in delta_grid
        },
    }

    pkl_path = out_base.with_suffix(".pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f)

    def _json_default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    json_path = out_base.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)

    print(f"Saved raw results:")
    print(f"  {pkl_path}")
    print(f"  {json_path}")


def _set_delta_axis(ax, x_vals_ms):
    """Log-scale delta axis with sensible tick labels."""
    ax.set_xscale("log")
    major_ticks = [10, 50, 100, 1000, 12000]
    major_labels = ["10", "50", "100", "1000", "12000"]
    ax.set_xticks(major_ticks)
    ax.set_xticklabels(major_labels)
    ax.set_xlim(min(x_vals_ms) * 0.85, max(x_vals_ms) * 1.18)
    ax.set_xlabel(r"Slot duration $\Delta$ (ms)")

def plot(delta_grid_ms, delta_grid, abr_runs_by_delta, planner_runs_by_delta, K,
         n_runs_per_delta):
    """abr_runs_by_delta[delta] = list of run dicts.
    planner_runs_by_delta[delta] = list of planner dicts (one per instance)."""

    def _abr_agg(key):
        meds, los, his = [], [], []
        for d in delta_grid:
            vals = [x[key] for x in abr_runs_by_delta[d]]
            m, lo, hi = _summarize(vals)
            meds.append(m); los.append(lo); his.append(hi)
        return np.array(meds), np.array(los), np.array(his)

    def _plan_agg(key):
        meds, los, his = [], [], []
        for d in delta_grid:
            vals = [x[key] for x in planner_runs_by_delta[d]]
            m, lo, hi = _summarize(vals)
            meds.append(m); los.append(lo); his.append(hi)
        return np.array(meds), np.array(los), np.array(his)

    wr_med, wr_lo, wr_hi = [], [], []
    for d in delta_grid:
        w_opt_by_inst = {
            p["instance_idx"]: p["w_opt"] for p in planner_runs_by_delta[d]
        }
        ratios = []
        for r in abr_runs_by_delta[d]:
            w_opt = w_opt_by_inst.get(r["instance_idx"])
            if w_opt is not None and w_opt > 1e-12:
                ratios.append(r["welfare"] / w_opt)
        m, lo, hi = _summarize(ratios)
        wr_med.append(m); wr_lo.append(lo); wr_hi.append(hi)
    wr_med = np.array(wr_med); wr_lo = np.array(wr_lo); wr_hi = np.array(wr_hi)

    welfare_med, welfare_lo, welfare_hi = _abr_agg("welfare")
    welfare_opt_med, welfare_opt_lo, welfare_opt_hi = _plan_agg("w_opt")

    geo_med, geo_lo, geo_hi = _abr_agg("geo_hhi")
    util_med, util_lo, util_hi = _abr_agg("utility_hhi")
    cov_hi_med, cov_hi_lo, cov_hi_hi = _abr_agg("cov_high")
    cov_pe_med, cov_pe_lo, cov_pe_hi = _abr_agg("cov_peripheral")
    mpd_med, mpd_lo, mpd_hi = _abr_agg("mean_pairwise_km")

    geo_opt_med, geo_opt_lo, geo_opt_hi = _plan_agg("geo_hhi_opt")
    cov_hi_opt_med, cov_hi_opt_lo, cov_hi_opt_hi = _plan_agg("cov_high_opt")
    cov_pe_opt_med, cov_pe_opt_lo, cov_pe_opt_hi = _plan_agg("cov_peripheral_opt")
    mpd_opt_med, mpd_opt_lo, mpd_opt_hi = _plan_agg("mean_pairwise_km_opt")

    fig = plt.figure(figsize=(14.5, 13.5))
    gs = fig.add_gridspec(
        nrows=3, ncols=6,
        height_ratios=[1.0, 1.0, 1.0],
        hspace=0.52, wspace=0.65,
    )

    # Row 1: panels A, B, C
    ax_A = fig.add_subplot(gs[0, 0:2])
    ax_B = fig.add_subplot(gs[0, 2:4])
    ax_C = fig.add_subplot(gs[0, 4:6])
    # Row 2: panels D, E, F
    ax_D = fig.add_subplot(gs[1, 0:2])
    ax_E = fig.add_subplot(gs[1, 2:4])
    ax_F = fig.add_subplot(gs[1, 4:6])
    # Row 3: panel G (mean pairwise distance), centered, narrower
    ax_G = fig.add_subplot(gs[2, 1:5])

    x_ms = np.array(delta_grid_ms, dtype=float)
    anchor_x = DELTA_ANCHOR * 1000  # 50

    band_label = "ABR IQR"
    planner_band_label = "Planner IQR"
    abr_marker_kwargs = dict(marker="o", ms=3.5, mec=ABR_COLOR, mfc=ABR_COLOR)
    plan_marker_kwargs = dict(marker="s", ms=3.2, mec=PLAN_COLOR, mfc=PLAN_COLOR)
    anchor_label = f"Anchor {int(anchor_x)} ms"

    # Panel A: welfare (bar chart)
    ax = ax_A
    x_idx = np.arange(len(delta_grid))
    width = 0.38
    ax.bar(
        x_idx - width / 2, welfare_med, width=width, color=ABR_COLOR, alpha=0.75,
        label="ABR median",
        yerr=np.vstack([welfare_med - welfare_lo, welfare_hi - welfare_med]),
        error_kw=dict(ecolor=ABR_COLOR, elinewidth=1.0, capsize=2),
    )
    ax.bar(
        x_idx + width / 2, welfare_opt_med, width=width, color=PLAN_COLOR, alpha=0.55,
        label="Planner median",
        yerr=np.vstack([welfare_opt_med - welfare_opt_lo, welfare_opt_hi - welfare_opt_med]),
        error_kw=dict(ecolor=PLAN_COLOR, elinewidth=1.0, capsize=2),
    )
    bar_tick_positions = list(range(0, len(delta_grid), 2))
    if (len(delta_grid) - 1) not in bar_tick_positions:
        bar_tick_positions.append(len(delta_grid) - 1)
    ax.set_xticks(bar_tick_positions)
    ax.set_xticklabels([f"{delta_grid_ms[i]}" for i in bar_tick_positions])
    anchor_idx = delta_grid_ms.index(int(round(anchor_x)))
    ax.axvspan(anchor_idx - 0.5, anchor_idx + 0.5,
               color="black", alpha=0.06, label=anchor_label)
    ax.set_xlabel(r"Slot duration $\Delta$ (ms)")
    ax.set_ylabel("Expected welfare")
    ax.set_title("(A) Welfare")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    # Panel B: welfare ratio
    ax = ax_B
    ax.plot(x_ms, wr_med, "-", color=ABR_COLOR, lw=1.5, label="ABR median",
            **abr_marker_kwargs)
    ax.fill_between(x_ms, wr_lo, wr_hi, color=ABR_COLOR, alpha=0.18, label=band_label)
    ax.axhline(0.5, ls="--", color="gray", lw=1.0, label="PoA floor = 0.5")
    ax.axhline(1.0, ls=":", color="black", lw=0.8, alpha=0.6)
    ax.axvline(anchor_x, ls=":", color="black", lw=1.0, alpha=0.6, label=anchor_label)
    _set_delta_axis(ax, x_ms)
    ax.set_ylabel(r"$W_{\rm ABR}\,/\,W^*$")
    ax.set_ylim(0.45, 1.05)
    ax.set_title("(B) Welfare ratio")
    ax.legend(fontsize=8, loc="lower left")

    # Panel C: geographic HHI
    ax = ax_C
    ax.plot(x_ms, geo_med, "-", color=ABR_COLOR, lw=1.5, label="ABR median",
            **abr_marker_kwargs)
    ax.fill_between(x_ms, geo_lo, geo_hi, color=ABR_COLOR, alpha=0.18, label=band_label)
    ax.plot(x_ms, geo_opt_med, "-.", color=PLAN_COLOR, lw=1.5, label="Planner median",
            **plan_marker_kwargs)
    ax.fill_between(x_ms, geo_opt_lo, geo_opt_hi, color=PLAN_COLOR, alpha=0.14,
                    label=planner_band_label)
    ax.axhline(1 / K, ls=":", color="black", lw=1.0, alpha=0.6,
               label=rf"Floor $1/K = {1/K:.3f}$")
    ax.axvline(anchor_x, ls=":", color="black", lw=1.0, alpha=0.6)
    _set_delta_axis(ax, x_ms)
    ax.set_ylabel("Geographic HHI")
    ax.set_title("(C) Geographic HHI")
    ax.legend(fontsize=8)

    # Panel D: utility HHI
    ax = ax_D
    ax.plot(x_ms, util_med, "-", color=ABR_COLOR, lw=1.5, label="ABR median",
            **abr_marker_kwargs)
    ax.fill_between(x_ms, util_lo, util_hi, color=ABR_COLOR, alpha=0.18, label=band_label)
    ax.axhline(9 / (8 * K), ls="--", color="gray", lw=1.0,
               label=rf"NE ceiling $9/(8K)$")
    ax.axhline(1 / K, ls=":", color="black", lw=1.0, alpha=0.6,
               label=rf"Egalitarian $1/K$")
    ax.axvline(anchor_x, ls=":", color="black", lw=1.0, alpha=0.6)
    ax.set_ylim(1 / K - 0.005, 9 / (8 * K) + 0.005)
    _set_delta_axis(ax, x_ms)
    ax.set_ylabel("Utility HHI")
    ax.set_title("(D) Utility HHI")
    ax.legend(fontsize=8)

    # Panel E: high-value cluster coverage
    ax = ax_E
    ax.plot(x_ms, cov_hi_med, "-", color=ABR_COLOR, lw=1.5, label="ABR median",
            **abr_marker_kwargs)
    ax.fill_between(x_ms, cov_hi_lo, cov_hi_hi, color=ABR_COLOR, alpha=0.18,
                    label=band_label)
    ax.plot(x_ms, cov_hi_opt_med, "-.", color=PLAN_COLOR, lw=1.5,
            label="Planner median", **plan_marker_kwargs)
    ax.fill_between(x_ms, cov_hi_opt_lo, cov_hi_opt_hi, color=PLAN_COLOR, alpha=0.14,
                    label=planner_band_label)
    ax.axvline(anchor_x, ls=":", color="black", lw=1.0, alpha=0.6)
    _set_delta_axis(ax, x_ms)
    ax.set_ylabel("Coverage fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("(E) High-value cluster coverage")
    ax.legend(fontsize=8, loc="lower right")

    # Panel F: peripheral cluster coverage
    ax = ax_F
    ax.plot(x_ms, cov_pe_med, "-", color=ABR_COLOR, lw=1.5, label="ABR median",
            **abr_marker_kwargs)
    ax.fill_between(x_ms, cov_pe_lo, cov_pe_hi, color=ABR_COLOR, alpha=0.18,
                    label=band_label)
    ax.plot(x_ms, cov_pe_opt_med, "-.", color=PLAN_COLOR, lw=1.5,
            label="Planner median", **plan_marker_kwargs)
    ax.fill_between(x_ms, cov_pe_opt_lo, cov_pe_opt_hi, color=PLAN_COLOR, alpha=0.14,
                    label=planner_band_label)
    ax.axvline(anchor_x, ls=":", color="black", lw=1.0, alpha=0.6)
    _set_delta_axis(ax, x_ms)
    ax.set_ylabel("Coverage fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("(F) Peripheral cluster coverage")
    ax.legend(fontsize=8, loc="lower right")

    # Panel G: mean pairwise distance (km)
    ax = ax_G
    ax.plot(x_ms, mpd_med, "-", color=ABR_COLOR, lw=1.5, label="ABR median",
            **abr_marker_kwargs)
    ax.fill_between(x_ms, mpd_lo, mpd_hi, color=ABR_COLOR, alpha=0.18,
                    label=band_label)
    ax.plot(x_ms, mpd_opt_med, "-.", color=PLAN_COLOR, lw=1.5,
            label="Planner median", **plan_marker_kwargs)
    ax.fill_between(x_ms, mpd_opt_lo, mpd_opt_hi, color=PLAN_COLOR, alpha=0.14,
                    label=planner_band_label)
    ax.axvline(anchor_x, ls=":", color="black", lw=1.0, alpha=0.6, label=anchor_label)
    _set_delta_axis(ax, x_ms)
    ax.set_ylabel("Mean pairwise distance (km)")
    ax.set_title("(G) Mean pairwise distance")
    ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        rf"Experiment 4: Slot duration sweep "
        rf"($K={K}$, value ratio = {VALUE_RATIO:g}x [$\alpha \approx {ALPHA:.3f}$], "
        rf"{N_INSTANCES} source instances $\times$ {N_SEEDS_PER_INSTANCE} inits)",
        fontsize=12, y=0.995,
    )
    out = FIGURES_DIR / "exp4_slot_duration.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=180, bbox_inches="tight")
    print(f"Saved {out}")

def main():
    regions, _, _ = load_propagation_model(REGIONS_EXP1)
    n_regions = len(regions)
    pool_regions = set(HIGH_VALUE_POOL) | set(PERIPHERAL_POOL)
    missing_regions = [r for r in pool_regions if r not in REGIONS_EXP1]
    missing_coords = [r for r in pool_regions if r not in GCP_REGION_COORDS]
    if missing_regions or missing_coords:
        raise ValueError(
            f"Pool regions misconfigured: "
            f"missing from REGIONS_EXP1={missing_regions}, "
            f"missing from GCP_REGION_COORDS={missing_coords}"
        )

    opt_method = _opt_method_for(K, n_regions)
    n_profiles = comb(n_regions + K - 1, K)
    n_runs_per_delta = N_INSTANCES * N_SEEDS_PER_INSTANCE

    print(f"Exp 4: slot duration sweep: {DELTA_GRID_MS} ms")
    print(f"MASTER_SEED={MASTER_SEED}")
    print(f"K={K}, value ratio={VALUE_RATIO:g}x (alpha={ALPHA:.4f})")
    print(f"Per delta: {N_INSTANCES} source instances x {N_SEEDS_PER_INSTANCE} "
          f"random inits = {n_runs_per_delta} ABR runs")
    print(f"Optimum method: {opt_method} (profile count = {n_profiles:,})")
    print(f"Total ABR jobs: {len(DELTA_GRID) * n_runs_per_delta}")
    print(f"Total planner jobs: {len(DELTA_GRID) * N_INSTANCES}")
    print()

    n_workers = max(1, cpu_count() - 1)

    planner_tasks = [
        (di, delta, inst, opt_method)
        for di, delta in enumerate(DELTA_GRID)
        for inst in range(N_INSTANCES)
    ]
    print(f"Computing {len(planner_tasks)} planner benchmarks "
          f"({n_workers} workers) ...")
    planner_runs_by_delta = {d: [] for d in DELTA_GRID}
    with Pool(n_workers) as pool:
        for i, p in enumerate(pool.imap_unordered(compute_planner_metrics_one,
                                                  planner_tasks)):
            d_key = DELTA_GRID[p["delta_idx"]]
            planner_runs_by_delta[d_key].append(p)
            if (i + 1) % 10 == 0 or (i + 1) == len(planner_tasks):
                print(f"  planner [{i+1}/{len(planner_tasks)}] "
                      f"delta={int(p['delta']*1000)}ms inst={p['instance_idx']} "
                      f"W*={p['w_opt']:.4f}")
    abr_tasks = [
        (di, delta, inst, seed)
        for di, delta in enumerate(DELTA_GRID)
        for inst in range(N_INSTANCES)
        for seed in range(N_SEEDS_PER_INSTANCE)
    ]
    print(f"\nRunning {len(abr_tasks)} ABR jobs ({n_workers} workers) ...")
    abr_runs_by_delta = {d: [] for d in DELTA_GRID}
    with Pool(n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, abr_tasks)):
            d_key = DELTA_GRID[r["delta_idx"]]
            abr_runs_by_delta[d_key].append(r)
            if (i + 1) % 25 == 0 or (i + 1) == len(abr_tasks):
                print(f"  abr [{i+1}/{len(abr_tasks)}] "
                      f"delta={int(r['delta']*1000)}ms inst={r['instance_idx']} "
                      f"seed={r['seed_within_instance']} "
                      f"welfare={r['welfare']:.4f}")

    n_truncated = sum(
        1 for runs in abr_runs_by_delta.values()
        for r in runs
        if (r.get("converged") is False
            or (r.get("rounds_used", 0) >= MAX_ROUNDS))
    )
    if n_truncated > 0:
        warnings.warn(
            f"{n_truncated}/{len(abr_tasks)} ABR runs hit MAX_ROUNDS={MAX_ROUNDS} "
            f"without converging. Band width may reflect incomplete convergence."
        )

    save_results(
        DELTA_GRID_MS,
        DELTA_GRID,
        abr_runs_by_delta,
        planner_runs_by_delta,
        region_names=regions,
        n_runs_per_delta=n_runs_per_delta,
        opt_method=opt_method,
    )

    plot(DELTA_GRID_MS, DELTA_GRID, abr_runs_by_delta, planner_runs_by_delta, K,
         n_runs_per_delta)


if __name__ == "__main__":
    main()
