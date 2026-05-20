"""
Experiment 1: Value Asymmetry Sweep (with source-instance randomization)

K=5 builders, 24 GCP regions, 10 sources (5 high-value, 5 low-value).
Sweeps the per-source value ratio between high-value and low-value sources.
Internally, this is converted to alpha: the fraction of total emitted value
assigned to the high-value cluster.

For each ratio we draw N_INSTANCES random source layouts and run
N_SEEDS_PER_INSTANCE ABR runs from random initial placements. Reported
curves are median + IQR over all (instance, seed) runs.
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
DELTA = 0.05
TOTAL_VALUE = 10.0

VALUE_RATIO_GRID = np.array([1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 5.0, 10.0, 20.0])

N_INSTANCES = 5  # random source-layout instances per ratio
N_SEEDS_PER_INSTANCE = 3   # random ABR initialisations per instance
N_T = 100
N_T_FINAL = 200
MAX_ROUNDS = 6000
N_HIGH = 5
N_PERI = 5


def alpha_from_value_ratio(ratio, n_high=N_HIGH, n_peri=N_PERI):
    """Convert per-source high/low value ratio into cluster-level alpha."""
    return float((ratio * n_high) / (ratio * n_high + n_peri))


ALPHA_GRID = np.array([alpha_from_value_ratio(r) for r in VALUE_RATIO_GRID])


def _format_ratio_label(r):
    if abs(r - round(r)) < 1e-9:
        return f"{int(round(r))}x"
    return f"{r:g}x"


VALUE_RATIO_LABELS = [_format_ratio_label(r) for r in VALUE_RATIO_GRID]

# Plausible high-value pool: regions near major financial / tech centers
HIGH_VALUE_POOL = [
    "us-east1", "us-east4", "us-central1",
    "europe-west1", "europe-west2", "europe-west3", "europe-west4",
    "europe-north1",
    "asia-northeast1", "asia-northeast2",
    "asia-southeast1",
]

# Plausible peripheral pool: regions geographically distant from the above
PERIPHERAL_POOL = [
    "southamerica-east1", "southamerica-west1",
    "africa-south1",
    "australia-southeast1", "australia-southeast2",
    "asia-south1", "asia-south2",
    "us-west1", "us-west2",
]

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


# Worker: run one (ratio_idx, ratio, alpha, instance, seed) job

def _worker(args):
    ratio_idx, value_ratio, alpha, instance_idx, seed_within_instance = args
    inst_rng = np.random.default_rng(MASTER_SEED + instance_idx)
    high_regions, peri_regions = sample_source_layout(inst_rng)

    regions, prop, region_index_map = load_propagation_model(REGIONS_EXP1)
    sources = build_two_cluster_sources(
        alpha, TOTAL_VALUE, region_index_map,
        high_value_regions=high_regions,
        distant_regions=peri_regions,
    )
    sliced_prop = make_sliced_prop(sources, prop)
    n_regions = len(regions)

    # ABR initialisation: deterministic for each (instance, seed), intentionally not
    # ratio-dependent so the same initial placements are used across the sweep.
    init_rng = np.random.default_rng(
        MASTER_SEED + 1_000_000 + instance_idx * 10_000 + seed_within_instance
    )
    init_regions = [int(init_rng.integers(0, n_regions)) for _ in range(K)]

    # Seed for ABR's internal randomness (candidate region shuffling).
    abr_seed = (
        MASTER_SEED + 2_000_000 + ratio_idx * 100_000
        + instance_idx * 10_000 + seed_within_instance
    )

    result = run_abr_full(
        K, sources, sliced_prop, regions, DELTA, init_regions, abr_seed,
        n_t=N_T, max_rounds=MAX_ROUNDS, n_t_final=N_T_FINAL,
        n_high_sources=N_HIGH,
    )
    result["ratio_idx"] = ratio_idx
    result["value_ratio"] = value_ratio
    result["alpha"] = alpha
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
    ratio_idx, value_ratio, alpha, instance_idx, opt_method = args

    inst_rng = np.random.default_rng(MASTER_SEED + instance_idx)
    high_regions, peri_regions = sample_source_layout(inst_rng)

    regions, prop, region_index_map = load_propagation_model(REGIONS_EXP1)
    sources = build_two_cluster_sources(
        alpha, TOTAL_VALUE, region_index_map,
        high_value_regions=high_regions,
        distant_regions=peri_regions,
    )
    sliced_prop = make_sliced_prop(sources, prop)
    n_regions = len(regions)

    w_opt, opt_profile = compute_opt_sliced(
        K, sources, sliced_prop, n_regions, DELTA,
        n_t=N_T_FINAL, method=opt_method,
    )
    high_ids = list(range(N_HIGH))
    peri_ids = list(range(N_HIGH, N_HIGH + N_PERI))

    planner_utilities = compute_all_builder_utilities(
        opt_profile, sources, sliced_prop, DELTA, N_T_FINAL,
    )
    planner_util_hhi = float(_hhi(planner_utilities))

    return {
        "ratio_idx": ratio_idx,
        "value_ratio": value_ratio,
        "alpha": alpha,
        "instance_idx": instance_idx,
        "w_opt": w_opt,
        "opt_profile": opt_profile,
        "geo_hhi_opt": geo_hhi(opt_profile, n_regions),
        "mean_pairwise_km_opt": mean_pairwise_distance_km(opt_profile, list(regions)),
        "utility_hhi_opt": planner_util_hhi,
        "cov_high_opt": cluster_coverage_fraction(
            opt_profile, sources, sliced_prop, DELTA, high_ids, n_t=N_T_FINAL),
        "cov_peripheral_opt": cluster_coverage_fraction(
            opt_profile, sources, sliced_prop, DELTA, peri_ids, n_t=N_T_FINAL),
        "high_regions": high_regions,
        "peri_regions": peri_regions,
    }

def _summarize(values):
    """Median, 25th percentile, 75th percentile."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.median(arr)), float(np.percentile(arr, 25)), float(np.percentile(arr, 75))


def save_results(value_ratio_grid, alpha_grid, abr_runs_by_ratio, planner_runs_by_ratio,
                 region_names, n_runs_per_ratio, opt_method):
    """Save raw experiment outputs so figures can be inspected/replotted later."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = FIGURES_DIR / f"exp1_value_asymmetry_results_{ts}"

    payload = {
        "metadata": {
            "experiment": "exp1_value_asymmetry",
            "timestamp": ts,
            "MASTER_SEED": MASTER_SEED,
            "K": K,
            "DELTA": DELTA,
            "TOTAL_VALUE": TOTAL_VALUE,
            "VALUE_RATIO_GRID": value_ratio_grid.tolist(),
            "VALUE_RATIO_LABELS": VALUE_RATIO_LABELS,
            "ALPHA_GRID": alpha_grid.tolist(),
            "N_INSTANCES": N_INSTANCES,
            "N_SEEDS_PER_INSTANCE": N_SEEDS_PER_INSTANCE,
            "N_T": N_T,
            "N_T_FINAL": N_T_FINAL,
            "MAX_ROUNDS": MAX_ROUNDS,
            "N_HIGH": N_HIGH,
            "N_PERI": N_PERI,
            "OPT_METHOD": OPT_METHOD,
            "opt_method_used": opt_method,
            "n_runs_per_ratio": n_runs_per_ratio,
            "region_names": list(region_names),
            "high_value_pool": HIGH_VALUE_POOL,
            "peripheral_pool": PERIPHERAL_POOL,
            "value_ratio_definition": "per-source high-cluster expected value divided by per-source peripheral-cluster expected value",
        },
        "abr_runs_by_ratio": {
            str(r): abr_runs_by_ratio[r] for r in value_ratio_grid
        },
        "planner_runs_by_ratio": {
            str(r): planner_runs_by_ratio[r] for r in value_ratio_grid
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

def _plot_world_map(ax, builder_counts_by_region, region_names, K,
                    title, region_coords=GCP_REGION_COORDS):
    """Equirectangular world map with markers sized by mean builder count.

    builder_counts_by_region: dict {region_name: mean_count_over_runs}
    """
    ax.set_xlim(-180, 180)
    ax.set_ylim(-65, 80)
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
    ax.set_yticks([-60, -30, 0, 30, 60])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, alpha=0.25, lw=0.5, ls=":")
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(left=False, bottom=False)

    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color("black")

    for r_name in region_names:
        if r_name not in region_coords:
            continue
        lat, lon = region_coords[r_name]
        ax.scatter(lon, lat, s=8, c="#cccccc", zorder=2, edgecolors="none")

    for r_name, count in builder_counts_by_region.items():
        if count <= 0 or r_name not in region_coords:
            continue
        lat, lon = region_coords[r_name]
        size = 30 + 80 * count
        ax.scatter(lon, lat, s=size, c=ABR_COLOR, alpha=0.65,
                   edgecolors="white", linewidths=0.6, zorder=3)

    ax.set_title(title, fontsize=10)


def _builder_counts(profiles, region_names):
    """Aggregate a list of converged profiles into mean per-region builder counts."""
    counts = {r: 0.0 for r in region_names}
    n_profiles = len(profiles)
    if n_profiles == 0:
        return counts
    for profile in profiles:
        for r_idx in profile:
            counts[region_names[r_idx]] += 1.0 / n_profiles
    return counts


def _set_ratio_axis(ax, x_vals):
    major_ticks = [1.0, 2.0, 5.0, 10.0, 20.0]
    major_labels = ["1x", "2x", "5x", "10x", "20x"]

    ax.set_xscale("log")
    ax.set_xticks(major_ticks)
    ax.set_xticklabels(major_labels)
    ax.set_xlim(min(x_vals) * 0.92, max(x_vals) * 1.08)
    ax.set_xlabel("Expected per-source value ratio (high / low)")

def plot(value_ratio_grid, abr_runs_by_ratio, planner_runs_by_ratio, K,
         n_runs_per_ratio, region_names):
    """abr_runs_by_ratio[ratio] = list of run dicts.
    planner_runs_by_ratio[ratio] = list of planner dicts (one per instance)."""

    def _abr_agg(key):
        meds, los, his = [], [], []
        for r in value_ratio_grid:
            vals = [x[key] for x in abr_runs_by_ratio[r]]
            m, lo, hi = _summarize(vals)
            meds.append(m); los.append(lo); his.append(hi)
        return np.array(meds), np.array(los), np.array(his)

    def _plan_agg(key):
        meds, los, his = [], [], []
        for r in value_ratio_grid:
            vals = [x[key] for x in planner_runs_by_ratio[r]]
            m, lo, hi = _summarize(vals)
            meds.append(m); los.append(lo); his.append(hi)
        return np.array(meds), np.array(los), np.array(his)


    wr_med, wr_lo, wr_hi = [], [], []
    for ratio in value_ratio_grid:
        w_opt_by_inst = {
            p["instance_idx"]: p["w_opt"] for p in planner_runs_by_ratio[ratio]
        }
        ratios = []
        for r in abr_runs_by_ratio[ratio]:
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

    geo_opt_med, geo_opt_lo, geo_opt_hi = _plan_agg("geo_hhi_opt")
    cov_hi_opt_med, cov_hi_opt_lo, cov_hi_opt_hi = _plan_agg("cov_high_opt")
    cov_pe_opt_med, cov_pe_opt_lo, cov_pe_opt_hi = _plan_agg("cov_peripheral_opt")

    # Layout: top row 3 panels, middle row 3 panels, bottom row 2 maps.
    fig = plt.figure(figsize=(14.5, 11))
    gs = fig.add_gridspec(
        nrows=3, ncols=6,
        height_ratios=[1.0, 1.0, 1.05],
        hspace=0.52, wspace=0.65,
    )

    # Row 1: panels A, B, C (cols 0-1, 2-3, 4-5)
    ax_A = fig.add_subplot(gs[0, 0:2])
    ax_B = fig.add_subplot(gs[0, 2:4])
    ax_C = fig.add_subplot(gs[0, 4:6])
    # Row 2: panels D, E, F (cols 0-1, 2-3, 4-5)
    ax_D = fig.add_subplot(gs[1, 0:2])
    ax_E = fig.add_subplot(gs[1, 2:4])
    ax_F = fig.add_subplot(gs[1, 4:6])
    # Row 3: world maps at smallest and largest ratio (cols 0-2, 3-5)
    ax_M1 = fig.add_subplot(gs[2, 0:3])
    ax_M2 = fig.add_subplot(gs[2, 3:6])

    x = value_ratio_grid
    x_idx = np.arange(len(value_ratio_grid))
    band_label = "ABR IQR"
    planner_band_label = "Planner IQR"
    abr_marker_kwargs = dict(marker="o", ms=3.5, mec=ABR_COLOR, mfc=ABR_COLOR)
    plan_marker_kwargs = dict(marker="s", ms=3.2, mec=PLAN_COLOR, mfc=PLAN_COLOR)

    # Panel A: welfare (bar chart)
    ax = ax_A
    width = 0.36
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
    bar_tick_positions = [
        i for i, r in enumerate(value_ratio_grid)
        if r in {1.0, 2.0, 5.0, 10.0, 20.0}
    ]
    bar_tick_labels = [
        _format_ratio_label(value_ratio_grid[i]) for i in bar_tick_positions
    ]
    ax.set_xticks(bar_tick_positions)
    ax.set_xticklabels(bar_tick_labels)
    ax.set_xlabel("Expected per-source value ratio (high / low)")
    ax.set_ylabel("Expected welfare")
    ax.set_title("(A) Welfare")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    # Panel B: welfare ratio
    ax = ax_B
    ax.plot(x, wr_med, "-", color=ABR_COLOR, lw=1.5, label="ABR median", **abr_marker_kwargs)
    ax.fill_between(x, wr_lo, wr_hi, color=ABR_COLOR, alpha=0.18, label=band_label)
    ax.axhline(0.5, ls="--", color="gray", lw=1.0, label="PoA floor = 0.5")
    ax.axhline(1.0, ls=":", color="black", lw=0.8, alpha=0.6)
    _set_ratio_axis(ax, x)
    ax.set_ylabel(r"$W_{\rm ABR}\,/\,W^*$")
    ax.set_ylim(0.45, 1.05)
    ax.set_title("(B) Welfare ratio")
    ax.legend(fontsize=8, loc="lower left")

    # Panel C: geographic HHI
    ax = ax_C
    ax.plot(x, geo_med, "-", color=ABR_COLOR, lw=1.5, label="ABR median", **abr_marker_kwargs)
    ax.fill_between(x, geo_lo, geo_hi, color=ABR_COLOR, alpha=0.18, label=band_label)
    ax.plot(x, geo_opt_med, "-.", color=PLAN_COLOR, lw=1.5, label="Planner median",
            **plan_marker_kwargs)
    ax.fill_between(x, geo_opt_lo, geo_opt_hi, color=PLAN_COLOR, alpha=0.14,
                    label=planner_band_label)
    ax.axhline(1 / K, ls=":", color="black", lw=1.0, alpha=0.6,
               label=rf"Floor $1/K = {1/K:.3f}$")
    _set_ratio_axis(ax, x)
    ax.set_ylabel("Geographic HHI")
    ax.set_title("(C) Geographic HHI")
    ax.legend(fontsize=8)

    # Panel D: utility HHI
    ax = ax_D
    ax.plot(x, util_med, "-", color=ABR_COLOR, lw=1.5, label="ABR median", **abr_marker_kwargs)
    ax.fill_between(x, util_lo, util_hi, color=ABR_COLOR, alpha=0.18, label=band_label)
    ax.axhline(9 / (8 * K), ls="--", color="gray", lw=1.0,
               label=rf"NE ceiling $9/(8K)$")
    ax.axhline(1 / K, ls=":", color="black", lw=1.0, alpha=0.6,
               label=rf"Egalitarian $1/K$")
    ax.set_ylim(1 / K - 0.005, 9 / (8 * K) + 0.005)
    _set_ratio_axis(ax, x)
    ax.set_ylabel("Utility HHI")
    ax.set_title("(D) Utility HHI")
    ax.legend(fontsize=8)

    # Panel E: high-value cluster coverage
    ax = ax_E
    ax.plot(x, cov_hi_med, "-", color=ABR_COLOR, lw=1.5, label="ABR median", **abr_marker_kwargs)
    ax.fill_between(x, cov_hi_lo, cov_hi_hi, color=ABR_COLOR, alpha=0.18, label=band_label)
    ax.plot(x, cov_hi_opt_med, "-.", color=PLAN_COLOR, lw=1.5, label="Planner median",
            **plan_marker_kwargs)
    ax.fill_between(x, cov_hi_opt_lo, cov_hi_opt_hi, color=PLAN_COLOR, alpha=0.14,
                    label=planner_band_label)
    _set_ratio_axis(ax, x)
    ax.set_ylabel("Coverage fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("(E) High-value cluster coverage")
    ax.legend(fontsize=8, loc="lower right")

    # Panel F: peripheral cluster coverage
    ax = ax_F
    ax.plot(x, cov_pe_med, "-", color=ABR_COLOR, lw=1.5, label="ABR median", **abr_marker_kwargs)
    ax.fill_between(x, cov_pe_lo, cov_pe_hi, color=ABR_COLOR, alpha=0.18, label=band_label)
    ax.plot(x, cov_pe_opt_med, "-.", color=PLAN_COLOR, lw=1.5,
            label="Planner median", **plan_marker_kwargs)
    ax.fill_between(x, cov_pe_opt_lo, cov_pe_opt_hi, color=PLAN_COLOR, alpha=0.14,
                    label=planner_band_label)
    _set_ratio_axis(ax, x)
    ax.set_ylabel("Coverage fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("(F) Peripheral cluster coverage")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    # Bottom row: world maps at smallest and largest ratio with data
    ratios_with_data = [r for r in value_ratio_grid if abr_runs_by_ratio.get(r)]
    if len(ratios_with_data) >= 2:
        map_ratios = [ratios_with_data[0], ratios_with_data[-1]]
    elif len(ratios_with_data) == 1:
        map_ratios = [ratios_with_data[0], ratios_with_data[0]]
    else:
        map_ratios = []

    for ax_map, ratio in zip([ax_M1, ax_M2], map_ratios):
        profiles = [r["final_profile"] for r in abr_runs_by_ratio.get(ratio, [])]
        counts = _builder_counts(profiles, region_names)
        _plot_world_map(ax_map, counts, region_names, K,
                        title=rf"high/low value ratio = {ratio:g}x  (mean ABR placement)")

    fig.suptitle(
        rf"Experiment 1: Value asymmetry sweep "
        rf"($K={K}$, $\Delta={int(DELTA*1000)}\,\mathrm{{ms}}$, "
        rf"{N_INSTANCES} source instances $\times$ {N_SEEDS_PER_INSTANCE} inits)",
        fontsize=12, y=0.995,
    )
    out = FIGURES_DIR / "exp1_value_asymmetry.pdf"
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
    n_runs_per_ratio = N_INSTANCES * N_SEEDS_PER_INSTANCE

    print(f"Exp 1: per-source value ratio sweep: {VALUE_RATIO_GRID}")
    print(f"Corresponding alpha values: {ALPHA_GRID}")
    print(f"MASTER_SEED={MASTER_SEED}")
    print(f"K={K}, delta={DELTA*1000:.0f} ms")
    print(f"Per ratio: {N_INSTANCES} source instances x {N_SEEDS_PER_INSTANCE} "
          f"random inits = {n_runs_per_ratio} ABR runs")
    print(f"Optimum method: {opt_method} (profile count = {n_profiles:,})")
    print(f"Total ABR jobs: {len(VALUE_RATIO_GRID) * n_runs_per_ratio}")
    print(f"Total planner jobs: {len(VALUE_RATIO_GRID) * N_INSTANCES}")
    print()

    n_workers = max(1, cpu_count() - 1)

    planner_tasks = [
        (i, ratio, ALPHA_GRID[i], inst, opt_method)
        for i, ratio in enumerate(VALUE_RATIO_GRID)
        for inst in range(N_INSTANCES)
    ]
    print(f"Computing {len(planner_tasks)} planner benchmarks "
          f"({n_workers} workers) ...")
    planner_runs_by_ratio = {r: [] for r in VALUE_RATIO_GRID}
    with Pool(n_workers) as pool:
        for i, p in enumerate(pool.imap_unordered(compute_planner_metrics_one,
                                                  planner_tasks)):
            r_key = VALUE_RATIO_GRID[p["ratio_idx"]]
            planner_runs_by_ratio[r_key].append(p)
            if (i + 1) % 10 == 0 or (i + 1) == len(planner_tasks):
                print(f"  planner [{i+1}/{len(planner_tasks)}] "
                      f"ratio={p['value_ratio']:.2f}x alpha={p['alpha']:.3f} "
                      f"inst={p['instance_idx']} W*={p['w_opt']:.4f}")

    abr_tasks = [
        (i, ratio, ALPHA_GRID[i], inst, seed)
        for i, ratio in enumerate(VALUE_RATIO_GRID)
        for inst in range(N_INSTANCES)
        for seed in range(N_SEEDS_PER_INSTANCE)
    ]
    print(f"\nRunning {len(abr_tasks)} ABR jobs ({n_workers} workers) ...")
    abr_runs_by_ratio = {r: [] for r in VALUE_RATIO_GRID}
    with Pool(n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, abr_tasks)):
            r_key = VALUE_RATIO_GRID[r["ratio_idx"]]
            abr_runs_by_ratio[r_key].append(r)
            if (i + 1) % 25 == 0 or (i + 1) == len(abr_tasks):
                print(f"  abr [{i+1}/{len(abr_tasks)}] "
                      f"ratio={r['value_ratio']:.2f}x alpha={r['alpha']:.3f} "
                      f"inst={r['instance_idx']} "
                      f"seed={r['seed_within_instance']} "
                      f"welfare={r['welfare']:.4f}")

    n_truncated = sum(
        1 for runs in abr_runs_by_ratio.values()
        for r in runs
        if (r.get("converged") is False
            or (r.get("rounds_used", 0) >= MAX_ROUNDS))
    )
    if n_truncated > 0:
        warnings.warn(
            f"{n_truncated}/{len(abr_tasks)} ABR runs hit MAX_ROUNDS={MAX_ROUNDS} "
            f"without converging. Band width may reflect incomplete convergence."
        )

    print("\nGeographic HHI diagnostic:")
    for ratio in VALUE_RATIO_GRID:
        profiles = [r["final_profile"] for r in abr_runs_by_ratio[ratio]]
        unique_counts = [len(set(p)) for p in profiles]
        geo_vals = [r["geo_hhi"] for r in abr_runs_by_ratio[ratio]]
        print(
            f"  ratio={ratio:.2f}x alpha={alpha_from_value_ratio(ratio):.3f}: "
            f"geo_hhi={geo_vals}, "
            f"unique_builder_regions={unique_counts}, "
            f"profiles={profiles}"
        )

    save_results(
        VALUE_RATIO_GRID,
        ALPHA_GRID,
        abr_runs_by_ratio,
        planner_runs_by_ratio,
        region_names=regions,
        n_runs_per_ratio=n_runs_per_ratio,
        opt_method=opt_method,
    )

    plot(VALUE_RATIO_GRID, abr_runs_by_ratio, planner_runs_by_ratio, K,
         n_runs_per_ratio, region_names=regions)


if __name__ == "__main__":
    main()
