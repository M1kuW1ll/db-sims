"""
Experiment 2 - Builder Count Sweep (with source-instance randomisation)

Fixed alpha=0.9, delta=50ms, 24 GCP regions, 10 sources
(5 high-value, 5 peripheral). Sweeps K from 3 to 12.

For each K we draw N_INSTANCES random source layouts and run
N_SEEDS_PER_INSTANCE ABR runs from random initial placements. Reported
curves are median + IQR over all (instance, seed) ABR runs.

Planner benchmarks are computed once per (K, instance), so planner curves
are also shown as median + IQR over source-layout instances.
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

REGIONS_EXP2 = list(REGIONS_DEFAULT) + ["europe-west2", "asia-northeast2", "asia-south2", "us-west2"]

# Experiment parameters
ALPHA = 0.9
DELTA = 0.05
TOTAL_VALUE = 10.0
K_GRID = list(range(3, 13))  # K = 3 .. 12
N_INSTANCES = 5  # random source-layout instances per K
N_SEEDS_PER_INSTANCE = 3  # random ABR initialisations per instance
N_T = 100
N_T_FINAL = 200
MAX_ROUNDS = 6000
N_HIGH = 6
N_PERI = 6

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

OPT_METHOD = "greedy"
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
    K, instance_idx, seed_within_instance = args

    inst_rng = np.random.default_rng(instance_idx)
    high_regions, peri_regions = sample_source_layout(inst_rng)

    regions, prop, region_index_map = load_propagation_model(REGIONS_EXP2)
    sources = build_two_cluster_sources(
        ALPHA, TOTAL_VALUE, region_index_map,
        high_value_regions=high_regions,
        distant_regions=peri_regions,
    )

    # Source-sliced propagation model: shape is (n_regions, n_sources), so
    # source.id is the correct column index inside ABR and welfare routines.
    sliced_prop = make_sliced_prop(sources, prop)
    n_regions = len(regions)

    # ABR initialisation: deterministic per (K, instance, seed)
    init_rng = np.random.default_rng(K * 1_000_000 + instance_idx * 10_000 + seed_within_instance)
    init_regions = [int(init_rng.integers(0, n_regions)) for _ in range(K)]

    # Seed for ABR's internal randomness (candidate region shuffling)
    abr_seed = K * 1_000_000 + instance_idx * 10_000 + seed_within_instance + 1

    result = run_abr_full(
        K, sources, sliced_prop, regions, DELTA, init_regions, abr_seed,
        n_t=N_T, max_rounds=MAX_ROUNDS, n_t_final=N_T_FINAL,
        n_high_sources=N_HIGH,
    )
    result["K"] = K
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
    K, instance_idx, opt_method = args

    inst_rng = np.random.default_rng(instance_idx)
    high_regions, peri_regions = sample_source_layout(inst_rng)
    regions, prop, region_index_map = load_propagation_model(REGIONS_EXP2)
    sources = build_two_cluster_sources(
        ALPHA, TOTAL_VALUE, region_index_map,
        high_value_regions=high_regions,
        distant_regions=peri_regions,
    )

    # Source-sliced propagation model used by both planner and ABR
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
        "K": K,
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

def save_results(K_grid, abr_runs_by_K, planner_runs_by_K,
                 region_names, n_runs_per_K, opt_method):
    """Save raw experiment outputs so figures can be inspected/replotted later."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = FIGURES_DIR / f"exp2_builder_count_results_{ts}"

    payload = {
        "metadata": {
            "experiment": "exp2_builder_count",
            "timestamp": ts,
            "ALPHA": ALPHA,
            "DELTA": DELTA,
            "TOTAL_VALUE": TOTAL_VALUE,
            "K_GRID": list(K_grid),
            "N_INSTANCES": N_INSTANCES,
            "N_SEEDS_PER_INSTANCE": N_SEEDS_PER_INSTANCE,
            "N_T": N_T,
            "N_T_FINAL": N_T_FINAL,
            "MAX_ROUNDS": MAX_ROUNDS,
            "N_HIGH": N_HIGH,
            "N_PERI": N_PERI,
            "OPT_METHOD": OPT_METHOD,
            "opt_method_used": opt_method,
            "n_runs_per_K": n_runs_per_K,
            "region_names": list(region_names),
            "high_value_pool": HIGH_VALUE_POOL,
            "peripheral_pool": PERIPHERAL_POOL,
        },
        "abr_runs_by_K": {
            str(K): abr_runs_by_K[K] for K in K_grid
        },
        "planner_runs_by_K": {
            str(K): planner_runs_by_K[K] for K in K_grid
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
        size = 30 + 55 * count
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


def plot(K_grid, abr_runs_by_K, planner_runs_by_K, n_runs_per_K, region_names):
    """abr_runs_by_K[K] = list of run dicts.
    planner_runs_by_K[K] = list of planner dicts (one per instance)."""

    def _abr_agg(key):
        meds, los, his = [], [], []
        for K in K_grid:
            vals = [r[key] for r in abr_runs_by_K[K]]
            m, lo, hi = _summarize(vals)
            meds.append(m); los.append(lo); his.append(hi)
        return np.array(meds), np.array(los), np.array(his)

    def _plan_agg(key):
        meds, los, his = [], [], []
        for K in K_grid:
            vals = [r[key] for r in planner_runs_by_K[K]]
            m, lo, hi = _summarize(vals)
            meds.append(m); los.append(lo); his.append(hi)
        return np.array(meds), np.array(los), np.array(his)

    wr_med, wr_lo, wr_hi = [], [], []
    for K in K_grid:
        w_opt_by_inst = {p["instance_idx"]: p["w_opt"] for p in planner_runs_by_K[K]}
        ratios = []
        for r in abr_runs_by_K[K]:
            w_opt = w_opt_by_inst.get(r["instance_idx"])
            if w_opt is not None and w_opt > 1e-12:
                ratios.append(r["welfare"] / w_opt)
        m, lo, hi = _summarize(ratios)
        wr_med.append(m); wr_lo.append(lo); wr_hi.append(hi)
    wr_med = np.array(wr_med); wr_lo = np.array(wr_lo); wr_hi = np.array(wr_hi)

    geo_med, geo_lo, geo_hi = _abr_agg("geo_hhi")
    util_med, util_lo, util_hi = _abr_agg("utility_hhi")
    cov_hi_med, cov_hi_lo, cov_hi_hi = _abr_agg("cov_high")
    cov_pe_med, cov_pe_lo, cov_pe_hi = _abr_agg("cov_peripheral")
    mpd_med, mpd_lo, mpd_hi = _abr_agg("mean_pairwise_km")

    geo_opt_med, geo_opt_lo, geo_opt_hi = _plan_agg("geo_hhi_opt")
    cov_hi_opt_med, cov_hi_opt_lo, cov_hi_opt_hi = _plan_agg("cov_high_opt")
    cov_pe_opt_med, cov_pe_opt_lo, cov_pe_opt_hi = _plan_agg("cov_peripheral_opt")
    mpd_opt_med, mpd_opt_lo, mpd_opt_hi = _plan_agg("mean_pairwise_km_opt")

    x = np.array(K_grid)
    welfare_floor = np.array([1.0 / (2.0 - 1.0 / K) for K in K_grid])
    geo_floor = np.array([1.0 / K for K in K_grid])
    util_ceiling = np.array([9.0 / (8.0 * K) for K in K_grid])
    util_egalitarian = np.array([1.0 / K for K in K_grid])

    band_label = (
        f"ABR IQR "
        f"({N_INSTANCES} source instances × {N_SEEDS_PER_INSTANCE} init)"
    )
    planner_band_label = f"Planner IQR ({N_INSTANCES} source instances)"

    # Layout: top row 2 panels, middle row 3 panels, bottom row 3 panels.
    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(
        nrows=3, ncols=6,
        height_ratios=[1.0, 1.0, 1.05],
        hspace=0.45, wspace=0.55,
    )

    # Row 1: panels A and B
    ax_A = fig.add_subplot(gs[0, 0:3])
    ax_B = fig.add_subplot(gs[0, 3:6])
    # Row 2: panels C, D, E
    ax_C = fig.add_subplot(gs[1, 0:2])
    ax_D = fig.add_subplot(gs[1, 2:4])
    ax_E = fig.add_subplot(gs[1, 4:6])
    # Row 3: mean pairwise distance and two maps
    ax_F = fig.add_subplot(gs[2, 0:2])
    ax_M1 = fig.add_subplot(gs[2, 2:4])
    ax_M2 = fig.add_subplot(gs[2, 4:6])

    # Panel A: welfare ratio
    ax = ax_A
    ax.plot(x, wr_med, "-o", color=ABR_COLOR, lw=1.5, ms=4, label="ABR (median)")
    ax.fill_between(x, wr_lo, wr_hi, color=ABR_COLOR, alpha=0.18, label=band_label)
    ax.plot(x, welfare_floor, "--", color="gray", lw=1.2, label=r"Floor $1/(2-1/K)$")
    ax.axhline(1.0, ls=":", color="black", lw=0.8, alpha=0.6)
    ax.set_xlabel("Number of builders $K$")
    ax.set_ylabel(r"$W_{\rm ABR}\,/\,W^*$")
    ax.set_ylim(0.45, 1.05)
    ax.set_title("(A) Welfare ratio")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # Panel B: geographic HHI
    ax = ax_B
    ax.plot(x, geo_med, "-o", color=ABR_COLOR, lw=1.5, ms=4, label="ABR (median)")
    ax.fill_between(x, geo_lo, geo_hi, color=ABR_COLOR, alpha=0.18, label=band_label)
    ax.plot(x, geo_opt_med, "-.", color=PLAN_COLOR, lw=1.5, label="Planner median")
    ax.fill_between(x, geo_opt_lo, geo_opt_hi, color=PLAN_COLOR, alpha=0.14,
                    label=planner_band_label)
    ax.plot(x, geo_floor, ":", color="black", lw=1.0, alpha=0.7, label=r"Floor $1/K$")
    ax.set_xlabel("Number of builders $K$")
    ax.set_ylabel("Geographic HHI")
    ax.set_title("(B) Geographic HHI")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # Panel C: utility HHI
    ax = ax_C
    ax.plot(x, util_med, "-o", color=ABR_COLOR, lw=1.5, ms=4, label="ABR (median)")
    ax.fill_between(x, util_lo, util_hi, color=ABR_COLOR, alpha=0.18, label=band_label)
    ax.plot(x, util_ceiling, "--", color="gray", lw=1.2, label=r"NE ceiling $9/(8K)$")
    ax.plot(x, util_egalitarian, ":", color="black", lw=1.0, alpha=0.7,
            label=r"Egalitarian $1/K$")
    ax.set_xlabel("Number of builders $K$")
    ax.set_ylabel("Utility HHI")
    ax.set_title("(C) Utility HHI")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # Panel D: high-value cluster coverage
    ax = ax_D
    ax.plot(x, cov_hi_med, "-o", color=ABR_COLOR, lw=1.5, ms=4, label="ABR (median)")
    ax.fill_between(x, cov_hi_lo, cov_hi_hi, color=ABR_COLOR, alpha=0.18, label=band_label)
    ax.plot(x, cov_hi_opt_med, "-.", color=PLAN_COLOR, lw=1.5, label="Planner median")
    ax.fill_between(x, cov_hi_opt_lo, cov_hi_opt_hi, color=PLAN_COLOR, alpha=0.14,
                    label=planner_band_label)
    ax.set_xlabel("Number of builders $K$")
    ax.set_ylabel("Coverage fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("(D) High-value cluster coverage")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.2)

    # Panel E: peripheral cluster coverage
    ax = ax_E
    ax.plot(x, cov_pe_med, "-o", color=ABR_COLOR, lw=1.5, ms=4, label="ABR (median)")
    ax.fill_between(x, cov_pe_lo, cov_pe_hi, color=ABR_COLOR, alpha=0.18, label=band_label)
    ax.plot(x, cov_pe_opt_med, "-.", color=PLAN_COLOR, lw=1.5, label="Planner median")
    ax.fill_between(x, cov_pe_opt_lo, cov_pe_opt_hi, color=PLAN_COLOR, alpha=0.14,
                    label=planner_band_label)
    ax.set_xlabel("Number of builders $K$")
    ax.set_ylabel("Coverage fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("(E) Peripheral cluster coverage")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.2)

    # Panel F: mean pairwise distance
    ax = ax_F
    ax.plot(x, mpd_med, "-o", color=ABR_COLOR, lw=1.5, ms=4, label="ABR (median)")
    ax.fill_between(x, mpd_lo, mpd_hi, color=ABR_COLOR, alpha=0.18, label=band_label)
    ax.plot(x, mpd_opt_med, "-.", color=PLAN_COLOR, lw=1.5, label="Planner median")
    ax.fill_between(x, mpd_opt_lo, mpd_opt_hi, color=PLAN_COLOR, alpha=0.14,
                    label=planner_band_label)
    ax.set_xlabel("Number of builders $K$")
    ax.set_ylabel("Mean pairwise distance (km)")
    ax.set_title("(F) Mean pairwise distance")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # Bottom row: world maps at smallest and largest K with data.
    Ks_with_data = [K for K in K_grid if abr_runs_by_K.get(K)]
    if len(Ks_with_data) >= 2:
        map_Ks = [Ks_with_data[0], Ks_with_data[-1]]
    elif len(Ks_with_data) == 1:
        map_Ks = [Ks_with_data[0], Ks_with_data[0]]
    else:
        map_Ks = []

    for ax_map, K in zip([ax_M1, ax_M2], map_Ks):
        profiles = [r["final_profile"] for r in abr_runs_by_K.get(K, [])]
        counts = _builder_counts(profiles, region_names)
        _plot_world_map(ax_map, counts, region_names, K,
                        title=rf"$K = {K}$  (mean ABR placement)")

    fig.suptitle(
        rf"Experiment 2: Builder count sweep "
        rf"($\alpha={ALPHA}$, $\Delta={int(DELTA*1000)}\,\mathrm{{ms}}$, "
        rf"{N_INSTANCES} source instances $\times$ {N_SEEDS_PER_INSTANCE} inits)",
        fontsize=12, y=0.995,
    )
    out = FIGURES_DIR / "exp2_builder_count.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=180, bbox_inches="tight")
    print(f"Saved {out}")


def main():
    regions, _, _ = load_propagation_model(REGIONS_EXP2)
    n_regions = len(regions)

    pool_regions = set(HIGH_VALUE_POOL) | set(PERIPHERAL_POOL)
    missing_regions = [r for r in pool_regions if r not in REGIONS_EXP2]
    missing_coords = [r for r in pool_regions if r not in GCP_REGION_COORDS]
    if missing_regions or missing_coords:
        raise ValueError(
            f"Pool regions misconfigured: "
            f"missing from REGIONS_EXP2={missing_regions}, "
            f"missing from GCP_REGION_COORDS={missing_coords}"
        )

    opt_method = _opt_method_for(max(K_GRID), n_regions)
    n_runs_per_K = N_INSTANCES * N_SEEDS_PER_INSTANCE

    print(f"Exp 2: K sweep: {K_GRID}")
    print(f"alpha={ALPHA}, delta={DELTA*1000:.0f} ms")
    print(f"Per K: {N_INSTANCES} source instances x {N_SEEDS_PER_INSTANCE} "
          f"random inits = {n_runs_per_K} ABR runs")
    print(f"Optimum method: {opt_method}")
    print(f"Total ABR jobs: {len(K_GRID) * n_runs_per_K}")
    print(f"Total planner jobs: {len(K_GRID) * N_INSTANCES}")
    print()

    n_workers = max(1, cpu_count() - 1)

    # Planner benchmarks (one per (K, instance))
    planner_tasks = [
        (K, inst, opt_method)
        for K in K_GRID
        for inst in range(N_INSTANCES)
    ]
    print(f"Computing {len(planner_tasks)} planner benchmarks "
          f"({n_workers} workers) ...")
    planner_runs_by_K = {K: [] for K in K_GRID}
    with Pool(n_workers) as pool:
        for i, p in enumerate(pool.imap_unordered(compute_planner_metrics_one,
                                                    planner_tasks)):
            planner_runs_by_K[p["K"]].append(p)
            if (i + 1) % 10 == 0 or (i + 1) == len(planner_tasks):
                print(f"  planner [{i+1}/{len(planner_tasks)}] "
                      f"K={p['K']} inst={p['instance_idx']} "
                      f"W*={p['w_opt']:.4f}")
    abr_tasks = [
        (K, inst, seed)
        for K in K_GRID
        for inst in range(N_INSTANCES)
        for seed in range(N_SEEDS_PER_INSTANCE)
    ]
    print(f"\nRunning {len(abr_tasks)} ABR jobs ({n_workers} workers) ...")
    abr_runs_by_K = {K: [] for K in K_GRID}
    with Pool(n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, abr_tasks)):
            abr_runs_by_K[r["K"]].append(r)
            if (i + 1) % 25 == 0 or (i + 1) == len(abr_tasks):
                print(f"  abr [{i+1}/{len(abr_tasks)}] "
                      f"K={r['K']} inst={r['instance_idx']} "
                      f"seed={r['seed_within_instance']} "
                      f"welfare={r['welfare']:.4f}")
    n_truncated = sum(
        1 for runs in abr_runs_by_K.values()
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
    for K in K_GRID:
        profiles = [r["final_profile"] for r in abr_runs_by_K[K]]
        unique_counts = [len(set(p)) for p in profiles]
        geo_vals = [r["geo_hhi"] for r in abr_runs_by_K[K]]
        print(
            f"  K={K}: "
            f"geo_hhi={geo_vals}, "
            f"unique_builder_regions={unique_counts}, "
            f"profiles={profiles}"
        )

    save_results(
        K_GRID,
        abr_runs_by_K,
        planner_runs_by_K,
        region_names=regions,
        n_runs_per_K=n_runs_per_K,
        opt_method=opt_method,
    )

    plot(K_GRID, abr_runs_by_K, planner_runs_by_K,
         n_runs_per_K, region_names=regions)


if __name__ == "__main__":
    main()
