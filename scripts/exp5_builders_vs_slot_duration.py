"""
Experiment 5: Heatmaps over (number of builders K, delta)

Sweeps number of builders K on the y-axis and slot duration delta on the
x-axis. Value concentration is fixed at ratio = 10x (alpha ~ 0.91).

Each cell is the median over (N_INSTANCES random source layouts) x
(N_SEEDS_PER_INSTANCE random ABR initialisations). Results are cached so
plotting can be iterated without re-running the simulation.
"""
import argparse
import hashlib
import json
import sys
import warnings
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
    max_mean_pairwise_distance_km,
    placement_emd_km,
    placement_multiset_jaccard,
)

REGIONS_EXP = list(REGIONS_DEFAULT) + ["europe-west2", "asia-northeast2", "asia-south2", "us-west2"]

# Experiment parameters
MASTER_SEED = 1234
TOTAL_VALUE = 10.0

# Fixed value concentration: per-source ratio of 10x.
# With n_high = n_peri = 5, alpha = (10 * 5) / (10 * 5 + 5) = 50/55 ~ 0.909.
VALUE_RATIO = 10.0

K_GRID = np.arange(3, 13)
N_K = len(K_GRID)

DELTA_GRID_MS = [10, 25, 50, 100, 250, 500, 1000, 3000, 6000, 12000]
DELTA_GRID = np.array([d / 1000.0 for d in DELTA_GRID_MS])
N_DELTA = len(DELTA_GRID)

N_INSTANCES = 3  # random source-layout instances per cell
N_SEEDS_PER_INSTANCE = 3  # random ABR initialisations per instance
N_T = 100
N_T_FINAL = 200
MAX_ROUNDS = 6000
N_HIGH = 5
N_PERI = 5


def alpha_from_value_ratio(ratio, n_high=N_HIGH, n_peri=N_PERI):
    """Convert per-source high/low value ratio into cluster-level alpha."""
    return float((ratio * n_high) / (ratio * n_high + n_peri))


ALPHA = alpha_from_value_ratio(VALUE_RATIO)


HIGH_VALUE_POOL = [
    "us-east1", "us-east4", "us-central1",
    "europe-west1", "europe-west2", "europe-west3", "europe-west4",
    "europe-north1",
    "asia-northeast1", "asia-northeast2",
    "asia-southeast1",
]
PERIPHERAL_POOL = [
    "southamerica-east1", "southamerica-west1",
    "africa-south1",
    "australia-southeast1", "australia-southeast2",
    "asia-south1", "asia-south2",
    "us-west1", "us-west2",
]

OPT_METHOD = "greedy"

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _cache_key():
    """Deterministic hash of all experiment parameters."""
    payload = {
        "MASTER_SEED": MASTER_SEED,
        "TOTAL_VALUE": TOTAL_VALUE,
        "VALUE_RATIO": VALUE_RATIO,
        "K_GRID": K_GRID.tolist(),
        "DELTA_GRID": DELTA_GRID.tolist(),
        "N_INSTANCES": N_INSTANCES,
        "N_SEEDS_PER_INSTANCE": N_SEEDS_PER_INSTANCE,
        "N_T": N_T,
        "N_T_FINAL": N_T_FINAL,
        "MAX_ROUNDS": MAX_ROUNDS,
        "N_HIGH": N_HIGH,
        "N_PERI": N_PERI,
        "HIGH_VALUE_POOL": HIGH_VALUE_POOL,
        "PERIPHERAL_POOL": PERIPHERAL_POOL,
        "REGIONS_EXP": REGIONS_EXP,
        "OPT_METHOD": OPT_METHOD,
    }
    payload_str = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(payload_str.encode()).hexdigest()[:12]


def _cache_path():
    return RESULTS_DIR / f"exp_K_delta_heatmaps_{_cache_key()}.npz"


def sample_source_layout(rng):
    """Return (high_value_regions, peripheral_regions) drawn from the pools."""
    high = list(rng.choice(HIGH_VALUE_POOL, size=N_HIGH, replace=False))
    peri = list(rng.choice(PERIPHERAL_POOL, size=N_PERI, replace=False))
    return high, peri


def _abr_worker(args):
    (k_idx, K, delta_idx, delta, instance_idx, seed_within_instance) = args

    inst_rng = np.random.default_rng(MASTER_SEED + instance_idx)
    high_regions, peri_regions = sample_source_layout(inst_rng)

    regions, prop, region_index_map = load_propagation_model(REGIONS_EXP)
    sources = build_two_cluster_sources(
        ALPHA, TOTAL_VALUE, region_index_map,
        high_value_regions=high_regions,
        distant_regions=peri_regions,
    )
    sliced_prop = make_sliced_prop(sources, prop)
    n_regions = len(regions)

    init_rng = np.random.default_rng(
        MASTER_SEED + 1_000_000 + k_idx * 1_000_000
        + instance_idx * 10_000 + seed_within_instance
    )
    init_regions = [int(init_rng.integers(0, n_regions)) for _ in range(K)]
    abr_seed = (
        MASTER_SEED + 2_000_000
        + k_idx * 1_000_000 + delta_idx * 100_000
        + instance_idx * 10_000 + seed_within_instance
    )

    result = run_abr_full(
        K, sources, sliced_prop, regions, delta, init_regions, abr_seed,
        n_t=N_T, max_rounds=MAX_ROUNDS, n_t_final=N_T_FINAL,
        n_high_sources=N_HIGH,
    )
    result["k_idx"] = k_idx
    result["K"] = int(K)
    result["delta_idx"] = delta_idx
    result["delta"] = float(delta)
    result["instance_idx"] = instance_idx
    result["seed_within_instance"] = seed_within_instance
    return result


def _planner_worker(args):
    (k_idx, K, delta_idx, delta, instance_idx, opt_method) = args

    inst_rng = np.random.default_rng(MASTER_SEED + instance_idx)
    high_regions, peri_regions = sample_source_layout(inst_rng)

    regions, prop, region_index_map = load_propagation_model(REGIONS_EXP)
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
    return {
        "k_idx": k_idx,
        "K": int(K),
        "delta_idx": delta_idx,
        "delta": float(delta),
        "instance_idx": instance_idx,
        "w_opt": w_opt,
        "opt_profile": opt_profile,
    }


def _compute_d_max_by_K():
    """Compute D_max(K) for each K in K_GRID: the maximum mean pairwise
    distance over all K-subsets of REGIONS_EXP. Returns dict {K: D_max_km}.
    """
    region_tuple = tuple(REGIONS_EXP)
    d_max = {}
    for K in K_GRID:
        K_int = int(K)
        n_subsets = comb(len(REGIONS_EXP), K_int)
        print(f"  D_max for K={K_int} ({n_subsets:,} subsets) ...",
              end=" ", flush=True)
        d_max[K_int] = max_mean_pairwise_distance_km(K_int, region_tuple)
        print(f"{d_max[K_int]:.0f} km")
    return d_max


def _build_grids(abr_runs, planner_runs, d_max_by_K):
    """Return dict {metric_name: (N_K, N_DELTA) array of medians}.

    HHI variants:
      - geo_hhi, geo_hhi_norm
      - utility_hhi, utility_hhi_norm

    Pairwise distance:
      - mean_pairwise_km, mean_pairwise_km_norm

    Planner comparison metrics (computed here from cached profiles):
      - placement_emd_km: avg km a builder would need to move to match planner
      - placement_emd_norm: emd_km / D_max(K), in [0, 1]
      - placement_jaccard: multiset Jaccard between ABR and planner placements

    Coverage metrics (kept in dict but not plotted):
      - cov_high, cov_peripheral
    """
    abr_by_cell = {}
    for r in abr_runs:
        key = (r["k_idx"], r["delta_idx"])
        abr_by_cell.setdefault(key, []).append(r)

    planner_by_cell = {}
    for p in planner_runs:
        key = (p["k_idx"], p["delta_idx"])
        planner_by_cell.setdefault(key, []).append(p)

    metrics = ["welfare_ratio",
               "geo_hhi", "geo_hhi_norm",
               "utility_hhi", "utility_hhi_norm",
               "mean_pairwise_km", "mean_pairwise_km_norm",
               "placement_emd_km", "placement_emd_norm",
               "placement_jaccard",
               "cov_high", "cov_peripheral"]
    grids = {m: np.full((N_K, N_DELTA), np.nan) for m in metrics}

    for (ki, di), runs in abr_by_cell.items():
        K = int(K_GRID[ki])
        floor = 1.0 / K
        ne_ceiling = 9.0 / (8.0 * K)
        d_max = d_max_by_K[K]

        planners = planner_by_cell.get((ki, di), [])
        w_opt_by_inst = {p["instance_idx"]: p["w_opt"] for p in planners}
        opt_profile_by_inst = {p["instance_idx"]: p["opt_profile"]
                                for p in planners}

        ratios_list = []
        emd_km_list = []
        jaccard_list = []
        for r in runs:
            inst = r["instance_idx"]
            w_opt = w_opt_by_inst.get(inst)
            if w_opt is not None and w_opt > 1e-12:
                ratios_list.append(r["welfare"] / w_opt)

            opt_profile = opt_profile_by_inst.get(inst)
            if opt_profile is not None and "final_profile" in r:
                emd_km = placement_emd_km(
                    r["final_profile"], opt_profile, REGIONS_EXP,
                )
                jaccard = placement_multiset_jaccard(
                    r["final_profile"], opt_profile,
                )
                emd_km_list.append(emd_km)
                jaccard_list.append(jaccard)

        if ratios_list:
            grids["welfare_ratio"][ki, di] = float(np.median(ratios_list))
        if emd_km_list:
            emd_med = float(np.median(emd_km_list))
            grids["placement_emd_km"][ki, di] = emd_med
            grids["placement_emd_norm"][ki, di] = emd_med / d_max \
                if d_max > 1e-12 else 0.0
        if jaccard_list:
            grids["placement_jaccard"][ki, di] = float(np.median(jaccard_list))

        geo_vals = [r["geo_hhi"] for r in runs if "geo_hhi" in r]
        if geo_vals:
            geo_med = float(np.median(geo_vals))
            grids["geo_hhi"][ki, di] = geo_med
            denom = 1.0 - floor
            grids["geo_hhi_norm"][ki, di] = max(0.0, (geo_med - floor) / denom) \
                if denom > 1e-12 else 0.0

        util_vals = [r["utility_hhi"] for r in runs if "utility_hhi" in r]
        if util_vals:
            util_med = float(np.median(util_vals))
            grids["utility_hhi"][ki, di] = util_med
            denom = ne_ceiling - floor
            grids["utility_hhi_norm"][ki, di] = (util_med - floor) / denom \
                if denom > 1e-12 else 0.0

        mpd_vals = [r["mean_pairwise_km"] for r in runs if "mean_pairwise_km" in r]
        if mpd_vals:
            mpd_med = float(np.median(mpd_vals))
            grids["mean_pairwise_km"][ki, di] = mpd_med
            grids["mean_pairwise_km_norm"][ki, di] = mpd_med / d_max \
                if d_max > 1e-12 else 0.0

        for key in ("cov_high", "cov_peripheral"):
            vals = [r[key] for r in runs if key in r]
            if vals:
                grids[key][ki, di] = float(np.median(vals))

    return grids


def _plot_heatmap(ax, grid_color, grid_labels, title, cbar_label,
                  vmin=None, vmax=None, cmap="viridis", label_fmt="{:.2f}"):
    """Plot one (N_K, N_DELTA) heatmap."""
    im = ax.imshow(
        grid_color,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        interpolation="nearest",
    )

    n_k, n_delta = grid_color.shape
    ax.set_yticks(np.arange(n_k))
    ax.set_yticklabels([str(k) for k in K_GRID], fontsize=7)
    ax.set_xticks(np.arange(n_delta))
    ax.set_xticklabels([f"{int(d*1000)}" for d in DELTA_GRID],
                       fontsize=7, rotation=45)

    delta_50_idx = int(np.argmin(np.abs(DELTA_GRID - 0.050)))
    if abs(DELTA_GRID[delta_50_idx] - 0.050) / 0.050 < 0.30:
        ax.axvline(delta_50_idx, color="white", lw=1.0, ls="--", alpha=0.7)

    ax.set_xlabel(r"Slot duration $\Delta$ (ms)")
    ax.set_ylabel("Number of builders $K$")
    ax.set_title(title, fontsize=10)

    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)

    if grid_labels is not None and n_k * n_delta <= 200:
        gmin = vmin if vmin is not None else np.nanmin(grid_color)
        gmax = vmax if vmax is not None else np.nanmax(grid_color)
        rng = max(1e-9, gmax - gmin)
        for ki in range(n_k):
            for di in range(n_delta):
                v_color = grid_color[ki, di]
                v_label = grid_labels[ki, di]
                if np.isnan(v_color) or np.isnan(v_label):
                    continue
                norm_v = (v_color - gmin) / rng
                color = "white" if norm_v < 0.5 else "black"
                ax.text(di, ki, label_fmt.format(v_label),
                        ha="center", va="center", fontsize=6, color=color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(cbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)


def plot_heatmaps(grids):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.subplots_adjust(hspace=0.45, wspace=0.45)

    _plot_heatmap(
        axes[0, 0],
        grids["welfare_ratio"], grids["welfare_ratio"],
        "(A) Welfare ratio $W_{\\rm ABR}/W^*$",
        "ratio", vmin=0.5, vmax=1.0, cmap="viridis",
    )

    _plot_heatmap(
        axes[0, 1],
        grids["geo_hhi_norm"], grids["geo_hhi_norm"],
        "(B) Geographic HHI (normalised)",
        "$(\\mathrm{HHI} - 1/K) / (1 - 1/K)$",
        vmin=0.0, vmax=1.0, cmap="magma",
        label_fmt="{:.2f}",
    )

    _plot_heatmap(
        axes[0, 2],
        grids["utility_hhi_norm"], grids["utility_hhi_norm"],
        "(C) Utility HHI (normalised)",
        "$(\\mathrm{HHI} - 1/K) / (9/(8K) - 1/K)$",
        vmin=0.0, vmax=1.0, cmap="magma",
        label_fmt="{:.2f}",
    )

    _plot_heatmap(
        axes[1, 0],
        grids["mean_pairwise_km_norm"], grids["mean_pairwise_km_norm"],
        "(D) Mean pairwise distance (normalised by $D_{\\max}(K)$)",
        "$D_{\\rm ABR}\\,/\\,D_{\\max}(K)$",
        vmin=0.0, vmax=1.0, cmap="cividis",
        label_fmt="{:.2f}",
    )

    _plot_heatmap(
        axes[1, 1],
        grids["placement_emd_norm"], grids["placement_emd_norm"],
        "(E) Placement EMD vs planner (normalised by $D_{\\max}(K)$)",
        "$\\mathrm{EMD}\\,/\\,D_{\\max}(K)$",
        vmin=0.0, vmax=1.0, cmap="inferno",
        label_fmt="{:.2f}",
    )

    _plot_heatmap(
        axes[1, 2],
        grids["placement_jaccard"], grids["placement_jaccard"],
        "(F) Placement Jaccard (multiset) vs planner",
        "Jaccard similarity",
        vmin=0.0, vmax=1.0, cmap="viridis",
        label_fmt="{:.2f}",
    )

    fig.suptitle(
        rf"Builder count vs slot duration heatmaps "
        rf"(value ratio = {VALUE_RATIO:g}x, $\alpha \approx {ALPHA:.3f}$, "
        rf"{N_INSTANCES} source instances $\times$ "
        rf"{N_SEEDS_PER_INSTANCE} inits per cell; dashed line: $\Delta=50$ ms; "
        rf"opt method: {OPT_METHOD})",
        fontsize=12, y=0.995,
    )
    out = FIGURES_DIR / "exp_K_delta_heatmaps.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=180, bbox_inches="tight")
    print(f"Saved {out}")


def _save_cache(abr_runs, planner_runs, path):
    np.savez_compressed(
        path,
        abr_runs=np.array(abr_runs, dtype=object),
        planner_runs=np.array(planner_runs, dtype=object),
        k_grid=K_GRID,
        delta_grid=DELTA_GRID,
        cache_key=_cache_key(),
    )
    print(f"Cached results to {path}")


def _load_cache(path):
    npz = np.load(path, allow_pickle=True)
    abr_runs = list(npz["abr_runs"])
    planner_runs = list(npz["planner_runs"])
    print(f"Loaded {len(abr_runs)} ABR runs and {len(planner_runs)} planner "
          f"runs from {path}")
    return abr_runs, planner_runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true",
                        help="Load cached results and replot only.")
    parser.add_argument("--rerun", action="store_true",
                        help="Force recomputation even if cache exists.")
    args = parser.parse_args()

    cache_path = _cache_path()
    cache_exists = cache_path.exists()

    pool_regions = set(HIGH_VALUE_POOL) | set(PERIPHERAL_POOL)
    missing_regions = [r for r in pool_regions if r not in REGIONS_EXP]
    missing_coords = [r for r in pool_regions if r not in GCP_REGION_COORDS]
    if missing_regions or missing_coords:
        raise ValueError(
            f"Pool regions misconfigured: "
            f"missing from REGIONS_EXP={missing_regions}, "
            f"missing from GCP_REGION_COORDS={missing_coords}"
        )

    n_cells = N_K * N_DELTA
    n_runs_per_cell = N_INSTANCES * N_SEEDS_PER_INSTANCE
    print(f"K vs Delta heatmaps: ({N_K} K values) x ({N_DELTA} deltas) = {n_cells} cells")
    print(f"K grid: {list(K_GRID)}")
    print(f"Delta grid: {DELTA_GRID_MS} ms")
    print(f"Value ratio: {VALUE_RATIO:g}x (alpha = {ALPHA:.4f})")
    print(f"{N_INSTANCES} source instances x {N_SEEDS_PER_INSTANCE} "
          f"random inits = {n_runs_per_cell} ABR runs per cell")
    print(f"Total ABR jobs: {n_cells * n_runs_per_cell}")
    print(f"Total planner jobs: {n_cells * N_INSTANCES} "
          f"(opt method: {OPT_METHOD})")
    print(f"Cache: {cache_path} (exists: {cache_exists})")
    print()

    print("Computing D_max(K) for each K:")
    d_max_by_K = _compute_d_max_by_K()
    print()

    if args.load:
        if not cache_exists:
            raise FileNotFoundError(
                f"No cache found at {cache_path}; cannot use --load."
            )
        abr_runs, planner_runs = _load_cache(cache_path)
    elif cache_exists and not args.rerun:
        print("Cache hit; loading. Use --rerun to force recomputation.")
        abr_runs, planner_runs = _load_cache(cache_path)
    else:
        abr_runs, planner_runs = _compute(args)
        _save_cache(abr_runs, planner_runs, cache_path)

    grids = _build_grids(abr_runs, planner_runs, d_max_by_K)
    plot_heatmaps(grids)


def _compute(args):
    n_workers = max(1, cpu_count() - 1)

    planner_tasks = [
        (ki, int(K), di, delta, inst, OPT_METHOD)
        for ki, K in enumerate(K_GRID)
        for di, delta in enumerate(DELTA_GRID)
        for inst in range(N_INSTANCES)
    ]
    print(f"Computing {len(planner_tasks)} planner benchmarks "
          f"({n_workers} workers) ...")
    planner_runs = []
    with Pool(n_workers) as pool:
        for i, p in enumerate(pool.imap_unordered(_planner_worker, planner_tasks)):
            planner_runs.append(p)
            if (i + 1) % 20 == 0 or (i + 1) == len(planner_tasks):
                print(f"  planner [{i+1}/{len(planner_tasks)}] "
                      f"K={p['K']} "
                      f"delta={int(p['delta']*1000)}ms "
                      f"W*={p['w_opt']:.4f}")

    abr_tasks = [
        (ki, int(K), di, delta, inst, seed)
        for ki, K in enumerate(K_GRID)
        for di, delta in enumerate(DELTA_GRID)
        for inst in range(N_INSTANCES)
        for seed in range(N_SEEDS_PER_INSTANCE)
    ]
    print(f"\nRunning {len(abr_tasks)} ABR jobs ({n_workers} workers) ...")
    abr_runs = []
    with Pool(n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_abr_worker, abr_tasks)):
            abr_runs.append(r)
            if (i + 1) % 50 == 0 or (i + 1) == len(abr_tasks):
                print(f"  abr [{i+1}/{len(abr_tasks)}] "
                      f"K={r['K']} "
                      f"delta={int(r['delta']*1000)}ms "
                      f"inst={r['instance_idx']} "
                      f"seed={r['seed_within_instance']} "
                      f"welfare={r['welfare']:.4f}")

    n_truncated = sum(
        1 for r in abr_runs
        if (r.get("converged") is False
            or (r.get("rounds_used", 0) >= MAX_ROUNDS))
    )
    if n_truncated > 0:
        warnings.warn(
            f"{n_truncated}/{len(abr_runs)} ABR runs hit MAX_ROUNDS={MAX_ROUNDS} "
            f"without converging."
        )

    return abr_runs, planner_runs


if __name__ == "__main__":
    main()
