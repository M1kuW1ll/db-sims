"""
Experiment 3: Heatmaps over (value ratio, delta)

Sweeps the value concentration ratio (high cluster / peripheral cluster) on the
y-axis and slot duration delta on the x-axis. Internally each ratio is
converted to alpha = (ratio * n_high) / (ratio * n_high + n_peri) for source
construction.
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
from matplotlib import colors
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Line3DCollection

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
from sim.metrics import hhi as _hhi
from sim.simulator import compute_all_builder_utilities

REGIONS_EXP3 = list(REGIONS_DEFAULT) + ["europe-west2", "asia-northeast2", "asia-south2", "us-west2"]

# Experiment parameters
MASTER_SEED = 1234
K = 5
TOTAL_VALUE = 10.0
VALUE_RATIO_GRID = np.array([1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 5.0, 10.0, 20.0])
N_RATIO = len(VALUE_RATIO_GRID)

DELTA_GRID_MS = [10, 25, 50, 100, 250, 500, 1000, 3000, 6000, 12000]
DELTA_GRID = np.array([d / 1000.0 for d in DELTA_GRID_MS])
N_DELTA = len(DELTA_GRID)

N_INSTANCES = 3  # random source-layout instances per cell
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

# Cache directory
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
        "K": K,
        "TOTAL_VALUE": TOTAL_VALUE,
        "VALUE_RATIO_GRID": VALUE_RATIO_GRID.tolist(),
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
        "REGIONS_EXP3": REGIONS_EXP3,
        "OPT_METHOD": OPT_METHOD,
    }
    payload_str = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(payload_str.encode()).hexdigest()[:12]


def _cache_path():
    return RESULTS_DIR / f"exp3_k{K}_{_cache_key()}.npz"

def sample_source_layout(rng):
    """Return (high_value_regions, peripheral_regions) drawn from the pools."""
    high = list(rng.choice(HIGH_VALUE_POOL, size=N_HIGH, replace=False))
    peri = list(rng.choice(PERIPHERAL_POOL, size=N_PERI, replace=False))
    return high, peri


def _source_model(alpha, instance_idx):
    inst_rng = np.random.default_rng(MASTER_SEED + instance_idx)
    high_regions, peri_regions = sample_source_layout(inst_rng)

    regions, prop, region_index_map = load_propagation_model(REGIONS_EXP3)
    sources = build_two_cluster_sources(
        alpha, TOTAL_VALUE, region_index_map,
        high_value_regions=high_regions,
        distant_regions=peri_regions,
    )
    sliced_prop = make_sliced_prop(sources, prop)
    return regions, sources, sliced_prop


def _abr_worker(args):
    (ratio_idx, value_ratio, alpha,
     delta_idx, delta, instance_idx, seed_within_instance) = args

    regions, sources, sliced_prop = _source_model(alpha, instance_idx)
    n_regions = len(regions)

    # ABR initialisation: deterministic per (instance, seed), not ratio/delta
    # dependent so the same initial placements are used across the sweep.
    init_rng = np.random.default_rng(
        MASTER_SEED + 1_000_000 + instance_idx * 10_000 + seed_within_instance
    )
    init_regions = [int(init_rng.integers(0, n_regions)) for _ in range(K)]
    abr_seed = (
        MASTER_SEED + 2_000_000
        + ratio_idx * 1_000_000 + delta_idx * 100_000
        + instance_idx * 10_000 + seed_within_instance
    )

    result = run_abr_full(
        K, sources, sliced_prop, regions, delta, init_regions, abr_seed,
        n_t=N_T, max_rounds=MAX_ROUNDS, n_t_final=N_T_FINAL,
        n_high_sources=N_HIGH,
    )
    result["ratio_idx"] = ratio_idx
    result["value_ratio"] = float(value_ratio)
    result["alpha"] = float(alpha)
    result["delta_idx"] = delta_idx
    result["delta"] = float(delta)
    result["instance_idx"] = instance_idx
    result["seed_within_instance"] = seed_within_instance
    return result


def _planner_metrics(profile, sources, sliced_prop, regions, delta):
    profile = [int(r) for r in profile]
    high_ids = list(range(N_HIGH))
    peri_ids = list(range(N_HIGH, N_HIGH + N_PERI))
    utilities = compute_all_builder_utilities(
        profile, sources, sliced_prop, delta, N_T_FINAL,
    )
    return {
        "geo_hhi_opt": geo_hhi(profile, len(regions)),
        "mean_pairwise_km_opt": mean_pairwise_distance_km(profile, list(regions)),
        "utility_hhi_opt": float(_hhi(utilities)),
        "cov_high_opt": cluster_coverage_fraction(
            profile, sources, sliced_prop, delta, high_ids, n_t=N_T_FINAL),
        "cov_peripheral_opt": cluster_coverage_fraction(
            profile, sources, sliced_prop, delta, peri_ids, n_t=N_T_FINAL),
    }


def _planner_worker(args):
    (ratio_idx, value_ratio, alpha,
     delta_idx, delta, instance_idx, opt_method) = args

    regions, sources, sliced_prop = _source_model(alpha, instance_idx)
    n_regions = len(regions)
    w_opt, opt_profile = compute_opt_sliced(
        K, sources, sliced_prop, n_regions, delta,
        n_t=N_T_FINAL, method=opt_method,
    )
    metrics = _planner_metrics(opt_profile, sources, sliced_prop, regions, delta)
    return {
        "ratio_idx": ratio_idx,
        "value_ratio": float(value_ratio),
        "alpha": float(alpha),
        "delta_idx": delta_idx,
        "delta": float(delta),
        "instance_idx": instance_idx,
        "w_opt": w_opt,
        "opt_profile": opt_profile,
        **metrics,
    }


def _build_grids(abr_runs, planner_runs):
    """Return dict {metric_name: (N_RATIO, N_DELTA) array of medians}."""
    abr_by_cell = {}
    for r in abr_runs:
        key = (r["ratio_idx"], r["delta_idx"])
        abr_by_cell.setdefault(key, []).append(r)

    planner_by_cell = {}
    for p in planner_runs:
        key = (p["ratio_idx"], p["delta_idx"])
        planner_by_cell.setdefault(key, []).append(p)

    metrics = ["welfare_ratio", "geo_hhi", "utility_hhi",
               "mean_pairwise_km", "cov_high", "cov_peripheral"]
    grids = {m: np.full((N_RATIO, N_DELTA), np.nan) for m in metrics}

    for (ri, di), runs in abr_by_cell.items():
        planners = planner_by_cell.get((ri, di), [])
        w_opt_by_inst = {p["instance_idx"]: p["w_opt"] for p in planners}

        ratios_list = []
        for r in runs:
            w_opt = w_opt_by_inst.get(r["instance_idx"])
            if w_opt is not None and w_opt > 1e-12:
                ratios_list.append(r["welfare"] / w_opt)
        if ratios_list:
            grids["welfare_ratio"][ri, di] = float(np.median(ratios_list))

        for key in ("geo_hhi", "utility_hhi", "mean_pairwise_km",
                    "cov_high", "cov_peripheral"):
            vals = [r[key] for r in runs]
            if vals:
                grids[key][ri, di] = float(np.median(vals))

    return grids


_PLANNER_METRIC_KEYS = (
    "geo_hhi_opt",
    "mean_pairwise_km_opt",
    "utility_hhi_opt",
    "cov_high_opt",
    "cov_peripheral_opt",
)


def _ensure_planner_metrics(planner_runs):
    """Add planner concentration/coverage metrics to old cached planner runs."""
    out = []
    missing = 0
    for run in planner_runs:
        if all(key in run for key in _PLANNER_METRIC_KEYS):
            out.append(run)
            continue

        missing += 1
        enriched = dict(run)
        alpha = float(enriched.get(
            "alpha", ALPHA_GRID[int(enriched["ratio_idx"])]
        ))
        regions, sources, sliced_prop = _source_model(
            alpha, int(enriched["instance_idx"])
        )
        enriched.update(_planner_metrics(
            enriched["opt_profile"], sources, sliced_prop,
            regions, float(enriched["delta"]),
        ))
        out.append(enriched)

    if missing:
        print(f"Computed planner metrics for {missing} cached planner runs.")
    return out


def _build_planner_grids(planner_runs):
    """Return planner-optimal grids in the same shape as the PNE grids."""
    planner_runs = _ensure_planner_metrics(planner_runs)
    planner_by_cell = {}
    for p in planner_runs:
        key = (p["ratio_idx"], p["delta_idx"])
        planner_by_cell.setdefault(key, []).append(p)

    metrics = ["welfare_ratio", "geo_hhi", "utility_hhi",
               "mean_pairwise_km", "cov_high", "cov_peripheral"]
    grids = {m: np.full((N_RATIO, N_DELTA), np.nan) for m in metrics}

    key_map = {
        "geo_hhi": "geo_hhi_opt",
        "utility_hhi": "utility_hhi_opt",
        "mean_pairwise_km": "mean_pairwise_km_opt",
        "cov_high": "cov_high_opt",
        "cov_peripheral": "cov_peripheral_opt",
    }
    for (ri, di), runs in planner_by_cell.items():
        if any(p.get("w_opt", 0.0) > 1e-12 for p in runs):
            grids["welfare_ratio"][ri, di] = 1.0
        for out_key, planner_key in key_map.items():
            vals = [p[planner_key] for p in runs if planner_key in p]
            if vals:
                grids[out_key][ri, di] = float(np.median(vals))

    return grids


def _plot_heatmap(ax, grid, title, cbar_label, vmin=None, vmax=None,
                  cmap="viridis", compact_x=False, show_ylabel=True,
                  show_xlabel=True):
    """Plot one (N_RATIO, N_DELTA) heatmap. ratio on y, delta on x."""
    im = ax.imshow(
        grid,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        interpolation="nearest",
    )

    n_ratio, n_delta = grid.shape
    ax.set_yticks(np.arange(n_ratio))
    ax.set_yticklabels(VALUE_RATIO_LABELS)
    ax.tick_params(axis="both", labelsize=10)

    if compact_x:
        tick_idx = [0, 2, 4, 6, n_delta - 1]
    else:
        tick_idx = list(range(n_delta))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{int(DELTA_GRID[i] * 1000)}" for i in tick_idx],
                       rotation=45, ha="right")

    delta_50_idx = int(np.argmin(np.abs(DELTA_GRID - 0.050)))
    if abs(DELTA_GRID[delta_50_idx] - 0.050) / 0.050 < 0.30:
        ax.axvline(delta_50_idx, color="white", lw=1.0, ls="--", alpha=0.7)

    if show_xlabel:
        ax.set_xlabel(r"Slot duration $\Delta$ (ms)", fontsize=11)
    if show_ylabel:
        ax.set_ylabel(r"High-to-low value ratio $v$", fontsize=11)
    ax.set_title(title, fontsize=12)

    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)

    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label(cbar_label, fontsize=10)
    cbar.ax.tick_params(labelsize=10)


def plot_heatmaps(grids, output_stem=None, paper_stem=None, title_prefix=""):
    egal_floor = 1 / K
    ne_ceiling = 9 / (8 * K)
    utility_vmax = max(ne_ceiling, float(np.nanmax(grids["utility_hhi"])))
    prefix = f"{title_prefix} " if title_prefix else ""

    fig = plt.figure(figsize=(12, 6.1))
    gs = fig.add_gridspec(
        nrows=2, ncols=18,
        hspace=0.55, wspace=0.70,
    )

    ax_welfare = fig.add_subplot(gs[0, 0:8])
    ax_high = fig.add_subplot(gs[0, 10:13])
    ax_peri = fig.add_subplot(gs[0, 15:18])
    ax_geo = fig.add_subplot(gs[1, 0:8])
    ax_util = fig.add_subplot(gs[1, 10:18])

    _plot_heatmap(
        ax_welfare, grids["welfare_ratio"], f"{prefix}Welfare Ratio", "ratio",
        vmin=0.5, vmax=1.0, cmap="viridis",
    )
    _plot_heatmap(
        ax_high, grids["cov_high"], f"{prefix}High-value Coverage", "fraction",
        vmin=0.0, vmax=1.0, cmap="viridis", compact_x=True,
        show_ylabel=False,
    )
    _plot_heatmap(
        ax_peri, grids["cov_peripheral"], f"{prefix}Peripheral Coverage", "fraction",
        vmin=0.0, vmax=1.0, cmap="viridis", compact_x=True,
        show_ylabel=False,
    )
    _plot_heatmap(
        ax_geo, grids["geo_hhi"], f"{prefix}Geographic HHI", "HHI",
        vmin=egal_floor, vmax=1.0, cmap="magma",
    )
    _plot_heatmap(
        ax_util, grids["utility_hhi"], f"{prefix}Utility HHI", "HHI",
        vmin=egal_floor, vmax=utility_vmax, cmap="magma",
    )

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.13, top=0.94)
    if output_stem is None:
        output_stem = f"exp3_k{K}_ratio_delta_heatmaps"
    if paper_stem is None:
        paper_stem = "paper_exp3_2x2"
    out = FIGURES_DIR / f"{output_stem}.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=180, bbox_inches="tight")
    paper_out = FIGURES_DIR / f"{paper_stem}.pdf"
    fig.savefig(paper_out, bbox_inches="tight")
    fig.savefig(str(paper_out).replace(".pdf", ".png"), dpi=180, bbox_inches="tight")
    print(f"Saved {out}")
    print(f"Saved {paper_out}")


def _surface_mesh():
    x = np.log10(np.asarray(DELTA_GRID_MS, dtype=float))
    y = np.log10(np.asarray(VALUE_RATIO_GRID, dtype=float))
    return np.meshgrid(x, y)


def _format_surface_axis(ax, title, _zlabel, zlim, elev=24, azim=-58,
                         invert_y_axis=False):
    delta_ticks = [10, 50, 250, 1000, 6000, 12000]
    ratio_ticks = [1, 2, 5, 10, 20]

    ax.set_title(title, fontsize=14, pad=5)
    ax.set_xlabel("Slot duration (ms)", fontsize=11, labelpad=8)
    ax.set_ylabel("Value ratio", fontsize=11, labelpad=8)
    ax.set_xticks(np.log10(delta_ticks))
    ax.set_xticklabels([str(t) for t in delta_ticks], rotation=45, ha="right")
    ax.set_yticks(np.log10(ratio_ticks))
    ax.set_yticklabels([_format_ratio_label(t) for t in ratio_ticks])
    if invert_y_axis:
        ax.set_ylim(np.log10(max(ratio_ticks)), np.log10(min(ratio_ticks)))
    ax.set_zlim(*zlim)
    ax.tick_params(axis="x", labelsize=10, pad=-7)
    ax.tick_params(axis="y", labelsize=10, pad=2)
    ax.tick_params(axis="z", labelsize=10, pad=2)
    ax.view_init(elev=elev, azim=azim)
    ax.xaxis.pane.set_facecolor((1, 1, 1, 0.0))
    ax.yaxis.pane.set_facecolor((1, 1, 1, 0.0))
    ax.zaxis.pane.set_facecolor((1, 1, 1, 0.0))


def _surface_facecolors(z, zlim, cmap_name, alpha, cmap_floor=0.0):
    norm = colors.Normalize(vmin=zlim[0], vmax=zlim[1])
    scaled = cmap_floor + (1.0 - cmap_floor) * norm(z)
    facecolors = plt.get_cmap(cmap_name)(scaled)
    facecolors[..., -1] = alpha
    return facecolors


def _pne_blue_cmap():
    return colors.LinearSegmentedColormap.from_list(
        "pne_overlay_blues", ["#7fb8e8", "#2f7ed8", "#004caa", "#00164d"]
    )


def _finite_zlim(z, fallback_zlim):
    finite = z[np.isfinite(z)]
    if finite.size == 0:
        return fallback_zlim
    zmin = float(np.min(finite))
    zmax = float(np.max(finite))
    if abs(zmax - zmin) < 1e-12:
        pad = max(0.5, 0.01 * abs(zmin))
        return zmin - pad, zmax + pad
    return zmin, zmax


def _pne_overlay_facecolors(z, alpha, norm_zlim):
    norm = colors.Normalize(vmin=norm_zlim[0], vmax=norm_zlim[1])
    facecolors = _pne_blue_cmap()(norm(z))
    facecolors[..., -1] = alpha
    return facecolors


def _plot_value_wireframe(ax, x, y, z, zlim, cmap_name, offset=0.0,
                          linewidth=1.75, alpha=1.0, norm_zlim=None):
    if norm_zlim is None:
        norm_zlim = zlim
    norm_zlim = _finite_zlim(z, norm_zlim)
    norm = colors.Normalize(vmin=norm_zlim[0], vmax=norm_zlim[1])
    if cmap_name == "PNEBlues":
        cmap = _pne_blue_cmap()
    else:
        cmap = plt.get_cmap(cmap_name)
    segments = []
    segment_colors = []

    def add_segment(p0, p1):
        avg_z = 0.5 * (p0[2] + p1[2]) - offset
        rgba = list(cmap(norm(avg_z)))
        rgba[-1] = alpha
        segments.append([p0, p1])
        segment_colors.append(rgba)

    z_lifted = z + offset
    for row in range(x.shape[0]):
        for col in range(x.shape[1] - 1):
            add_segment(
                (x[row, col], y[row, col], z_lifted[row, col]),
                (x[row, col + 1], y[row, col + 1], z_lifted[row, col + 1]),
            )
    for row in range(x.shape[0] - 1):
        for col in range(x.shape[1]):
            add_segment(
                (x[row, col], y[row, col], z_lifted[row, col]),
                (x[row + 1, col], y[row + 1, col], z_lifted[row + 1, col]),
            )

    line_collection = Line3DCollection(
        segments,
        colors=segment_colors,
        linewidths=linewidth,
    )
    ax.add_collection3d(line_collection)


def _gap_facecolors(z, zlim, alpha):
    norm = colors.TwoSlopeNorm(vmin=zlim[0], vcenter=0.0, vmax=zlim[1])
    facecolors = plt.get_cmap("RdBu_r")(norm(z))
    facecolors[..., -1] = alpha
    return facecolors


def _plot_single_surface(ax, x, y, z, cmap_name, title, zlabel, zlim,
                         elev=24, azim=-58, invert_y_axis=False):
    ax.plot_surface(
        x, y, z,
        facecolors=_surface_facecolors(z, zlim, cmap_name, 0.86),
        shade=False,
        edgecolor=(0, 0, 0, 0.18),
        linewidth=0.25,
        antialiased=True,
    )
    _format_surface_axis(
        ax, title, zlabel, zlim, elev=elev, azim=azim,
        invert_y_axis=invert_y_axis,
    )


def _plot_comparison_surface(ax, x, y, pne_z, planner_z, title, zlabel, zlim,
                             elev=24, azim=-58, invert_y_axis=False,
                             mesh_overlay=False, filled_overlay=False,
                             filled_overlay_alpha=0.52):
    wire_offset = 0.0
    planner_alpha = 0.56 if filled_overlay else 0.78
    ax.plot_surface(
        x, y, planner_z,
        facecolors=_surface_facecolors(
            planner_z, zlim, "Reds", planner_alpha, cmap_floor=0.30
        ),
        shade=False,
        edgecolor=(0.55, 0.05, 0.05, 0.28),
        linewidth=0.25,
        antialiased=True,
    )
    fill_offset = 0.006 * (zlim[1] - zlim[0]) if filled_overlay else 0.0
    pne_norm_zlim = _finite_zlim(pne_z, zlim)
    pne_alpha = (
        filled_overlay_alpha if filled_overlay
        else (0.12 if mesh_overlay else 0.88)
    )
    if filled_overlay:
        pne_facecolors = _pne_overlay_facecolors(
            pne_z, alpha=pne_alpha, norm_zlim=pne_norm_zlim
        )
    else:
        pne_facecolors = _surface_facecolors(
            pne_z, zlim, "Blues", pne_alpha, cmap_floor=0.38
        )
    ax.plot_surface(
        x, y, pne_z + fill_offset,
        facecolors=pne_facecolors,
        shade=False,
        edgecolor=(
            (0.02, 0.11, 0.28, 0.18)
            if filled_overlay else
            (0.05, 0.18, 0.35, 0.0 if mesh_overlay else 0.28)
        ),
        linewidth=0.25,
        antialiased=True,
    )
    if mesh_overlay:
        wire_offset = 0.012 * (zlim[1] - zlim[0])
        _plot_value_wireframe(
            ax, x, y, pne_z, zlim, "PNEBlues",
            offset=wire_offset, norm_zlim=pne_norm_zlim,
        )
    _format_surface_axis(
        ax, title, zlabel, (zlim[0], zlim[1] + max(wire_offset, fill_offset)),
        elev=elev, azim=azim,
        invert_y_axis=invert_y_axis,
    )


def _plot_gap_surface(ax, x, y, gap_z, title, zlim,
                      elev=24, azim=-58, invert_y_axis=False):
    ax.plot_surface(
        x, y, np.zeros_like(gap_z),
        color=(0.5, 0.5, 0.5, 0.16),
        edgecolor=(0.35, 0.35, 0.35, 0.18),
        linewidth=0.2,
        antialiased=True,
    )
    ax.plot_surface(
        x, y, gap_z,
        facecolors=_gap_facecolors(gap_z, zlim, 0.88),
        shade=False,
        edgecolor=(0, 0, 0, 0.18),
        linewidth=0.25,
        antialiased=True,
    )
    _format_surface_axis(
        ax, title, "gap", zlim,
        elev=elev, azim=azim,
        invert_y_axis=invert_y_axis,
    )


def plot_3d_surfaces(pne_grids, planner_grids, output_stem=None,
                     paper_stem=None, elev=24, azim=-58,
                     invert_y_axis=False, coverage_mesh_overlay=False,
                     hhi_mesh_overlay=False, coverage_filled_overlay=False,
                     hhi_filled_overlay=False, coverage_filled_alpha=0.52,
                     hhi_filled_alpha=0.52):
    x, y = _surface_mesh()
    fig = plt.figure(figsize=(15, 8.6))
    gs = fig.add_gridspec(2, 6, hspace=0.22, wspace=0.02)

    axes = [
        fig.add_subplot(gs[0, 0:2], projection="3d"),
        fig.add_subplot(gs[0, 2:4], projection="3d"),
        fig.add_subplot(gs[0, 4:6], projection="3d"),
        fig.add_subplot(gs[1, 1:3], projection="3d"),
        fig.add_subplot(gs[1, 3:5], projection="3d"),
    ]

    _plot_single_surface(
        axes[0], x, y, pne_grids["welfare_ratio"], "Greens",
        "PNE Welfare Ratio", "ratio", (0.5, 1.0), elev=elev, azim=azim,
        invert_y_axis=invert_y_axis,
    )
    _plot_comparison_surface(
        axes[1], x, y, pne_grids["cov_high"], planner_grids["cov_high"],
        "High-value Coverage", "fraction", (0.0, 1.0),
        elev=elev, azim=azim, invert_y_axis=invert_y_axis,
        mesh_overlay=coverage_mesh_overlay,
        filled_overlay=coverage_filled_overlay,
        filled_overlay_alpha=coverage_filled_alpha,
    )
    _plot_comparison_surface(
        axes[2], x, y, pne_grids["cov_peripheral"], planner_grids["cov_peripheral"],
        "Peripheral Coverage", "fraction", (0.0, 1.0),
        elev=elev, azim=azim, invert_y_axis=invert_y_axis,
        mesh_overlay=coverage_mesh_overlay,
        filled_overlay=coverage_filled_overlay,
        filled_overlay_alpha=coverage_filled_alpha,
    )

    geo_max = max(
        float(np.nanmax(pne_grids["geo_hhi"])),
        float(np.nanmax(planner_grids["geo_hhi"])),
        1 / K,
    )
    _plot_comparison_surface(
        axes[3], x, y, pne_grids["geo_hhi"], planner_grids["geo_hhi"],
        "Geographic HHI", "HHI", (1 / K, min(1.0, geo_max + 0.05)),
        elev=elev, azim=azim, invert_y_axis=invert_y_axis,
        mesh_overlay=hhi_mesh_overlay,
        filled_overlay=hhi_filled_overlay,
        filled_overlay_alpha=hhi_filled_alpha,
    )

    util_max = max(
        float(np.nanmax(pne_grids["utility_hhi"])),
        float(np.nanmax(planner_grids["utility_hhi"])),
        9 / (8 * K),
    )
    _plot_comparison_surface(
        axes[4], x, y, pne_grids["utility_hhi"], planner_grids["utility_hhi"],
        "Utility HHI", "HHI", (1 / K, util_max + 0.01),
        elev=elev, azim=azim, invert_y_axis=invert_y_axis,
        mesh_overlay=hhi_mesh_overlay,
        filled_overlay=hhi_filled_overlay,
        filled_overlay_alpha=hhi_filled_alpha,
    )

    legend_handles = [
        Patch(facecolor="#2ca02c", edgecolor="none", alpha=0.86,
              label="PNE welfare ratio (darker = higher)"),
        Patch(facecolor="#1f77b4", edgecolor="none", alpha=0.68,
              label="PNE (darker = higher)"),
        Patch(facecolor="#d62728", edgecolor="none", alpha=0.58,
              label="Planner (darker = higher)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.10, top=0.94)

    if output_stem is None:
        output_stem = f"exp3_k{K}_3d_surfaces"
    if paper_stem is None:
        paper_stem = "paper_exp3_3d_surfaces"
    out = FIGURES_DIR / f"{output_stem}.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=180, bbox_inches="tight")
    paper_out = FIGURES_DIR / f"{paper_stem}.pdf"
    fig.savefig(paper_out, bbox_inches="tight")
    fig.savefig(str(paper_out).replace(".pdf", ".png"), dpi=180, bbox_inches="tight")
    print(f"Saved {out}")
    print(f"Saved {paper_out}")


def plot_3d_surfaces_coverage_gap(pne_grids, planner_grids, output_stem=None,
                                  paper_stem=None, elev=24, azim=-58,
                                  invert_y_axis=False):
    x, y = _surface_mesh()
    fig = plt.figure(figsize=(15, 8.6))
    gs = fig.add_gridspec(2, 6, hspace=0.22, wspace=0.02)

    axes = [
        fig.add_subplot(gs[0, 0:2], projection="3d"),
        fig.add_subplot(gs[0, 2:4], projection="3d"),
        fig.add_subplot(gs[0, 4:6], projection="3d"),
        fig.add_subplot(gs[1, 1:3], projection="3d"),
        fig.add_subplot(gs[1, 3:5], projection="3d"),
    ]

    _plot_single_surface(
        axes[0], x, y, pne_grids["welfare_ratio"], "Greens",
        "PNE Welfare Ratio", "ratio", (0.5, 1.0), elev=elev, azim=azim,
        invert_y_axis=invert_y_axis,
    )

    high_gap = planner_grids["cov_high"] - pne_grids["cov_high"]
    peri_gap = planner_grids["cov_peripheral"] - pne_grids["cov_peripheral"]
    gap_abs = max(
        0.05,
        float(np.nanmax(np.abs(high_gap))),
        float(np.nanmax(np.abs(peri_gap))),
    )
    gap_zlim = (-gap_abs, gap_abs)
    _plot_gap_surface(
        axes[1], x, y, high_gap,
        "High-value Coverage Gap", gap_zlim,
        elev=elev, azim=azim, invert_y_axis=invert_y_axis,
    )
    _plot_gap_surface(
        axes[2], x, y, peri_gap,
        "Peripheral Coverage Gap", gap_zlim,
        elev=elev, azim=azim, invert_y_axis=invert_y_axis,
    )

    geo_max = max(
        float(np.nanmax(pne_grids["geo_hhi"])),
        float(np.nanmax(planner_grids["geo_hhi"])),
        1 / K,
    )
    _plot_comparison_surface(
        axes[3], x, y, pne_grids["geo_hhi"], planner_grids["geo_hhi"],
        "Geographic HHI", "HHI", (1 / K, min(1.0, geo_max + 0.05)),
        elev=elev, azim=azim, invert_y_axis=invert_y_axis,
    )

    util_max = max(
        float(np.nanmax(pne_grids["utility_hhi"])),
        float(np.nanmax(planner_grids["utility_hhi"])),
        9 / (8 * K),
    )
    _plot_comparison_surface(
        axes[4], x, y, pne_grids["utility_hhi"], planner_grids["utility_hhi"],
        "Utility HHI", "HHI", (1 / K, util_max + 0.01),
        elev=elev, azim=azim, invert_y_axis=invert_y_axis,
    )

    legend_handles = [
        Patch(facecolor="#2ca02c", edgecolor="none", alpha=0.86,
              label="PNE welfare ratio"),
        Patch(facecolor="#2166ac", edgecolor="none", alpha=0.88,
              label="Coverage gap < 0: PNE higher"),
        Patch(facecolor="#b2182b", edgecolor="none", alpha=0.88,
              label="Coverage gap > 0: Planner higher"),
        Patch(facecolor="#1f77b4", edgecolor="none", alpha=0.68,
              label="PNE HHI"),
        Patch(facecolor="#d62728", edgecolor="none", alpha=0.58,
              label="Planner HHI"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.10, top=0.94)

    if output_stem is None:
        output_stem = f"exp3_k{K}_3d_surfaces_coverage_gap"
    if paper_stem is None:
        paper_stem = "paper_exp3_3d_surfaces_coverage_gap"
    out = FIGURES_DIR / f"{output_stem}.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=180, bbox_inches="tight")
    paper_out = FIGURES_DIR / f"{paper_stem}.pdf"
    fig.savefig(paper_out, bbox_inches="tight")
    fig.savefig(str(paper_out).replace(".pdf", ".png"), dpi=180, bbox_inches="tight")
    print(f"Saved {out}")
    print(f"Saved {paper_out}")


def _save_cache(abr_runs, planner_runs, path):
    np.savez_compressed(
        path,
        abr_runs=np.array(abr_runs, dtype=object),
        planner_runs=np.array(planner_runs, dtype=object),
        value_ratio_grid=VALUE_RATIO_GRID,
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
    missing_regions = [r for r in pool_regions if r not in REGIONS_EXP3]
    missing_coords = [r for r in pool_regions if r not in GCP_REGION_COORDS]
    if missing_regions or missing_coords:
        raise ValueError(
            f"Pool regions misconfigured: "
            f"missing from REGIONS_EXP3={missing_regions}, "
            f"missing from GCP_REGION_COORDS={missing_coords}"
        )

    n_cells = N_RATIO * N_DELTA
    n_runs_per_cell = N_INSTANCES * N_SEEDS_PER_INSTANCE
    n_profiles = comb(len(REGIONS_EXP3) + K - 1, K)
    print(f"Exp 3 (K={K}): ({N_RATIO} ratios) x ({N_DELTA} deltas) = {n_cells} cells")
    print(f"K={K}, {N_INSTANCES} source instances x {N_SEEDS_PER_INSTANCE} "
          f"random inits = {n_runs_per_cell} ABR runs per cell")
    print(f"Total ABR jobs: {n_cells * n_runs_per_cell}")
    print(f"Total planner jobs: {n_cells * N_INSTANCES} "
          f"(opt method: {OPT_METHOD}, profiles per run: {n_profiles:,})")
    print(f"Value ratio grid: {VALUE_RATIO_GRID}")
    print(f"Corresponding alpha grid: {ALPHA_GRID}")
    print(f"Cache: {cache_path} (exists: {cache_exists})")
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

    pne_grids = _build_grids(abr_runs, planner_runs)
    plot_heatmaps(pne_grids)

    planner_grids = _build_planner_grids(planner_runs)
    plot_heatmaps(
        planner_grids,
        output_stem=f"exp3_k{K}_planner_ratio_delta_heatmaps",
        paper_stem="paper_exp3_planner_2x2",
        title_prefix="Planner",
    )
    plot_3d_surfaces(pne_grids, planner_grids)


def _compute(args):
    n_workers = max(1, cpu_count() - 1)

    planner_tasks = [
        (ri, ratio, ALPHA_GRID[ri], di, delta, inst, OPT_METHOD)
        for ri, ratio in enumerate(VALUE_RATIO_GRID)
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
                      f"ratio={p['value_ratio']:.2f}x "
                      f"delta={int(p['delta']*1000)}ms "
                      f"W*={p['w_opt']:.4f}")

    abr_tasks = [
        (ri, ratio, ALPHA_GRID[ri], di, delta, inst, seed)
        for ri, ratio in enumerate(VALUE_RATIO_GRID)
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
                      f"ratio={r['value_ratio']:.2f}x "
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
