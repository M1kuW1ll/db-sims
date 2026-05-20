# scripts/exp_poa_heatmap_K_alpha.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import matplotlib.pyplot as plt

from scripts.poa_experiment_helpers import (
    REGIONS_DEFAULT,
    load_propagation_model,
    build_two_cluster_sources,
    run_abr,
    compute_optimal,
    FIGURES_DIR,
)


# Slot duration parameterised as a multiple of median pairwise latency
DELTA_LATENCY_RATIO = 1.25

TOTAL_VALUE = 10.0
N_RANDOM_INITS = 8  # random builder initializations per cell
RANDOMIZE_SOURCES = False  # set True to also vary which regions host high-value sources
N_SOURCE_VARIANTS = 5  # used only if RANDOMIZE_SOURCES = True


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def median_pairwise_latency(latency_mean):
    """Median of off-diagonal entries in the latency matrix (in seconds)."""
    n = latency_mean.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(np.median(latency_mean[mask]))


def random_source_assignment(rng, source_regions_pool, n_high, n_distant):
    """Randomly choose n_high regions for high-value cluster and n_distant for distant cluster."""
    chosen = list(rng.choice(source_regions_pool, size=n_high + n_distant, replace=False))
    return chosen[:n_high], chosen[n_high:]


def evaluate_cell_with_random_inits(K, sources, prop, regions, delta,
                                     n_random_inits, seed_base=0):
    """Run ABR from concentrated + multiple random builder initializations.
    Returns (max_poa, mean_poa, std_poa)."""
    n_regions = len(regions)
    w_opt, _ = compute_optimal(K, n_regions, sources, prop, delta, method="auto")

    if w_opt < 1e-12:
        return float("inf"), float("inf"), float("inf")

    # Concentrated baseline
    _, w_concentrated = run_abr(K, sources, prop, regions, delta,
                                 init_placement="concentrated", seed=seed_base)
    poas = [w_opt / w_concentrated if w_concentrated > 1e-12 else float("inf")]

    # Random initialisations
    for k in range(n_random_inits):
        seed = seed_base + 1 + k
        try:
            _, w_random = run_abr(K, sources, prop, regions, delta,
                                  init_placement="random", seed=seed)
        except Exception:
            # Fallback if "random" init isnt supported
            _, w_random = run_abr(K, sources, prop, regions, delta,
                                  init_placement="dispersed", seed=seed)
        poa = w_opt / w_random if w_random > 1e-12 else float("inf")
        poas.append(poa)

    return max(poas), float(np.mean(poas)), float(np.std(poas))


def main():
    regions, prop, region_index_map = load_propagation_model(REGIONS_DEFAULT)

    median_lat = median_pairwise_latency(prop.latency_mean)
    delta_slot = DELTA_LATENCY_RATIO * median_lat

    print(f"Median pairwise latency: {median_lat*1000:.1f} ms")
    print(f"Delta = {DELTA_LATENCY_RATIO} * median latency = {delta_slot*1000:.1f} ms = {delta_slot:.4f}s")
    print()

    K_grid = np.arange(3, 13)
    alpha_grid = np.linspace(0.5, 0.95, 10)

    poa_max_grid = np.zeros((len(K_grid), len(alpha_grid)))
    poa_mean_grid = np.zeros((len(K_grid), len(alpha_grid)))
    poa_std_grid = np.zeros((len(K_grid), len(alpha_grid)))

    for i, K in enumerate(K_grid):
        for j, alpha in enumerate(alpha_grid):
            cell_results_max = []

            if RANDOMIZE_SOURCES:
                for src_seed in range(N_SOURCE_VARIANTS):
                    src_rng = np.random.default_rng(src_seed)
                    high_regions, distant_regions = random_source_assignment(
                        src_rng,
                        list(region_index_map.keys()),
                        n_high=5, n_distant=5,
                    )
                    sources = build_two_cluster_sources(
                        alpha, TOTAL_VALUE, region_index_map,
                        high_value_regions=high_regions,
                        distant_regions=distant_regions,
                    )
                    poa_max, _, _ = evaluate_cell_with_random_inits(
                        K, sources, prop, regions, delta_slot,
                        n_random_inits=N_RANDOM_INITS,
                        seed_base=src_seed * 100,
                    )
                    cell_results_max.append(poa_max)

                poa_max_grid[i, j] = float(np.max(cell_results_max))
                poa_mean_grid[i, j] = float(np.mean(cell_results_max))
                poa_std_grid[i, j] = float(np.std(cell_results_max))
            else:
                sources = build_two_cluster_sources(
                    alpha, TOTAL_VALUE, region_index_map,
                )
                poa_max, poa_mean, poa_std = evaluate_cell_with_random_inits(
                    K, sources, prop, regions, delta_slot,
                    n_random_inits=N_RANDOM_INITS,
                )
                poa_max_grid[i, j] = poa_max
                poa_mean_grid[i, j] = poa_mean
                poa_std_grid[i, j] = poa_std

            print(f"K={K:2d}  alpha={alpha:.2f}  "
                  f"PoA(max)={poa_max_grid[i,j]:.3f}  "
                  f"PoA(mean)={poa_mean_grid[i,j]:.3f} ± {poa_std_grid[i,j]:.3f}")

    # Plot: side-by-side max and mean
    fig, (ax_max, ax_mean) = plt.subplots(1, 2, figsize=(14, 5.5))

    vmax = max(np.max(poa_max_grid), 1.05)

    extent = [
        alpha_grid[0] - (alpha_grid[1] - alpha_grid[0])/2,
        alpha_grid[-1] + (alpha_grid[1] - alpha_grid[0])/2,
        K_grid[0] - 0.5,
        K_grid[-1] + 0.5,
    ]

    for ax, grid, label in [
        (ax_max, poa_max_grid, "Worst-case PoA across random inits"),
        (ax_mean, poa_mean_grid, "Mean PoA across random inits"),
    ]:
        im = ax.imshow(
            grid,
            aspect="auto",
            origin="lower",
            cmap="YlOrRd",
            vmin=1.0,
            vmax=vmax,
            extent=extent,
        )
        for i, K in enumerate(K_grid):
            for j, alpha in enumerate(alpha_grid):
                value = grid[i, j]
                text_color = "white" if value > (1.0 + vmax) / 2 else "black"
                ax.text(alpha, K, f"{value:.2f}",
                        ha="center", va="center", fontsize=8, color=text_color)

        ax.set_xticks(alpha_grid)
        ax.set_xticklabels([f"{a:.2f}" for a in alpha_grid], rotation=30)
        ax.set_yticks(K_grid)
        ax.set_xlabel(r"Value asymmetry $\alpha$ (fraction in high-value cluster)")
        ax.set_ylabel("Number of builders $K$")
        ax.set_title(label)
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("PoA")

    title_suffix = (
        f"random source assignments + random builder inits"
        if RANDOMIZE_SOURCES
        else f"{N_RANDOM_INITS+1} random builder inits per cell"
    )
    fig.suptitle(
        rf"Empirical PoA across $(K, \alpha)$ — ABR robustness check"
        rf" ($\Delta = {DELTA_LATENCY_RATIO}\times$ median latency $= {delta_slot*1000:.0f}$ ms, {title_suffix})",
        fontsize=12,
    )

    fig.tight_layout()
    fname = "exp_poa_heatmap_K_alpha"
    if RANDOMIZE_SOURCES:
        fname += "_with_source_randomization"
    fig.savefig(FIGURES_DIR / f"{fname}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{fname}.png", dpi=200, bbox_inches="tight")
    print(f"\nSaved {FIGURES_DIR / fname}.pdf and .png")


if __name__ == "__main__":
    main()
