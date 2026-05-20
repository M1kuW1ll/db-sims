# scripts/exp_helpers.py
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathlib import Path
import numpy as np

from sim.datasets import load_gcp, subregion
from sim.metrics import hhi
from sim.simulator import (
    Source, Builder, Region, FixedLatencyPropagationModel, LatencyPropagationModel,
    PropagationModel,
    StochasticTransactionGenerator, EqualSplitSharingRule,
    LocationGamesSimulator, FixedPolicy,
    compute_all_builder_utilities, compute_expected_reward,
)
from analysis.poa import optimal_welfare_brute_force, optimal_welfare_greedy, _compute_welfare_analytical
from functools import lru_cache
from itertools import combinations
from scipy.optimize import linear_sum_assignment

_D_MAX_DISK_CACHE_PATH = Path(__file__).resolve().parent.parent / "results" / "d_max_cache.npz"

REGIONS_DEFAULT = [
    # US/EU high-value cluster
    "us-east1", "us-east4", "us-central1", "us-west1",
    "europe-west1", "europe-west3", "europe-west4", "europe-north1",
    # Asia/Pacific/SA/Africa distant cluster
    "asia-northeast1", "asia-southeast1", "asia-south1",
    "australia-southeast1", "australia-southeast2",
    "southamerica-east1", "southamerica-west1",
    "africa-south1",
    # Intermediate
    "me-central1", "me-west1", "europe-southwest1", "europe-central2",
]

HIGH_VALUE_SOURCE_REGIONS_DEFAULT = [
    "us-east1", "us-east4", "europe-west1", "europe-west3", "europe-west4",
]
DISTANT_SOURCE_REGIONS_DEFAULT = [
    "asia-northeast1", "asia-southeast1", "australia-southeast1",
    "southamerica-east1", "africa-south1",
]


FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def load_propagation_model(regions=REGIONS_DEFAULT, latency_std_fraction=0.15):
    """Returns (regions, propagation_model, region_index_map).
    Uses LatencyPropagationModel (lognormal) so that reception is stochastic.
    """
    all_regions, full_mean, full_std = load_gcp(latency_std_fraction)
    _, latency_mean, latency_std = subregion(all_regions, full_mean, full_std, regions)
    prop = LatencyPropagationModel(latency_mean, latency_std)
    region_index_map = {r: i for i, r in enumerate(regions)}
    return regions, prop, region_index_map


def build_two_cluster_sources(alpha, total_value, region_index_map,
                              high_value_regions=HIGH_VALUE_SOURCE_REGIONS_DEFAULT,
                              distant_regions=DISTANT_SOURCE_REGIONS_DEFAULT,
                              lambda_rate=1.0, sigma_val=0.0):
    """
    Build sources split between a high-value cluster and a distant cluster.

    alpha: fraction of total_value in the high-value cluster.
    Each cluster's value is divided equally among its sources.
    """
    n_high = len(high_value_regions)
    n_distant = len(distant_regions)
    value_per_high = total_value * alpha / n_high
    value_per_distant = total_value * (1 - alpha) / n_distant

    sources = []
    sid = 0
    for r in high_value_regions:
        sources.append(Source(
            id=sid, name=f"src_{r}", region=region_index_map[r],
            lambda_rate=lambda_rate, mu_val=np.log(value_per_high),
            sigma_val=sigma_val,
        ))
        sid += 1
    for r in distant_regions:
        sources.append(Source(
            id=sid, name=f"src_{r}", region=region_index_map[r],
            lambda_rate=lambda_rate, mu_val=np.log(value_per_distant),
            sigma_val=sigma_val,
        ))
        sid += 1
    return sources

def build_high_background_sources(alpha, total_value, region_index_map,
                                  high_value_regions,
                                  background_regions,
                                  lambda_rate=1.0, sigma_val=0.0):
    """
    Build sources split between high-value sources and background sources.

    alpha: fraction of total_value assigned to the high-value sources.
    The remaining 1-alpha fraction is assigned to background sources.
    Each group's value is divided equally among its sources.

    Source IDs are ordered as:
      0 .. n_high-1                  high-value sources
      n_high .. n_high+n_bg-1        background sources
    """
    n_high = len(high_value_regions)
    n_background = len(background_regions)
    if n_high <= 0:
        raise ValueError("high_value_regions must contain at least one region")
    if n_background <= 0:
        raise ValueError("background_regions must contain at least one region")

    value_per_high = total_value * alpha / n_high
    value_per_background = total_value * (1 - alpha) / n_background

    sources = []
    sid = 0
    for r in high_value_regions:
        sources.append(Source(
            id=sid, name=f"high_{r}", region=region_index_map[r],
            lambda_rate=lambda_rate, mu_val=np.log(value_per_high),
            sigma_val=sigma_val,
        ))
        sid += 1
    for r in background_regions:
        sources.append(Source(
            id=sid, name=f"bg_{r}", region=region_index_map[r],
            lambda_rate=lambda_rate, mu_val=np.log(value_per_background),
            sigma_val=sigma_val,
        ))
        sid += 1
    return sources

def compute_optimal(K, n_regions, sources, prop, delta, n_t=200,
                    method="auto", max_brute=100_000):
    """Auto-select brute-force or greedy depending on problem size."""
    if method == "greedy":
        return optimal_welfare_greedy(K, n_regions, sources, prop, delta, n_t)
    if method == "brute":
        return optimal_welfare_brute_force(K, n_regions, sources, prop, delta, n_t)

    from math import comb
    n_profiles = comb(n_regions + K - 1, K)
    if n_profiles <= max_brute:
        return optimal_welfare_brute_force(K, n_regions, sources, prop, delta, n_t)
    return optimal_welfare_greedy(K, n_regions, sources, prop, delta, n_t)

def run_abr(K, sources, prop, regions, delta, init_placement,
            n_slots=200, n_t=200, seed=42):
    """Run ABR to convergence; returns (final_profile, converged_welfare)."""
    n_regions = len(regions)
    regions_list = [Region(id=i, name=regions[i]) for i in range(n_regions)]
    builders = [Builder(id=i, policy=FixedPolicy(n_regions)) for i in range(K)]
    sim = LocationGamesSimulator(
        regions=regions_list, sources=sources, builders=builders,
        tx_generator=StochasticTransactionGenerator(),
        propagation_model=prop, sharing_rule=EqualSplitSharingRule(),
        delta=delta, seed=seed, initial_placement=init_placement,
    )
    sim.run_abr(n_slots=n_slots, n_t=n_t)
    final_profile = [b.current_region for b in sim.builders]
    w_converged = _compute_welfare_analytical(
        final_profile, sources, prop, delta, n_t,
    )
    return final_profile, w_converged


def evaluate_instance(K, sources, prop, regions, delta, n_slots=200, n_t=200,
                      opt_method="auto"):
    """Run ABR from both inits, compute optimum, return PoA for each."""
    n_regions = len(regions)
    w_opt, opt_profile = compute_optimal(
        K, n_regions, sources, prop, delta, n_t, method=opt_method,
    )
    profile_c, w_c = run_abr(K, sources, prop, regions, delta,
                             init_placement="concentrated",
                             n_slots=n_slots, n_t=n_t)
    profile_d, w_d = run_abr(K, sources, prop, regions, delta,
                             init_placement="dispersed",
                             n_slots=n_slots, n_t=n_t)
    return {
        "w_opt": w_opt, "opt_profile": opt_profile,
        "w_concentrated": w_c, "profile_concentrated": profile_c,
        "w_dispersed": w_d, "profile_dispersed": profile_d,
        "poa_concentrated": w_opt / w_c if w_c > 1e-12 else float("inf"),
        "poa_dispersed": w_opt / w_d if w_d > 1e-12 else float("inf"),
    }


def plot_poa_curve(x, poa_concentrated, poa_dispersed, K, x_label, title,
                   filename, log_x=False):
    """Standard two-line PoA plot with theoretical bound annotation."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, poa_concentrated, "o-", color="#d62728",
            label="ABR from concentrated init")
    ax.plot(x, poa_dispersed, "s-", color="#1f77b4",
            label="ABR from dispersed init")
    ax.axhline(2 - 1/K, ls="--", color="gray",
               label=f"Theoretical bound: $2 - 1/K = {2 - 1/K:.2f}$")
    ax.axhline(1.0, ls=":", color="black", alpha=0.5)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Empirical PoA")
    ax.set_title(title)
    ax.set_ylim(0.95, 2 - 1/K + 0.1)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename)
    print(f"Saved {FIGURES_DIR / filename}")

# High-value cluster: major financial centre regions
HIGH_VALUE_SOURCE_REGIONS = [
    "us-east1",  # South Carolina
    "us-east4",  # Virginia
    "europe-west1",  # Belgium
    "europe-west3",  # Frankfurt
    "asia-northeast1",  # Tokyo
]

# Peripheral cluster: geographically distant from financial centres
PERIPHERAL_SOURCE_REGIONS = [
    "southamerica-east1",  # Sao Paulo
    "africa-south1", # Johannesburg
    "australia-southeast1", # Sydney
    "asia-south1", # Mumbai
    "us-west1", # Oregon
]

# GCP region coordinates (lat, lon) for great-circle distance computation
GCP_REGION_COORDS: dict = {
    "us-east1": (33.84, -84.12),
    "us-east4": (38.68, -77.30),
    "us-central1": (41.26, -95.86),
    "us-west1": (45.60, -121.18),
    "us-west2": (34.05, -118.24),
    "europe-west1": (50.44, 3.82),
    "europe-west2": (51.51, -0.13),
    "europe-west3": (50.12, 8.68),
    "europe-west4": (53.44, 6.84),
    "europe-north1": (60.57, 27.18),
    "asia-northeast1": (35.69, 139.69),
    "asia-northeast2": (34.69, 135.50),
    "asia-southeast1": (1.35, 103.82),
    "asia-south1": (19.08, 72.88),
    "asia-south2": (28.70, 77.10),
    "australia-southeast1": (-33.87, 151.21),
    "australia-southeast2": (-37.81, 144.97),
    "southamerica-east1": (-23.55, -46.63),
    "southamerica-west1": (-33.45, -70.67),
    "africa-south1": (-26.20, 28.04),
    "me-central1": (25.20, 55.27),
    "me-west1": (32.08, 34.78),
    "europe-southwest1": (38.72, -9.14),
    "europe-central2": (52.23, 21.01),
}


# Geographic metric helpers

def great_circle_km(name1: str, name2: str,
                    coords: dict = GCP_REGION_COORDS) -> float:
    """Great-circle distance in km between two named GCP regions."""
    lat1, lon1 = math.radians(coords[name1][0]), math.radians(coords[name1][1])
    lat2, lon2 = math.radians(coords[name2][0]), math.radians(coords[name2][1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return 6371.0 * 2.0 * math.asin(min(1.0, math.sqrt(a)))


def mean_pairwise_distance_km(profile: list, region_names: list,
                               coords: dict = GCP_REGION_COORDS) -> float:
    """Mean great-circle distance (km) over all unordered builder pairs."""
    K = len(profile)
    if K < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(K):
        for j in range(i + 1, K):
            total += great_circle_km(region_names[profile[i]],
                                     region_names[profile[j]], coords)
            count += 1
    return total / count

def max_mean_pairwise_distance_km(K: int, region_names: tuple,
                                   coords: dict = GCP_REGION_COORDS) -> float:
    """Maximum mean pairwise great-circle distance achievable by placing K
    builders across region_names. Brute force over all K-subsets, cached
    in-memory (lru_cache) and on disk (pickle).
    """
    if K < 2:
        return 0.0
    # Disk-cache lookup
    cache = _load_d_max_disk_cache()
    key = (K, region_names)
    if key in cache:
        return cache[key]
    # Compute (also populates lru_cache)
    result = _max_mean_pairwise_distance_km_cached(K, region_names)
    cache[key] = result
    _save_d_max_disk_cache(cache)
    return result


def _load_d_max_disk_cache() -> dict:
    if _D_MAX_DISK_CACHE_PATH.exists():
        try:
            npz = np.load(_D_MAX_DISK_CACHE_PATH, allow_pickle=True)
            return npz["cache"].item()
        except Exception:
            return {}
    return {}


def _save_d_max_disk_cache(cache: dict) -> None:
    _D_MAX_DISK_CACHE_PATH.parent.mkdir(exist_ok=True)
    np.savez_compressed(_D_MAX_DISK_CACHE_PATH, cache=np.array(cache, dtype=object))

@lru_cache(maxsize=None)
def _max_mean_pairwise_distance_km_cached(K: int, region_names: tuple) -> float:
    """Cached brute-force search over K-subsets of region_names."""
    coords = GCP_REGION_COORDS
    best = 0.0
    n_pairs = K * (K - 1) / 2
    for subset in combinations(region_names, K):
        total = 0.0
        for i in range(K):
            for j in range(i + 1, K):
                total += great_circle_km(subset[i], subset[j], coords)
        mean = total / n_pairs
        if mean > best:
            best = mean
    return best

def placement_emd_km(profile_a: list, profile_b: list, region_names: list,
                      coords: dict = GCP_REGION_COORDS) -> float:
    """Earth Mover's Distance between two builder placements, in km per
    builder
 
    Finds the assignment of builders in profile_a to builders in profile_b
    that minimises total great-circle distance, using the Hungarian
    algorithm. Returns sum of distances along the optimal matching, divided
    by K.
 
    Both profiles must have the same length K. Order within each profile
    doesn't matter
    """
    K = len(profile_a)
    if K == 0 or K != len(profile_b):
        return 0.0
    cost = np.zeros((K, K))
    for i in range(K):
        ri = region_names[profile_a[i]]
        for j in range(K):
            rj = region_names[profile_b[j]]
            cost[i, j] = great_circle_km(ri, rj, coords)
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(cost[row_ind, col_ind].sum()) / K
 
 
def placement_multiset_jaccard(profile_a: list, profile_b: list) -> float:
    """Multiset Jaccard similarity between two builder placements.
 
    For each region, take the min/max occupancy across the two profiles.
    Returns sum(min) / sum(max), in [0, 1]
    """
    from collections import Counter
    if len(profile_a) == 0 or len(profile_b) == 0:
        return 0.0
    ca, cb = Counter(profile_a), Counter(profile_b)
    regions = set(ca) | set(cb)
    inter = sum(min(ca.get(r, 0), cb.get(r, 0)) for r in regions)
    union = sum(max(ca.get(r, 0), cb.get(r, 0)) for r in regions)
    return inter / union if union > 0 else 0.0
 

def geo_hhi(profile: list, n_regions: int) -> float:
    """Geographic HHI: sum of squared region-occupancy fractions."""
    counts = np.bincount(profile, minlength=n_regions).astype(float)
    return hhi(counts)


# Source-cluster coverage helpers

def make_sliced_prop(sources: list, prop: PropagationModel) -> PropagationModel:
    """Return a propagation model with columns sliced to source regions.

    The full latency matrix has shape (n_regions, n_regions); after slicing it
    has shape (n_regions, n_sources) where column j corresponds to source j
    (source.id == j is then the correct column index). Returns the same
    concrete class as the input (LatencyPropagationModel or
    FixedLatencyPropagationModel).
    """
    source_cols = [s.region for s in sources]
    if isinstance(prop, LatencyPropagationModel):
        # LatencyPropagationModel stores _mu_ln, _sigma_ln internally and the
        # constructor expects raw mean/std
        mu_ln = prop._mu_ln[:, source_cols]
        sigma_ln = prop._sigma_ln[:, source_cols]
        # E[X] = exp(mu + sigma^2/2); Var[X] = (exp(sigma^2)-1) * exp(2 mu + sigma^2)
        latency_mean = np.exp(mu_ln + sigma_ln ** 2 / 2)
        latency_var = (np.exp(sigma_ln ** 2) - 1.0) * np.exp(2 * mu_ln + sigma_ln ** 2)
        latency_std = np.sqrt(latency_var)
        return LatencyPropagationModel(latency_mean, latency_std)
    if isinstance(prop, FixedLatencyPropagationModel):
        return FixedLatencyPropagationModel(prop.latency_mean[:, source_cols])
    raise TypeError(f"Unsupported propagation model type: {type(prop)}")


def source_coverage_analytical(source_idx: int, active_regions: list,
                                prop: PropagationModel,
                                delta: float, n_t: int = 200) -> float:
    """Expected coverage f_bar_I(s) = int rho_I(t) [1 - prod_b (1 - q(s_b, t))] dt.

    Works for any PropagationModel (fixed or stochastic latency). Assumes
    uniform emission density rho_I(t) = 1/delta.

    source_idx must equal source.id (column index in the (sliced) prop matrix).
    active_regions: list of region indices, one per builder (duplicates allowed).
    """
    if not active_regions:
        return 0.0
    t_points = np.linspace(0, delta, n_t, endpoint=False)
    remaining = delta - t_points
    # For each builder, q(s_b, t) at every time point.
    coverage_at_t = np.zeros(n_t)
    for n, rem in enumerate(remaining):
        prod_miss = 1.0
        for r in active_regions:
            q = prop.reception_prob(r, source_idx, rem)
            prod_miss *= (1.0 - q)
        coverage_at_t[n] = 1.0 - prod_miss
    return float(np.mean(coverage_at_t))


def cluster_coverage_fraction(profile: list, sources: list,
                               prop: PropagationModel,
                               delta: float, source_ids: list,
                               n_t: int = 200) -> float:
    """Fraction of expected value from a cluster of sources that is covered.

    source_ids: list of source.id values (= column indices in prop).
    Works for any PropagationModel.
    """
    total_weight = covered_weight = 0.0
    for sid in source_ids:
        src = sources[sid]
        ev = np.exp(src.mu_val + src.sigma_val ** 2 / 2)
        weight = src.lambda_rate * delta * ev
        f = source_coverage_analytical(sid, profile, prop, delta, n_t)
        total_weight += weight
        covered_weight += weight * f
    return covered_weight / total_weight if total_weight > 1e-12 else 0.0


# Initialisation helpers for Experiment 3

def best_response_region(sources: list, prop: PropagationModel,
                          delta: float, n_regions: int, n_t: int = 200) -> int:
    """Region that maximises a lone builder's analytical expected reward (monopolist)."""
    # Lone builder: sharing weight = 1 for every (source, time) - no competition.
    weights_empty = np.ones((len(sources), n_t))
    rewards = [
        compute_expected_reward(r, weights_empty, sources, prop, delta, n_t)
        for r in range(n_regions)
    ]
    return int(np.argmax(rewards))


def farthest_first_regions(K: int, n_regions: int,
                            full_latency_mean: np.ndarray) -> list:
    """Greedy farthest-first selection of K regions using the latency matrix.

    Starts from the region with maximum total outgoing latency (most peripheral)
    and repeatedly adds the region farthest from the current selected set.
    """
    start = int(np.argmax(full_latency_mean.sum(axis=1)))
    selected = [start]
    while len(selected) < K:
        min_dists = np.min(full_latency_mean[selected, :], axis=0)
        for r in selected:
            min_dists[r] = -1.0  # exclude already selected
        selected.append(int(np.argmax(min_dists)))
    return selected

def run_abr_full(K: int, sources: list, sliced_prop: PropagationModel,
                 regions: list, delta: float, init_regions: list, seed: int,
                 n_t: int = 100, max_rounds: int = 4000, n_t_final: int = 200,
                 n_high_sources: int = 5) -> dict:
    """Run ABR to convergence from an explicit initial placement.

    sliced_prop must be source-sliced (shape (n_regions, n_sources)) so that
    source.id is the correct column index. Works for either lognormal or
    fixed-latency propagation models.

    Returns a dict with welfare, geo_hhi, mean_pairwise_km, utility_hhi,
    cov_high, cov_peripheral, and final_profile.
    """
    n_regions = len(regions)
    regions_list = [Region(id=i, name=r) for i, r in enumerate(regions)]
    builders = [Builder(id=i, policy=FixedPolicy(n_regions)) for i in range(K)]

    sim = LocationGamesSimulator(
        regions=regions_list, sources=sources, builders=builders,
        tx_generator=StochasticTransactionGenerator(),
        propagation_model=sliced_prop, sharing_rule=EqualSplitSharingRule(),
        delta=delta, seed=seed, initial_placement="random",
    )
    for i, b in enumerate(sim.builders):
        b.set_region(init_regions[i])

    np.random.seed(seed)
    sim.run_abr_until_convergence(n_t=n_t, max_rounds=max_rounds)

    final_profile = [b.current_region for b in sim.builders]
    welfare = _compute_welfare_analytical(final_profile, sources, sliced_prop, delta, n_t_final)
    utilities = compute_all_builder_utilities(
        final_profile, sources, sliced_prop, delta, n_t_final
    )

    high_ids = list(range(n_high_sources))
    peri_ids = list(range(n_high_sources, len(sources)))

    return {
        "final_profile": final_profile,
        "welfare": welfare,
        "geo_hhi": geo_hhi(final_profile, n_regions),
        "mean_pairwise_km": mean_pairwise_distance_km(final_profile, list(regions)),
        "utility_hhi": hhi(utilities),
        "utilities": utilities.tolist(),
        "cov_high": cluster_coverage_fraction(
            final_profile, sources, sliced_prop, delta, high_ids, n_t=n_t_final),
        "cov_peripheral": cluster_coverage_fraction(
            final_profile, sources, sliced_prop, delta, peri_ids, n_t=n_t_final),
    }


def run_abr_seeds(K: int, sources: list, sliced_prop: PropagationModel,
                  regions: list, delta: float, n_seeds: int = 10,
                  n_t: int = 100, max_rounds: int = 4000, n_t_final: int = 200,
                  seed_base: int = 0) -> list:
    """Run ABR from n_seeds independent random initialisations. Returns list of result dicts."""
    n_regions = len(regions)
    results = []
    for seed in range(seed_base, seed_base + n_seeds):
        rng = np.random.default_rng(seed)
        init_regions = [int(rng.integers(0, n_regions)) for _ in range(K)]
        r = run_abr_full(K, sources, sliced_prop, regions, delta, init_regions,
                         seed=seed, n_t=n_t, max_rounds=max_rounds, n_t_final=n_t_final)
        results.append(r)
    return results


def compute_opt_sliced(K: int, sources: list, sliced_prop: PropagationModel,
                        n_regions: int, delta: float, n_t: int = 200,
                        method: str = "auto") -> tuple:
    """Compute optimal welfare using a source-sliced propagation model."""
    return compute_optimal(K, n_regions, sources, sliced_prop, delta, n_t, method)
