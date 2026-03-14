import yaml
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from sim.datasets import gcp_sources, load_gcp, subregion
from sim.simulator import Region, Source

PRIMARY_SEED = 0

def get_seeds(n_runs: int) -> list:
    """Derive a reproducible list of seeds from PRIMARY_SEED."""
    return np.random.default_rng(PRIMARY_SEED).integers(0, 2**32, n_runs).tolist()


@dataclass
class ExperimentConfig:
    """All experiment parameters in one place."""

    # Experiment identification
    name: str = "default_experiment"

    # Regions configuration
    n_regions: int = 5
    region_names: List[str] = None  # If None, will auto-generate

    # Sources configuration
    # Each source: (name, region_id, lambda_rate, mu_val, sigma_val)
    sources_config: List[tuple] = None

    # Latency matrices: shape (n_regions, n_sources), raw seconds
    latency_mean: Optional[np.ndarray] = None
    latency_std: Optional[np.ndarray] = None

    # Policy / dynamic configuration
    policy_type: str = "EMA"  # "EMA", "UCB", "EXP3", "ABR", or "MWU"

    # EMA policy parameters
    eta: float = 0.12
    beta_reg: float = 1.5
    cost_c: float = 0.0

    # UCB policy parameters
    alpha: float = 2.0

    # EXP3 policy parameters
    exp3_gamma: float = 0.07

    # Asynchronous better-response parameters
    improvement_threshold_pct: float = 0.0
    utility_eval_time_steps: int = 200
    abr_max_updates: Optional[int] = None

    # MWU parameters
    mwu_eta: float = 0.1
    payoff_normalization: Optional[float] = None  # Used by EXP3 and MWU reward clipping

    # Simulation parameters
    n_builders: int = 8
    n_slots: int = 10000
    delta: float = 12.0
    n_runs: int = 1

    # Output configuration
    save_results: bool = True
    results_dir: str = "results"

    def __post_init__(self):
        if self.region_names is None:
            self.region_names = [f"Region_{i}" for i in range(self.n_regions)]

        if self.sources_config is None:
            self.sources_config = [
                ("SourceA", 0, 5.0, 1.0, 0.5),
                ("SourceB", self.n_regions // 2, 5.0, 1.0, 0.5),
                ("SourceC", self.n_regions - 1, 5.0, 1.0, 0.5),
            ]

# YAML config loading
def load_config(path) -> ExperimentConfig:
    """
    Load an ExperimentConfig from a YAML file
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    ds  = raw["dataset"]
    sim = raw["simulation"]
    pol = raw["policy"]

    dataset_type = ds["type"]

    if dataset_type in ("gcp_full", "gcp_subset"):
        region_names, latency_mean, latency_std = load_gcp()
        if dataset_type == "gcp_subset":
            keep = ds.get("subset_regions", ds.get("source_regions"))
            region_names, latency_mean, latency_std = subregion(
                region_names, latency_mean, latency_std, keep
            )
        sources_config = gcp_sources(
            region_names,
            ds["source_regions"],
            lambda_rate=ds.get("lambda_rate", 5.0),
            mu_val=ds.get("mu_val", 1.0),
            sigma_val=ds.get("sigma_val", 0.5),
        )
    elif dataset_type == "synthetic":
        region_names = ds["region_names"]
        n = len(region_names)
        dist = np.array([[abs(r - s) for s in range(n)] for r in range(n)], dtype=float)
        latency_mean = 0.1 + 0.05 * dist
        latency_std  = 0.05 + 0.02 * dist
        sources_config = [
            (f"Src_{name}", i, ds.get("lambda_rate", 5.0),
             ds.get("mu_val", 1.0), ds.get("sigma_val", 0.5))
            for i, name in enumerate(region_names)
        ]
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type!r}")

    return ExperimentConfig(
        name=raw["name"],
        n_regions=len(region_names),
        region_names=region_names,
        sources_config=sources_config,
        latency_mean=latency_mean,
        latency_std=latency_std,
        n_builders=sim["n_builders"],
        n_slots=sim["n_slots"],
        delta=sim.get("delta", 12.0),
        n_runs=sim.get("n_runs", 1),
        policy_type=pol["type"],
        eta=pol.get("eta", 0.12),
        beta_reg=pol.get("beta_reg", 1.5),
        cost_c=pol.get("cost_c", 0.0),
        alpha=pol.get("alpha", 2.0),
        exp3_gamma=pol.get("exp3_gamma", 0.07),
        improvement_threshold_pct=pol.get("improvement_threshold_pct", 0.0),
        utility_eval_time_steps=pol.get("utility_eval_time_steps", 200),
        abr_max_updates=pol.get("abr_max_updates"),
        mwu_eta=pol.get("mwu_eta", pol.get("eta", 0.1)),
        payoff_normalization=pol.get("payoff_normalization"),
    )


def create_scenario_from_config(config: ExperimentConfig):
    """Create regions, sources, and latency matrices from config."""
    regions = [Region(i, config.region_names[i]) for i in range(config.n_regions)]

    sources = []
    for i, (name, region_id, lambda_rate, mu_val, sigma_val) in enumerate(config.sources_config):
        sources.append(Source(i, name, region_id, lambda_rate, mu_val, sigma_val))

    n_sources = len(sources)

    if config.latency_mean is not None:
        latency_mean = config.latency_mean
        latency_std = config.latency_std
        # config provides a region*region matrix;
        # the propagation model needs (n_regions, n_sources)
        if latency_mean.shape[1] != n_sources:
            source_cols = [s.region for s in sources]
            latency_mean = latency_mean[:, source_cols]
            latency_std = latency_std[:, source_cols]
    else:
        raise ValueError("Latency values must be provided.")

    return regions, sources, latency_mean, latency_std
