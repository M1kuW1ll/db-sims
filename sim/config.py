import yaml
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from sim.datasets import gcp_sources, load_gcp, subregion
from sim.simulator import ABR_RESPONSE_RULES, ABR_UPDATE_MODES, Region, Source

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
    # Propagation model: "lognormal" for stochastic (GCP), "fixed" for deterministic (synthetic)
    propagation_model_type: str = "lognormal"

    # Policy configuration
    policy_type: str = "ABR"

    # EXP3 policy parameters
    eta: float = 0.12
    gamma: float = 0.05  # uniform exploration mixing parameter
    gamma_schedule: str = "static"  # "static" | "exponential" | "sqrt_decay" | "linear"
    gamma_min: float = 0.01  # exploration floor for decaying schedules
    gamma_decay: float = 0.0002  # rate constant for exponential schedule
    norm_alpha: float = 0.0  # exponential moving average rate for adaptive normalisation (0 = disabled)
    payoff_normalization: Optional[float] = None  # Optional EXP3 payoff scale before clipping to [0, 1]

    # Exact-response parameters
    improvement_threshold_pct: float = 0.0
    utility_eval_time_steps: int = 200
    abr_max_updates: Optional[int] = None
    abr_max_rounds: Optional[int] = None
    n_t: int = 100  # number of time discretisation points for the analytical integral
    abr_response_rule: str = "best"
    abr_update_mode: str = "async"

    # Simulation parameters
    n_builders: int = 8
    n_slots: int = 10000
    delta: float = 12.0
    n_runs: int = 1
    initial_placement: str = "dispersed"  # "dispersed", "random", or "concentrated"
    placement_seed: int = PRIMARY_SEED

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

        self.abr_response_rule = self.abr_response_rule.lower()
        if self.abr_response_rule not in ABR_RESPONSE_RULES:
            raise ValueError(
                f"Unknown ABR response rule: {self.abr_response_rule!r}. "
                "Expected 'best' or 'better'."
            )
        self.abr_update_mode = self.abr_update_mode.lower()
        if self.abr_update_mode not in ABR_UPDATE_MODES:
            raise ValueError(
                f"Unknown ABR update mode: {self.abr_update_mode!r}. "
                "Expected 'async' or 'simultaneous'."
            )

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
        topology = ds.get("latency_topology", "linear")
        if topology == "linear":
            dist = np.array([[abs(r - s) for s in range(n)] for r in range(n)], dtype=float)
        elif topology == "flat":
            # All off-diagonal pairs have the same distance (fully symmetric)
            dist = np.ones((n, n), dtype=float)
            np.fill_diagonal(dist, 0.0)
        elif topology == "zero":
            # Instant propagation everywhere
            dist = np.zeros((n, n), dtype=float)
        else:
            raise ValueError(f"Unknown latency_topology: {topology!r}. Use 'linear', 'flat', or 'zero'.")
        latency_mean_base = ds.get("latency_mean_base", 0.1)
        latency_mean_scale = ds.get("latency_mean_scale", 0.05)
        latency_std_base = ds.get("latency_std_base", 0.05)
        latency_std_scale = ds.get("latency_std_scale", 0.02)
        latency_mean = latency_mean_base + latency_mean_scale * dist
        latency_std  = latency_std_base  + latency_std_scale  * dist
        source_region_indices = ds.get("source_regions", list(range(n)))
        sources_config = [
            (f"Src_{region_names[i]}", i, ds.get("lambda_rate", 5.0),
             ds.get("mu_val", 1.0), ds.get("sigma_val", 0.5))
            for i in source_region_indices
        ]
        latency_mean = latency_mean[:, source_region_indices]
        latency_std  = latency_std[:, source_region_indices]
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type!r}")

    propagation_model_type = "fixed" if dataset_type == "synthetic" else "lognormal"

    return ExperimentConfig(
        name=raw["name"],
        n_regions=len(region_names),
        region_names=region_names,
        sources_config=sources_config,
        latency_mean=latency_mean,
        latency_std=latency_std,
        propagation_model_type=propagation_model_type,
        n_builders=sim["n_builders"],
        n_slots=sim["n_slots"],
        delta=sim.get("delta", 12.0),
        n_runs=sim.get("n_runs", 1),
        policy_type=pol["type"],
        eta=pol.get("eta", 0.12),
        gamma=pol.get("gamma", pol.get("exp3_gamma", 0.05)),
        gamma_schedule=pol.get("gamma_schedule", "static"),
        gamma_min=pol.get("gamma_min", 0.01),
        gamma_decay=pol.get("gamma_decay", 0.0002),
        norm_alpha=pol.get("norm_alpha", 0.0),
        payoff_normalization=pol.get("payoff_normalization"),
        improvement_threshold_pct=pol.get("improvement_threshold_pct", 0.0),
        utility_eval_time_steps=pol.get("utility_eval_time_steps", pol.get("n_t", 200)),
        abr_max_updates=pol.get("abr_max_updates"),
        abr_max_rounds=pol.get("abr_max_rounds"),
        n_t=pol.get("n_t", pol.get("utility_eval_time_steps", 100)),
        abr_response_rule=pol.get("abr_response_rule", "best"),
        abr_update_mode=pol.get("abr_update_mode", "async"),
        placement_seed=sim.get("placement_seed", PRIMARY_SEED),
        initial_placement=sim.get("initial_placement", "dispersed"),
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
