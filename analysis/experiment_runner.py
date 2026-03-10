#!/usr/bin/env python3
from typing import List

from sim.config import ExperimentConfig, create_scenario_from_config
from analysis.result import ExperimentResult
from analysis.poa import compute_poa_stats
from sim.simulator import (
    Region, Source, Builder, LocationGamesSimulator,
    EMASoftmaxPolicy, UCBPolicy, StochasticTransactionGenerator,
    LatencyPropagationModel, EqualSplitSharingRule,
)

def get_preset_config(preset_name: str) -> ExperimentConfig:
    """Get a predefined experiment configuration."""

    presets = {
        "small_uniform": ExperimentConfig(
            name="small_uniform",
            n_regions=3,
            region_names=["West", "Central", "East"],
            sources_config=[
                ("Oracle1", 0, 5.0, 1.0, 0.5),
                ("Oracle2", 1, 5.0, 1.0, 0.5),
                ("Oracle3", 2, 5.0, 1.0, 0.5),
            ],
            policy_type="EMA",
            n_builders=6,
            n_slots=5000
        ),

        "large_diverse": ExperimentConfig(
            name="large_diverse",
            n_regions=5,
            region_names=["West", "CentralWest", "Central", "CentralEast", "East"],
            sources_config=[
                ("FastOracle", 0, 8.0, 0.8, 0.4),
                ("BalancedOracle", 2, 5.0, 1.0, 0.5),
                ("PremiumOracle", 4, 3.0, 1.5, 0.6),
            ],
            policy_type="EMA",
            eta=0.12,
            beta_reg=1.5,
            n_builders=8,
            n_slots=10000
        ),

        "ucb_exploration": ExperimentConfig(
            name="ucb_exploration",
            n_regions=5,
            sources_config=[
                ("Source1", 0, 6.0, 1.0, 0.5),
                ("Source2", 2, 4.0, 1.2, 0.5),
                ("Source3", 4, 3.0, 1.5, 0.5),
            ],
            policy_type="UCB",
            alpha=2.0,
            n_builders=8,
            n_slots=10000
        ),

        "high_migration_cost": ExperimentConfig(
            name="high_migration_cost",
            n_regions=5,
            sources_config=[
                ("Source1", 0, 6.0, 1.0, 0.5),
                ("Source2", 4, 6.0, 1.0, 0.5),
            ],
            policy_type="EMA",
            eta=0.1,
            beta_reg=2.0,
            cost_c=0.0,
            n_builders=8,
            n_slots=10000
        )
    }

    if preset_name not in presets:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")

    return presets[preset_name]


def run_experiment(config: ExperimentConfig, verbose: bool = True,
                   compute_poa: bool = False, poa_method: str = 'brute_force') -> ExperimentResult:
    """Run a single experiment with given configuration."""

    if verbose:
        print(f"\n{'='*70}")
        print(f"Running Experiment: {config.name}")
        print(f"{'='*70}")
        print(f"Policy: {config.policy_type}")
        print(f"Regions: {config.n_regions}, Sources: {len(config.sources_config)}")
        print(f"Builders: {config.n_builders}, Slots: {config.n_slots}, Delta: {config.delta}")

    regions, sources, latency_mean, latency_std = create_scenario_from_config(config)

    if verbose:
        print(f"\nSources: {[(s.name, f'lambda={s.lambda_rate}', f'starting region={s.region}') for s in sources]}")
        print(f"\nLatency mean matrix:")
        print(latency_mean, "\n")

    builders = []
    for i in range(config.n_builders):
        if config.policy_type == "EMA":
            policy = EMASoftmaxPolicy(
                config.n_regions,
                eta=config.eta,
                beta=config.beta_reg,
                cost=config.cost_c,
            )
        elif config.policy_type == "UCB":
            policy = UCBPolicy(config.n_regions, alpha=config.alpha)
        else:
            raise ValueError(f"Unknown policy: {config.policy_type}")

        builders.append(Builder(i, policy))

    sim = LocationGamesSimulator(
        regions=regions,
        sources=sources,
        builders=builders,
        tx_generator=StochasticTransactionGenerator(),
        propagation_model=LatencyPropagationModel(latency_mean, latency_std),
        sharing_rule=EqualSplitSharingRule(),
        delta=config.delta,
        seed=config.seed,
    )

    if verbose:
        print(f"\nRunning simulation...")

    sim.run(config.n_slots)

    result = ExperimentResult(config, sim)

    if verbose:
        print_results(result, regions, sources)

    if compute_poa:
        if verbose:
            print(f"\nComputing PoA ({poa_method})...")
        result.poa_stats = compute_poa_stats(result, method=poa_method)
        if verbose:
            p = result.poa_stats
            print(f"W* (optimal): {p['w_star']:.4f}")
            print(f"W  (learned): {p['w_learned']:.4f}")
            print(f"PoA: {p['poa']:.4f}")
            print(f"Optimal profile: {p['opt_profile_names']}")

    if config.save_results:
        result.save()

    return result


def print_results(result: ExperimentResult, regions: List[Region], sources: List[Source]):
    """Print experiment results."""
    stats = result.stats

    print(f"\n{'='*70}")
    print("Results")
    print(f"{'='*70}")
    print(f"Average reward per builder per slot: {stats['avg_reward']:.4f}")
    print(f"Mean welfare per slot: {stats['mean_welfare']:.4f}")
    print(f"Mean txs emitted per round: {stats['mean_txs_emitted_per_round']:.2f}")
    print(f"Mean txs received per round: {stats['mean_txs_received_per_round']:.2f}")
    print(f"Mean coverage ratio: {stats['mean_coverage_ratio']:.4f}")

    print(f"\nBuilder distribution across regions (avg over time):")
    for i, count in enumerate(stats['avg_builder_distribution']):
        print(f"  {regions[i].name:15s}: {count:6.2f} builders")

    print(f"\nRegion selection per slot (avg builders per slot):")
    for i, count in enumerate(stats['avg_region_counts']):
        print(f"  {regions[i].name:15s}: {count:6.2f}")

    print(f"\nDiversity metrics:")
    print(f"  Builder Dist Gini:    {stats['builder_dist_gini']:.4f} (lower = more equal)")
    print(f"  Builder Dist Entropy: {stats['builder_dist_entropy']:.4f} (higher = more equal)")
    print(f"  Region Gini:           {stats['region_gini']:.4f}")
    print(f"  Region Entropy:        {stats['region_entropy']:.4f}")
