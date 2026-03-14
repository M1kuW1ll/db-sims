#!/usr/bin/env python3
import numpy as np
from typing import List

from sim.config import ExperimentConfig, create_scenario_from_config, get_seeds
from analysis.result import ExperimentResult
from analysis.poa import compute_poa_stats
from sim.simulator import (
    Region, Source, Builder, LocationGamesSimulator,
    EMASoftmaxPolicy, EXP3Policy, FixedPolicy, UCBPolicy, StochasticTransactionGenerator,
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

        "exp3_exploration": ExperimentConfig(
            name="exp3_exploration",
            n_regions=5,
            sources_config=[
                ("Source1", 0, 6.0, 1.0, 0.5),
                ("Source2", 2, 4.0, 1.2, 0.5),
                ("Source3", 4, 3.0, 1.5, 0.5),
            ],
            policy_type="EXP3",
            exp3_gamma=0.07,
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
        ),

        "abr_exact": ExperimentConfig(
            name="abr_exact",
            n_regions=5,
            sources_config=[
                ("Source1", 0, 6.0, 1.0, 0.5),
                ("Source2", 2, 4.0, 1.2, 0.5),
                ("Source3", 4, 3.0, 1.5, 0.5),
            ],
            policy_type="ABR",
            improvement_threshold_pct=0.0,
            utility_eval_time_steps=200,
            abr_max_updates=5000,
            n_builders=8,
            n_slots=5000,
        ),

        "mwu_baseline": ExperimentConfig(
            name="mwu_baseline",
            n_regions=5,
            sources_config=[
                ("Source1", 0, 6.0, 1.0, 0.5),
                ("Source2", 2, 4.0, 1.2, 0.5),
                ("Source3", 4, 3.0, 1.5, 0.5),
            ],
            policy_type="MWU",
            mwu_eta=0.1,
            n_builders=8,
            n_slots=5000,
        )
    }

    if preset_name not in presets:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")

    return presets[preset_name]


def _run_single(config: ExperimentConfig, seed: int,
                regions, sources, latency_mean, latency_std,
                initial_belief) -> ExperimentResult:
    """Run one simulation instance with the given seed and return its result."""
    builders = []
    for i in range(config.n_builders):
        if config.policy_type == "EMA":
            policy = EMASoftmaxPolicy(
                config.n_regions,
                eta=config.eta,
                beta=config.beta_reg,
                cost=config.cost_c,
                initial_belief=initial_belief,
            )
        elif config.policy_type == "UCB":
            policy = UCBPolicy(
                config.n_regions,
                alpha=config.alpha,
                cost=config.cost_c,
                initial_belief=initial_belief,
            )
        elif config.policy_type == "EXP3":
            payoff_normalization = (
                config.payoff_normalization
                if config.payoff_normalization is not None and config.payoff_normalization > 0
                else initial_belief
            )
            policy = EXP3Policy(
                config.n_regions,
                gamma=config.exp3_gamma,
                payoff_normalization=payoff_normalization,
                initial_belief=initial_belief,
            )
        elif config.policy_type in {"ABR", "MWU"}:
            policy = FixedPolicy(config.n_regions, initial_belief=initial_belief)
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
        seed=seed,
    )
    if config.policy_type in {"EMA", "UCB", "EXP3"}:
        sim.run(config.n_slots)
    elif config.policy_type == "ABR":
        sim.run_async_better_response(
            config.n_slots,
            improvement_threshold_pct=config.improvement_threshold_pct,
            n_time_steps=config.utility_eval_time_steps,
            max_updates=config.abr_max_updates,
        )
    elif config.policy_type == "MWU":
        sim.run_mwu(
            config.n_slots,
            eta=config.mwu_eta,
            payoff_normalization=config.payoff_normalization,
        )
    else:
        raise ValueError(f"Unknown policy: {config.policy_type}")

    result = ExperimentResult(config, sim)
    result.seed = seed
    return result


def run_experiment(config: ExperimentConfig, verbose: bool = True,
                   compute_poa: bool = False, poa_method: str = 'brute_force') -> ExperimentResult:
    """Run experiment over n_runs seeds; return the worst-welfare (worst equilibrium) result."""

    if verbose:
        print(f"\n{'='*70}")
        print(f"Running Experiment: {config.name}")
        print(f"{'='*70}")
        print(f"Policy: {config.policy_type}")
        print(f"Regions: {config.n_regions}, Sources: {len(config.sources_config)}")
        print(f"Builders: {config.n_builders}, Slots: {config.n_slots}, Delta: {config.delta}, Runs: {config.n_runs}")

    regions, sources, latency_mean, latency_std = create_scenario_from_config(config)

    if verbose:
        print(f"\nSources: {[(s.name, f'lambda={s.lambda_rate}', f'starting region={s.region}') for s in sources]}")
        print(f"\nLatency mean matrix:")
        print(latency_mean, "\n")

    initial_belief = sum(
        s.lambda_rate * config.delta * np.exp(s.mu_val + s.sigma_val ** 2 / 2)
        for s in sources
    )

    seeds = get_seeds(config.n_runs)

    results = []
    for i, seed in enumerate(seeds):
        if verbose and config.n_runs > 1:
            print(f"Run {i+1}/{config.n_runs} (seed={seed})...")
        results.append(_run_single(config, seed, regions, sources, latency_mean, latency_std, initial_belief))

    # Pick the worst equilibrium
    result = min(results, key=lambda r: r.stats['mean_welfare'])

    if verbose:
        if config.n_runs > 1:
            print(f"\nWorst-welfare run: seed={result.seed} "
                  f"(welfare={result.stats['mean_welfare']:.4f})")
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
    if result.config.policy_type == "ABR":
        print(f"ABR adaptation steps: {stats['abr_adaptation_steps']}")
        print(f"ABR converged to pure NE: {stats['abr_converged']}")
        print(f"ABR max profitable deviation: {stats['abr_max_profitable_deviation']:.6f}")

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
