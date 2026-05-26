#!/usr/bin/env python3
import numpy as np
from typing import List

from sim.config import ExperimentConfig, create_scenario_from_config, get_seeds
from analysis.result import ExperimentResult
from analysis.poa import compute_poa_stats
from sim.simulator import (
    Region, Source, Builder, LocationGamesSimulator,
    FixedPolicy, EXP3Policy, StochasticTransactionGenerator,
    LatencyPropagationModel, FixedLatencyPropagationModel, EqualSplitSharingRule,
)

def _run_single(config: ExperimentConfig, seed: int,
                regions, sources, latency_mean, latency_std,
                initial_belief) -> ExperimentResult:
    """Run one simulation instance with the given seed and return its result."""
    builders = []
    for i in range(config.n_builders):
        if config.policy_type == "EXP3":
            policy = EXP3Policy(
                config.n_regions,
                eta=config.eta,
                gamma=config.gamma,
                initial_belief=initial_belief,
                payoff_normalization=config.payoff_normalization,
                gamma_schedule=config.gamma_schedule,
                gamma_min=config.gamma_min,
                gamma_decay=config.gamma_decay,
                total_slots=config.n_slots,
                norm_alpha=config.norm_alpha,
            )
        elif config.policy_type == "ABR":
            policy = FixedPolicy(config.n_regions, initial_belief=initial_belief)
        else:
            raise ValueError(f"Unknown policy: {config.policy_type}")
        builders.append(Builder(i, policy))

    sim = LocationGamesSimulator(
        regions=regions,
        sources=sources,
        builders=builders,
        tx_generator=StochasticTransactionGenerator(),
        propagation_model=(
            FixedLatencyPropagationModel(latency_mean)
            if config.propagation_model_type == "fixed"
            else LatencyPropagationModel(latency_mean, latency_std)
        ),
        sharing_rule=EqualSplitSharingRule(),
        delta=config.delta,
        seed=seed,
        placement_seed=config.placement_seed,
        initial_placement=config.initial_placement,
    )
    if config.policy_type == "EXP3":
        sim.run(config.n_slots)
    elif config.policy_type == "ABR":
        if config.abr_update_mode == "simultaneous":
            sim.run_simultaneous_better_response(
                config.n_slots,
                improvement_threshold_pct=config.improvement_threshold_pct,
                n_time_steps=config.utility_eval_time_steps,
                max_rounds=config.abr_max_rounds or config.abr_max_updates,
                response_rule=config.abr_response_rule,
            )
        else:
            sim.run_async_better_response(
                config.n_slots,
                improvement_threshold_pct=config.improvement_threshold_pct,
                n_time_steps=config.utility_eval_time_steps,
                max_updates=config.abr_max_updates,
                response_rule=config.abr_response_rule,
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
            print(f"W* (optimal): {p['w_star']:.6f}")
            print(f"W (converged): {p['w_converged']:.6f}")
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
    print(f"Empirical epsilon-CCE gap: {stats['cce_gap']:.6f}")
    if result.config.policy_type == "ABR":
        print(f"ABR update mode: {result.config.abr_update_mode}")
        print(f"ABR response rule: {result.config.abr_response_rule}")
        print(f"ABR adaptation steps: {stats['abr_adaptation_steps']}")
        print(f"ABR converged to pure NE: {stats['abr_converged']}")
        print(f"ABR cycle detected: {stats['abr_cycle_detected']}")
        print(f"ABR max profitable deviation: {stats['abr_max_profitable_deviation']:.6f}")

    print(f"\nRegion selection per slot (avg builders per slot):")
    for i, count in enumerate(stats['avg_region_counts']):
        print(f"  {regions[i].name}: {count:.2f}")

    print(f"\nConcentration metrics (converged profile):")
    print(f"  Location Gini: {stats['location_gini']:.4f}")
    print(f"  Location Entropy: {stats['location_entropy']:.4f}")
    print(f"  Location HHI: {stats['location_hhi']:.4f}")
    print(f"  Utility Gini: {stats['utility_gini']:.4f}")
    print(f"  Utility Entropy: {stats['utility_entropy']:.4f}")
    print(f"  Utility HHI: {stats['utility_hhi']:.4f}")
