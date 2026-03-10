import numpy as np
from scipy.special import log_ndtr
from itertools import combinations_with_replacement

from sim.simulator import (Builder, LatencyPropagationModel, FixedPolicy, 
    LocationGamesSimulator, StochasticTransactionGenerator, EqualSplitSharingRule)
from sim.config import ExperimentConfig, create_scenario_from_config
from analysis.result import ExperimentResult

def _compute_welfare_analytical(
    profile: list,
    sources,
    propagation_model: LatencyPropagationModel,
    delta: float,
    n_time_steps: int = 200,
) -> float:
    """Compute E[W(s)] using Lemma from model doc"""
    n_regions = propagation_model._mu_ln.shape[0]
    counts = np.bincount(profile, minlength=n_regions).astype(float)  # (R,)

    # Time grid: exclude t=delta so remaining_time > 0
    t = np.linspace(0, delta, n_time_steps + 1)[:-1]
    remaining = delta - t  # (T,)
    log_remaining = np.log(remaining)  # (T,)

    welfare = 0.0
    for s_idx, source in enumerate(sources):
        mu_ln = propagation_model._mu_ln[:, s_idx] # (R,)
        sig_ln = propagation_model._sigma_ln[:, s_idx]  # (R,)
        ev = np.exp(source.mu_val + source.sigma_val ** 2 / 2)
        weight = source.lambda_rate * delta * ev

        # q[r, t] = P(d_{r,source} <= remaining[t])
        z = (log_remaining[None, :] - mu_ln[:, None]) / sig_ln[:, None]  # (R, T)
        log1mq = log_ndtr(-z)  # log(1 - phi(z)) = log(phi(-z))

        # log_no_coverage[t] = sum_r counts[r] * log(1 - q[r,t])
        log_no_cov = counts @ log1mq  # (T,)
        f_bar = float(np.mean(1.0 - np.exp(log_no_cov)))
        welfare += weight * f_bar

    return welfare


def compute_optimal_welfare_brute_force(
    config: ExperimentConfig,
    n_time_steps: int = 200,
) -> tuple:
    """Exact optimal welfare via exhaustive search over all builder multisets.
    Scales as C(R+K-1, K) evaluations. Feasible for small R or K"""
    _, sources, latency_mean, latency_std = create_scenario_from_config(config)
    prop_model = LatencyPropagationModel(latency_mean, latency_std)

    best_welfare, best_profile = -np.inf, None
    for profile in combinations_with_replacement(range(config.n_regions), config.n_builders):
        w = _compute_welfare_analytical(list(profile), sources, prop_model, config.delta, n_time_steps)
        if w > best_welfare:
            best_welfare, best_profile = w, list(profile)

    return best_welfare, best_profile


def _estimate_welfare_mc(
    profile: list,
    config: ExperimentConfig,
    n_rounds: int = 500,
) -> float:
    """Estimate E[W(s)] by running the simulator with builders frozen at profile."""
    regions, sources, latency_mean, latency_std = create_scenario_from_config(config)
    builders = [Builder(i, FixedPolicy(config.n_regions)) for i in range(config.n_builders)]
    sim = LocationGamesSimulator(
        regions=regions, sources=sources, builders=builders,
        tx_generator=StochasticTransactionGenerator(),
        propagation_model=LatencyPropagationModel(latency_mean, latency_std),
        sharing_rule=EqualSplitSharingRule(),
        delta=config.delta, seed=config.seed,
    )
    for i, builder in enumerate(sim.builders):
        builder.set_region(profile[i])
    sim.run(n_rounds)
    return sim.get_statistics()['mean_welfare']


def compute_optimal_welfare_greedy_mc(
    config: ExperimentConfig,
    n_rounds_per_eval: int = 500,
) -> tuple:
    """(1-1/e)-approximate optimal welfare via greedy + Monte Carlo welfare estimates.
    Runs K*R welfare evaluations and each costs n_rounds_per_eval simulation rounds."""
    profile = []
    for _ in range(config.n_builders):
        best_w, best_r = -np.inf, 0
        for r in range(config.n_regions):
            # Fill unassigned builders with region 0 (consistent across candidates so argmax unaffected)
            candidate = profile + [r] + [0] * (config.n_builders - len(profile) - 1)
            w = _estimate_welfare_mc(candidate, config, n_rounds_per_eval)
            if w > best_w:
                best_w, best_r = w, r
        profile.append(best_r)

    final_welfare = _estimate_welfare_mc(profile, config, n_rounds_per_eval * 5)
    return final_welfare, profile


def compute_poa_stats(
    result: ExperimentResult,
    method: str = 'brute_force',
    n_time_steps: int = 200,
    n_rounds_per_eval: int = 500,
) -> dict:
    """Compute PoA statistics for a completed experiment result.

    Args:
        method: 'brute_force' (exact, analytical) or 'greedy_mc' (approximate, simulation-based)
        n_time_steps: time discretisation for brute_force integral (higher = more accurate)
        n_rounds_per_eval: MC rounds per profile evaluation for greedy_mc
    """
    _, sources, _, _ = create_scenario_from_config(result.config)
    w_upper = sum(
        s.lambda_rate * result.config.delta * np.exp(s.mu_val + s.sigma_val ** 2 / 2)
        for s in sources
    )
    w_learned = result.stats['mean_welfare']

    if method == 'brute_force':
        w_star, opt_profile = compute_optimal_welfare_brute_force(result.config, n_time_steps)
    elif method == 'greedy_mc':
        w_star, opt_profile = compute_optimal_welfare_greedy_mc(result.config, n_rounds_per_eval)
    else:
        raise ValueError(f"Unknown PoA method: {method!r}. Use 'brute_force' or 'greedy_mc'.")

    return {
        'w_star': w_star,
        'w_upper': w_upper,
        'w_learned': w_learned,
        'opt_profile': opt_profile,
        'opt_profile_names': [result.config.region_names[r] for r in opt_profile],
        'poa': w_star / w_learned if w_learned > 0 else float('inf'),
        'poa_upper_bound': w_upper / w_learned if w_learned > 0 else float('inf'),
        'method': method,
    }
