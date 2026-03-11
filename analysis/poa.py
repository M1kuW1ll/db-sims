import numpy as np
from scipy.special import log_ndtr
from itertools import combinations_with_replacement

from sim.simulator import LatencyPropagationModel
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


def compute_optimal_welfare_greedy(
    config: ExperimentConfig,
    n_time_steps: int = 200,
) -> tuple:
    """(1-1/e)-approximate optimal welfare via greedy + analytical welfare.
    Runs K*R evaluations."""
    _, sources, latency_mean, latency_std = create_scenario_from_config(config)
    prop_model = LatencyPropagationModel(latency_mean, latency_std)

    profile = []
    for _ in range(config.n_builders):
        best_w, best_r = -np.inf, 0
        for r in range(config.n_regions):
            candidate = profile + [r] + [0] * (config.n_builders - len(profile) - 1)
            w = _compute_welfare_analytical(candidate, sources, prop_model, config.delta, n_time_steps)
            if w > best_w:
                best_w, best_r = w, r
        profile.append(best_r)

    final_welfare = _compute_welfare_analytical(profile, sources, prop_model, config.delta, n_time_steps)
    return final_welfare, profile


def compute_poa_stats(
    result: ExperimentResult,
    method: str = 'brute_force',
    n_time_steps: int = 200,
) -> dict:
    """Compute PoA statistics for a completed experiment result.

    Args:
        method: 'brute_force' (exact, analytical) or 'greedy'
        n_time_steps: time discretisation for the analytical integral
    """
    _, sources, _, _ = create_scenario_from_config(result.config)
    w_upper = sum(
        s.lambda_rate * result.config.delta * np.exp(s.mu_val + s.sigma_val ** 2 / 2)
        for s in sources
    )
    w_learned = result.stats['mean_welfare']

    if method == 'brute_force':
        w_star, opt_profile = compute_optimal_welfare_brute_force(result.config, n_time_steps)
    elif method == 'greedy':
        w_star, opt_profile = compute_optimal_welfare_greedy(result.config, n_time_steps)
    else:
        raise ValueError(f"Unknown PoA method: {method!r}. Use 'brute_force' or 'greedy'.")

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
