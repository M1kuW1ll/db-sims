import numpy as np
from scipy.special import log_ndtr
from itertools import combinations_with_replacement

from sim.simulator import LatencyPropagationModel, FixedLatencyPropagationModel, PropagationModel
from sim.config import ExperimentConfig, create_scenario_from_config
from analysis.result import ExperimentResult

def _compute_welfare_analytical(
    profile: list,
    sources,
    propagation_model: PropagationModel,
    delta: float,
    n_time_steps: int = 200,
) -> float:
    """Compute E[W(s)] analytically. Uses the correct method based on propagation model type."""
    if isinstance(propagation_model, FixedLatencyPropagationModel):
        return _compute_welfare_fixed(profile, sources, propagation_model, delta)
    return _compute_welfare_lognormal(profile, sources, propagation_model, delta, n_time_steps)


def _compute_welfare_lognormal(
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
        active_mask = counts > 0
        if active_mask.any():
            log_no_cov = counts[active_mask] @ log1mq[active_mask]  # (T,)
            f_bar = float(np.mean(1.0 - np.exp(log_no_cov)))
        else:
            f_bar = 0.0
        welfare += weight * f_bar

    return welfare


def _compute_welfare_fixed(
    profile: list,
    sources,
    propagation_model: FixedLatencyPropagationModel,
    delta: float,
) -> float:
    """Analytical welfare for deterministic fixed-latency propagation
    """
    active_regions = set(profile)

    welfare = 0.0
    for s_idx, source in enumerate(sources):
        ev = np.exp(source.mu_val + source.sigma_val ** 2 / 2)
        # Coverage fraction for each region in profile (others contribute 0)
        # # P(covered by at least one region) = 1 - prod(1 - coverage_r) but coverage_r is 0 or 1 here
        # so it simplifies: covered iff any region r in profile has latency <= delta - t
        min_latency = min(
            propagation_model.latency_mean[r, s_idx] for r in active_regions
        )
        coverage_fraction = max(0.0, delta - min_latency) / delta
        welfare += source.lambda_rate * delta * ev * coverage_fraction

    return welfare


def _make_prop_model(config: ExperimentConfig, latency_mean, latency_std) -> PropagationModel:
    if config.propagation_model_type == "fixed":
        return FixedLatencyPropagationModel(latency_mean)
    return LatencyPropagationModel(latency_mean, latency_std)

def optimal_welfare_brute_force(
    K: int,
    n_regions: int,
    sources,
    propagation_model: PropagationModel,
    delta: float,
    n_time_steps: int = 200,
    verbose: bool = False,
) -> tuple:
    """Exact optimal welfare via exhaustive search over multisets of size K
    over n_regions. Scales as C(R+K-1, K) evaluations."""
    best_welfare, best_profile = -np.inf, None
    best_profiles = []
    for profile in combinations_with_replacement(range(n_regions), K):
        w = _compute_welfare_analytical(list(profile), sources, propagation_model, delta, n_time_steps)
        if w > best_welfare:
            best_welfare, best_profile = w, list(profile)
            best_profiles = []
        if w == best_welfare:
            best_profiles.append(list(profile))

    if verbose:
        print(f"Optimal welfare: {best_welfare:.4f} achieved by {len(best_profiles)} profiles")

    return best_welfare, best_profile


def optimal_welfare_greedy(
    K: int,
    n_regions: int,
    sources,
    propagation_model: PropagationModel,
    delta: float,
    n_time_steps: int = 200,
) -> tuple:
    """(1 - 1/e)-approximate optimal welfare via greedy.
    Runs K * n_regions evaluations."""
    profile = []
    for _ in range(K):
        best_w, best_r = -np.inf, 0
        for r in range(n_regions):
            candidate = profile + [r]
            w = _compute_welfare_analytical(candidate, sources, propagation_model, delta, n_time_steps)
            if w > best_w:
                best_w, best_r = w, r
        profile.append(best_r)

    final_welfare = _compute_welfare_analytical(profile, sources, propagation_model, delta, n_time_steps)
    return final_welfare, profile


def compute_optimal_welfare_brute_force(
    config: ExperimentConfig,
    n_time_steps: int = 200,
) -> tuple:
    _, sources, latency_mean, latency_std = create_scenario_from_config(config)
    prop_model = _make_prop_model(config, latency_mean, latency_std)
    return optimal_welfare_brute_force(
        config.n_builders, config.n_regions, sources, prop_model,
        config.delta, n_time_steps, verbose=True,
    )


def compute_optimal_welfare_greedy(
    config: ExperimentConfig,
    n_time_steps: int = 200,
) -> tuple:
    _, sources, latency_mean, latency_std = create_scenario_from_config(config)
    prop_model = _make_prop_model(config, latency_mean, latency_std)
    return optimal_welfare_greedy(
        config.n_builders, config.n_regions, sources, prop_model,
        config.delta, n_time_steps,
    )

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
    _, sources, latency_mean, latency_std = create_scenario_from_config(result.config)
    prop_model = _make_prop_model(result.config, latency_mean, latency_std)

    w_upper = sum(
        s.lambda_rate * result.config.delta * np.exp(s.mu_val + s.sigma_val ** 2 / 2)
        for s in sources
    )

    final_counts = result.region_counts[-1].astype(int)
    converged_profile = [region for region, count in enumerate(final_counts) for _ in range(count)]
    w_converged = _compute_welfare_analytical(converged_profile, sources, prop_model, result.config.delta, n_time_steps)

    if method == 'brute_force':
        w_star, opt_profile = compute_optimal_welfare_brute_force(result.config, n_time_steps)
    elif method == 'greedy':
        w_star, opt_profile = compute_optimal_welfare_greedy(result.config, n_time_steps)
    else:
        raise ValueError(f"Unknown PoA method: {method!r}. Use 'brute_force' or 'greedy'.")

    return {
        'w_star': w_star,
        'w_upper': w_upper,
        'w_converged': w_converged,
        'opt_profile': opt_profile,
        'opt_profile_names': [result.config.region_names[r] for r in opt_profile],
        'poa': w_star / w_converged if w_converged > 0 else float('inf'),
        'poa_upper_bound': w_upper / w_converged if w_converged > 0 else float('inf'),
        'method': method,
    }
