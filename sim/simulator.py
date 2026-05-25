#!/usr/bin/env python3
"""
Decentralized Building Simulator (db-sims) - Core simulation engine

Inspired by GeoDec research, generalized for studying location choice in various
distributed and decentralized block building regimes.
"""
from collections import defaultdict

import numpy as np
from scipy.stats import norm
from typing import List, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod
from sim.metrics import gini, entropy, hhi

@dataclass
class Transaction:
    source_id: int
    emission_time: float  # t_j ~ Uniform(0, Delta)
    value: float # V_j ~ Lognormal(mu_I, sigma_I)

@dataclass
class Region:
    """A geographical region."""
    id: int
    name: str


@dataclass
class Source:
    """A signal source with constant value."""
    id: int
    name: str
    region: int  # The region to which the source belongs
    lambda_rate: float  # transactions per second
    mu_val: float  # lognormal mean param for transaction values
    sigma_val: float  # lognormal std param for transaction values


class LearningPolicy(ABC):
    """Abstract base class for learning policies."""
    beliefs: np.ndarray  # Policy's internal beliefs about region values

    def __init__(self, n_regions: int, initial_belief: float = 0.0):
        self.beliefs = np.ones(n_regions) * initial_belief

    @abstractmethod
    def choose(self, current_region: int) -> int:
        """Choose region_id."""
        pass

    @abstractmethod
    def update(self, region_id: int, reward: float):
        """Update policy state after observing reward."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return policy name for logging."""
        pass


class FixedPolicy(LearningPolicy):
    """Policy that never moves. We use this for welfare estimation at fixed profiles."""
    def choose(self, current_region: int) -> int:
        return current_region

    def update(self, region_id: int, reward: float):
        pass

    def get_name(self) -> str:
        return "Fixed"


class EXP3Policy(LearningPolicy):
    """
    EXP3 bandit: exponential weights with importance-weighted updates.
    Builders observe only their own realised reward (no counterfactuals).
    """
    def __init__(self, n_regions: int, eta: float = 0.02, gamma: float = 0.05,
                 initial_belief: float = 1.0,
                 gamma_schedule: str = "static",
                 gamma_min: float = 0.01,
                 gamma_decay: float = 0.0002,
                 total_slots: int = 20000,
                 norm_alpha: float = 0.0):
        super().__init__(n_regions, initial_belief)
        self.eta = eta
        self.gamma = gamma
        self._norm = max(initial_belief, 1e-10)
        self.weights = np.ones(n_regions)
        self._last_p = np.ones(n_regions) / n_regions  # p at most recent choose()
        self.gamma_schedule = gamma_schedule
        self.gamma_min = gamma_min
        self.gamma_decay = gamma_decay
        self.total_slots = total_slots
        self.norm_alpha = norm_alpha
        self._step = 0

    def _current_gamma(self) -> float:
        t = self._step
        g0 = self.gamma
        if self.gamma_schedule == "static":
            return g0
        elif self.gamma_schedule == "exponential":
            return self.gamma_min + (g0 - self.gamma_min) * np.exp(-self.gamma_decay * t)
        elif self.gamma_schedule == "sqrt_decay":
            return max(self.gamma_min, g0 / np.sqrt(t + 1))
        elif self.gamma_schedule == "linear":
            frac = 1.0 - t / max(self.total_slots, 1)
            return max(self.gamma_min, g0 * frac)
        else:
            raise ValueError(f"Unknown gamma_schedule: {self.gamma_schedule!r}")

    def choose(self, current_region: int) -> int:
        gamma_t = self._current_gamma()
        p = (1 - gamma_t) * self.weights / self.weights.sum() + gamma_t / len(self.weights)
        self._last_p = p
        return int(np.random.choice(len(p), p=p))

    def update(self, region_id: int, reward: float):
        if self.norm_alpha > 0.0:
            self._norm = (1 - self.norm_alpha) * self._norm + self.norm_alpha * max(reward, 1e-10)
        gain_hat = (reward / self._norm) / self._last_p[region_id]
        # Update in log-space to avoid overflow before max-normalisation
        log_w = np.log(np.maximum(self.weights, 1e-300))
        log_w[region_id] += self.eta * gain_hat
        log_w -= log_w.max()
        self.weights = np.exp(log_w)
        self.beliefs[region_id] = reward  # scalar tracking for chosen arm only
        self._step += 1

    def get_name(self) -> str:
        if self.gamma_schedule != "static":
            return f"EXP3({self.gamma_schedule})"
        return "EXP3"


class PropagationModel(ABC):
    @abstractmethod
    def receives(self, region_id: int, source_id: int, tx: Transaction, delta: float) -> bool:
        """Determine if a builder in region_id receives transaction tx from source_id within delta."""
        pass

    @abstractmethod
    def reception_prob(self, region_id: int, source_id: int, remaining_time: float) -> float:
        """Return P(latency <= remaining_time) for a builder in region_id from source_id."""
        pass

class LatencyPropagationModel(PropagationModel):
    """
    Accepts raw empirical latency_mean and latency_std (in seconds) and converts
    to lognormal parameters
    latency_mean: np.ndarray of shape (n_regions, n_sources)
    latency_std: np.ndarray of shape (n_regions, n_sources)
    """
    def __init__(self, latency_mean: np.ndarray, latency_std: np.ndarray):
        sigma_ln = np.sqrt(np.log(1 + (latency_std / latency_mean) ** 2))
        self._mu_ln = np.log(latency_mean) - sigma_ln ** 2/2
        self._sigma_ln = sigma_ln

    def receives(self, region_id: int, source_id: int, tx: Transaction, delta: float) -> bool:
        d = np.random.lognormal(self._mu_ln[region_id, source_id], self._sigma_ln[region_id, source_id])
        return tx.emission_time + d <= delta

    def reception_prob(self, region_id: int, source_id: int, remaining_time: float) -> float:
        if remaining_time <= 0.0:
            return 0.0
        mu = self._mu_ln[region_id, source_id]
        sigma = self._sigma_ln[region_id, source_id]
        if sigma < 1e-10:
            return 1.0 if remaining_time >= np.exp(mu) else 0.0
        return float(norm.cdf((np.log(remaining_time) - mu) / sigma))


class FixedLatencyPropagationModel(PropagationModel):
    """
    Deterministic propagation: a transaction is received iff emission_time + latency <= delta.
    Use for synthetic experiments to eliminate stochastic propagation noise.
    """
    def __init__(self, latency_mean: np.ndarray):
        self.latency_mean = latency_mean

    def receives(self, region_id: int, source_id: int, tx: Transaction, delta: float) -> bool:
        return tx.emission_time + self.latency_mean[region_id, source_id] <= delta

    def reception_prob(self, region_id: int, source_id: int, remaining_time: float) -> float:
        return 1.0 if self.latency_mean[region_id, source_id] <= remaining_time else 0.0


@dataclass
class Builder:
    """A builder/agent with learning state."""
    id: int
    policy: LearningPolicy
    current_region: int = 0

    def choose_region(self) -> int:
        self.current_region = self.policy.choose(self.current_region)
        return self.current_region

    def update(self, region_id: int, reward: float):
        self.policy.update(region_id, reward)

    def set_region(self, region_id: int):
        """Set the builder's current region."""
        self.current_region = region_id


class SharingRule(ABC):
    @abstractmethod
    def compute_rewards(self, tx_values: Dict[int, float], tx_receivers: Dict[int, List[int]]) -> Dict[int, float]:
        pass

class EqualSplitSharingRule(SharingRule):
    """V_j / k_j split among all receivers of transaction j."""
    def compute_rewards(self, tx_values: Dict[int, float], tx_receivers: Dict[int, List[int]]) -> Dict[int, float]:
        rewards: Dict[int, float] = defaultdict(float)
        for tx_id, receivers in tx_receivers.items():
            split = tx_values[tx_id] / len(receivers)
            for builder_id in receivers:
                rewards[builder_id] += split
        return rewards

class TransactionGenerator(ABC):
    @abstractmethod
    def generate(self, source: Source, delta: float) -> List[Transaction]:
        """Generate transactions for a given source and time interval."""
        pass

class StochasticTransactionGenerator(TransactionGenerator):
    """Poisson count, lognormal value, uniform emission time."""
    def generate(self, source: Source, delta: float) -> List[Transaction]:
        n = np.random.poisson(source.lambda_rate * delta)
        emission_times = np.random.uniform(0, delta, size=n)
        values = np.random.lognormal(source.mu_val, source.sigma_val, size=n)

        return [Transaction(source_id=source.id, emission_time=emission_times[i], value=values[i])
                for i in range(n)]


def precompute_sharing_weights(
    other_builder_regions: List[int],
    sources: List[Source],
    propagation_model: PropagationModel,
    delta: float,
    n_t: int = 200,
) -> np.ndarray:
    """
    Precompute E[1 / (1+X_{I,t})] for each (source, time point).
    X_{I,t} is the number of static builders that receive a tx from source I emitted at time t.
    """
    t_points = np.linspace(0, delta, n_t, endpoint=False)
    remaining = delta - t_points
    n_other = len(other_builder_regions)
    weights = np.zeros((len(sources), n_t))

    for i, source in enumerate(sources):
        # Reception probabilities: probs[b, n] = P(static builder b receives tx at time t_n)
        probs = np.zeros((n_other, n_t))
        for builder, region in enumerate(other_builder_regions):
            for n, rem in enumerate(remaining):
                probs[builder, n] = propagation_model.reception_prob(region, source.id, rem)

        for n in range(n_t):
            # Poisson Binomial dynamic programming approach
            # we build PMF of X = count of static builders receiving the tx
            pmf = np.array([1.0])
            for builder in range(n_other):
                prob = probs[builder, n]
                new_pmf = np.zeros(len(pmf)+1)
                new_pmf[:-1] += pmf * (1.0 - prob)
                new_pmf[1:] += pmf * prob
                pmf = new_pmf
            total_builders = np.array(range(1, len(pmf) + 1))
            weights[i, n] = float(np.sum(pmf / total_builders))

    return weights


def compute_expected_reward(
    candidate_region: int,
    sharing_weights: np.ndarray,
    sources: List[Source],
    propagation_model: PropagationModel,
    delta: float,
    n_t: int = 100,
) -> float:
    """
    Compute analytical expected reward for a builder at candidate_region,
    given precomputed sharing weights from static builders.
    """
    remaining = delta - np.linspace(0, delta, n_t, endpoint=False)
    total = 0.0
    for i, source in enumerate(sources):
        ev = np.exp(source.mu_val + 0.5 * source.sigma_val ** 2)
        q = np.array([
            propagation_model.reception_prob(candidate_region, source.id, rem)
            for rem in remaining
        ])
        # numerical integration: delta * mean approximates integral_{0->delta} q(t) * w(t) dt
        # => (1/n_t) * sum_n q(t_n) * w(t_n)
        integral = float(np.mean(q * sharing_weights[i]))
        total += source.lambda_rate * ev * delta * integral
    return total


def compute_all_builder_utilities(
    profile: List[int],
    sources: List[Source],
    propagation_model: PropagationModel,
    delta: float,
    n_t: int = 100,
) -> np.ndarray:
    """Compute analytical expected utility u_b(s) for every builder in the profile.
    """
    K = len(profile)
    utilities = np.zeros(K)
    for builder in range(K):
        other_regions = [profile[i] for i in range(K) if i != builder]
        weights = precompute_sharing_weights(
            other_regions, sources, propagation_model, delta, n_t
        )
        utilities[builder] = compute_expected_reward(
            profile[builder], weights, sources, propagation_model, delta, n_t
        )
    return utilities


class LocationGamesSimulator:
    """
    Core simulator for studying location choice in decentralized block building.

    Implements reward sharing among builders based on their chosen regions and the transactions they capture
    which are generated stochastically from information sources. Builders learn and adapt their region choices
    over time based on observed rewards.
    """

    def __init__(self,
                 regions: List[Region],
                 sources: List[Source],
                 builders: List[Builder],
                 tx_generator: TransactionGenerator,
                 propagation_model: PropagationModel,
                 sharing_rule: SharingRule,
                 delta: float,
                 seed: int = 42,
                 placement_seed: int = 0,
                 initial_placement: str = "dispersed"):
        """
        Args:
            regions: List of regions
            sources: List of sources
            builders: List of builders (agents)
            tx_generator: Transaction generator
            propagation_model: Propagation model
            sharing_rule: Sharing rule
            delta: Delta parameter
            seed: Random seed for dynamics (ABR shuffle, tx draws), changes across runs
            placement_seed: Random seed for initial builder placement, fixed across runs
            initial_placement: "dispersed", "random", or "concentrated"
        """
        self.regions = regions
        self.sources = sources
        self.builders = builders
        self.tx_generator = tx_generator
        self.propagation_model = propagation_model
        self.sharing_rule = sharing_rule
        self.delta = delta

        self.n_regions = len(regions)
        self.n_sources = len(sources)
        self.n_builders = len(builders)
        self.initial_placement = initial_placement
        self._placement_rng = np.random.default_rng(placement_seed)

        np.random.seed(seed)

        # Initialize builders across regions according to placement strategy
        self._initialize_builder_distribution()

        # Tracking
        self.region_counts_history: List[np.ndarray] = []
        self.reward_history: List[List[float]] = []
        self.welfare_history: List[float] = []
        self.builder_distribution_history: List[np.ndarray] = []
        self.region_reward_pairs_history: List[List[tuple]] = []
        self.tx_emitted_history: List[int] = []
        self.tx_received_history: List[int] = []


    def _initialize_builder_distribution(self):
        """Initialize builder locations according to self.initial_placement."""
        for i, builder in enumerate(self.builders):
            if self.initial_placement == "dispersed":
                # Evenly space builders across regions: builder i goes to region i * n_regions // n_builders
                # eg 5 builders / 10 regions -> [0, 2, 4, 6, 8]
                # TODO: Once we start using GCP data we should incorporate latitude/longitude of regions
                region = i * self.n_regions // self.n_builders
            elif self.initial_placement == "concentrated":
                region = 0
            elif self.initial_placement == "random":
                region = int(self._placement_rng.integers(0, self.n_regions))
            else:
                raise ValueError(f"Unknown initial_placement: {self.initial_placement!r}. "
                                 f"Use 'dispersed', 'concentrated', or 'random'.")
            builder.set_region(region)

    def _get_builder_distribution(self) -> np.ndarray:
        """Get current distribution of all builders across regions."""
        distribution = np.zeros(self.n_regions)
        for builder in self.builders:
            distribution[builder.current_region] += 1
        return distribution

    def run_round(self):
        # All builders choose a region
        builder_selected_regions = {builder.id: builder.choose_region() for builder in self.builders}

        # For each source, determine which builders receive its transactions and compute rewards
        tx_receivers: Dict[int, List[int]] = defaultdict(list)  # tx_id -> list of builder_ids
        tx_values: Dict[int, float] = {}  # tx_id -> value
        tx_emitted_counter = 0
        tx_received_counter = 0

        for source in self.sources:
            txs = self.tx_generator.generate(source, self.delta)

            for tx in txs:
                receivers = [builder.id for builder in self.builders if 
                            self.propagation_model.receives(
                            builder_selected_regions[builder.id],
                            source.id, tx, self.delta)]
                if receivers:
                    tx_receivers[tx_received_counter] = receivers
                    tx_values[tx_received_counter] = tx.value
                    tx_received_counter += 1

                tx_emitted_counter += 1
        
        rewards = self.sharing_rule.compute_rewards(tx_values=tx_values, tx_receivers=tx_receivers)

        # Update builders with their rewards and new regions
        slot_rewards = []
        for builder in self.builders:
            reward = rewards.get(builder.id, 0.0)
            builder.update(builder_selected_regions[builder.id], reward)
            slot_rewards.append(reward)

        # Track history
        region_counts = np.zeros(self.n_regions)
        for region_id in builder_selected_regions.values():
            region_counts[region_id] += 1

        self.region_counts_history.append(region_counts)
        self.reward_history.append(slot_rewards)
        self.welfare_history.append(float(sum(slot_rewards)))
        self.builder_distribution_history.append(self._get_builder_distribution())
        self.region_reward_pairs_history.append([
            (builder_selected_regions[builder.id], rewards.get(builder.id, 0.0)) for builder in self.builders
        ])
        self.tx_emitted_history.append(tx_emitted_counter)
        self.tx_received_history.append(tx_received_counter)


    def run(self, n_slots: int):
        """Run simulation for n_slots."""
        for _ in range(n_slots):
            self.run_round()

    def _step_abr(self, round_index: int, n_t: int) -> bool:
        """Analytical migration for one builder (round robin). Returns True if it moved."""
        active = self.builders[round_index % self.n_builders]
        old_region = active.current_region
        other_builder_regions = [builder.current_region for builder in self.builders if builder.id != active.id]

        sharing_weights = precompute_sharing_weights(
            other_builder_regions, self.sources, self.propagation_model, self.delta, n_t
        )
        u_current = compute_expected_reward(
            active.current_region, sharing_weights, self.sources,
            self.propagation_model, self.delta, n_t
        )

        candidates = [region for region in range(self.n_regions) if region != active.current_region]
        np.random.shuffle(candidates)
        for region in candidates:
            u_r = compute_expected_reward(
                region, sharing_weights, self.sources, self.propagation_model, self.delta, n_t
            )
            if u_r > u_current:
                # Builder migrates to the first strictly better region found (no cost for now)
                active.set_region(region)
                break

        return active.current_region != old_region

    def run_round_abr(self, round_index: int, n_t: int) -> bool:
        """
        One round of asynchronous better response dynamics.
        A single builder is selected by round robin. It evaluates all regions
        analytically and migrates to the first one (in random order) that strictly
        improves its expected reward. All builders then compete from their current
        locations and rewards are recorded. Returns True if the builder migrated.
        """
        migrated = self._step_abr(round_index, n_t)
        self.run_round()
        return migrated

    def run_abr_until_convergence(self, n_t: int = 100, max_rounds: int = 5000,
                                   convergence_sweeps: int = None) -> int:
        """Run ABR until no builder migrates for `convergence_sweeps` consecutive full sweeps.
        Does not record stochastic transaction history and only updates builder positions.
        Returns total migration steps taken."""
        K = self.n_builders
        convergence_sweeps = convergence_sweeps or K
        no_move_count = 0
        total_rounds = 0
        sweep = 0
        while total_rounds < max_rounds:
            any_move = False
            for k in range(K):
                if self._step_abr(sweep * K + k, n_t):
                    any_move = True
                total_rounds += 1
            sweep += 1
            if any_move:
                no_move_count = 0
            else:
                no_move_count += 1
                if no_move_count >= convergence_sweeps:
                    break
        return total_rounds

    def run_abr(self, n_slots: int, n_t: int = 100):
        """Run asynchronous better response dynamics for n_slots rounds."""
        for i in range(n_slots):
            self.run_round_abr(i, n_t)

    def get_statistics(self) -> Dict:
        """Compute summary statistics."""
        region_counts = np.array(self.region_counts_history)
        builder_distribution = np.array(self.builder_distribution_history)

        avg_region_counts = np.mean(region_counts, axis=0) if len(region_counts) > 0 else np.zeros(self.n_regions)

        all_rewards = [r for slot_rewards in self.reward_history for r in slot_rewards]
        avg_reward = float(np.mean(all_rewards)) if all_rewards else 0.0

        welfare = np.array(self.welfare_history)
        mean_txs_emitted = float(np.mean(self.tx_emitted_history)) if self.tx_emitted_history else 0.0
        mean_txs_received = float(np.mean(self.tx_received_history)) if self.tx_received_history else 0.0
        coverage_per_round = [r / e if e > 0 else 0.0
                              for e, r in zip(self.tx_emitted_history, self.tx_received_history)]
        mean_coverage_ratio = float(np.mean(coverage_per_round)) if coverage_per_round else 0.0
        mean_txs_per_builder = mean_txs_received / self.n_builders if self.n_builders > 0 else 0.0
        all_slot_rewards = [sum(s) / len(s) for s in self.reward_history if s]
        mean_value_per_builder = float(np.mean(all_slot_rewards)) if all_slot_rewards else 0.0

        # Location metrics: use last slot's deterministic builder distribution
        last_builder_dist = (
            builder_distribution[-1] if len(builder_distribution) > 0
            else np.zeros(self.n_regions)
        )

        # Utility metrics: analytical expected utility u_b(s*) at the converged profile
        final_profile = [b.current_region for b in self.builders]
        utilities = compute_all_builder_utilities(
            final_profile, self.sources, self.propagation_model, self.delta
        )

        return {
            'avg_region_counts': avg_region_counts,
            'avg_builder_distribution': np.mean(builder_distribution, axis=0) if len(builder_distribution) > 0 else np.zeros(self.n_regions),
            'avg_reward': avg_reward,
            'location_gini': gini(last_builder_dist),
            'location_entropy': entropy(last_builder_dist),
            'location_hhi': hhi(last_builder_dist),
            'utility_gini': gini(utilities),
            'utility_entropy': entropy(utilities),
            'utility_hhi': hhi(utilities),
            'total_slots': len(self.region_counts_history),
            'mean_welfare': float(np.mean(welfare)) if len(welfare) > 0 else 0.0,
            'mean_txs_emitted_per_round': mean_txs_emitted,
            'mean_txs_received_per_round': mean_txs_received,
            'mean_coverage_ratio': mean_coverage_ratio,
            'mean_txs_received_per_builder': mean_txs_per_builder,
            'mean_value_per_builder': mean_value_per_builder,
        }
