#!/usr/bin/env python3
"""
Decentralized Building Simulator (db-sims) - Core simulation engine.

Supported dynamics:
  (A) EMA + softmax
  (B) Individual UCB bandit
  (C) Individual EXP3 bandit
  (D) Asynchronous exact-response dynamics (best or better response)
  (E) Full-information multiplicative weights update (MWU)
"""
from collections import defaultdict
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from sim.metrics import gini, entropy, hhi

from scipy.special import ndtr

ABR_RESPONSE_RULES = {"best", "better"}
ABR_UPDATE_MODES = {"async", "simultaneous"}


@dataclass
class Transaction:
    source_id: int
    emission_time: float
    value: float


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
    region: int
    lambda_rate: float
    mu_val: float
    sigma_val: float


class LearningPolicy(ABC):
    """Abstract base class for learning policies."""
    beliefs: np.ndarray

    def __init__(self, n_regions: int, initial_belief: float = 0.0):
        self.beliefs = np.ones(n_regions) * initial_belief

    @abstractmethod
    def choose(self, current_region: int) -> int:
        pass

    @abstractmethod
    def update(self, region_id: int, reward: float):
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


class EMASoftmaxPolicy(LearningPolicy):
    """Policy A: EMA + softmax."""

    def __init__(
        self,
        n_regions: int,
        eta: float = 0.1,
        beta: float = 2.0,
        cost: float = 0.0,
        initial_belief: float = 0.0,
    ):
        super().__init__(n_regions, initial_belief)
        self.eta = eta
        self.beta = beta
        self.cost = cost

    def choose(self, current_region: int) -> int:
        shifted = self.beta * (self.beliefs - np.max(self.beliefs))
        exp_scores = np.exp(shifted)
        probs_reg = exp_scores / np.sum(exp_scores)
        region_id = int(np.random.choice(len(self.beliefs), p=probs_reg))

        if self.beliefs[region_id] - self.beliefs[current_region] <= self.cost:
            return current_region

        return region_id

    def update(self, region_id: int, reward: float):
        self.beliefs[region_id] = (1 - self.eta) * self.beliefs[region_id] + self.eta * reward

    def get_name(self) -> str:
        return "EMA-Softmax"


class UCBPolicy(LearningPolicy):
    """Policy B: individual UCB bandit."""

    def __init__(
        self,
        n_regions: int,
        alpha: float = 1.0,
        cost: float = 0.0,
        initial_belief: float = 0.0,
    ):
        super().__init__(n_regions, initial_belief)
        self.alpha = alpha
        self.cost = cost
        self.N = np.zeros(len(self.beliefs))
        self.t = 0

    def choose(self, current_region: int) -> int:
        exploration_bonus = self.alpha * np.sqrt(np.log(1 + self.t) / (1 + self.N))
        ucb_scores = self.beliefs + exploration_bonus

        region_id = int(np.argmax(ucb_scores))
        if ucb_scores[region_id] - ucb_scores[current_region] <= self.cost:
            return current_region
        return region_id

    def update(self, region_id: int, reward: float):
        self.N[region_id] += 1
        self.beliefs[region_id] += (reward - self.beliefs[region_id]) / self.N[region_id]
        self.t += 1

    def get_name(self) -> str:
        return "UCB"


class FixedPolicy(LearningPolicy):
    """Policy that never moves."""

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
    def __init__(self, n_regions: int, eta: Optional[float] = None, gamma: float = 0.05,
                 initial_belief: float = 1.0,
                 payoff_normalization: Optional[float] = None,
                 gamma_schedule: str = "static",
                 gamma_min: float = 0.01,
                 gamma_decay: float = 0.0002,
                 total_slots: int = 20000,
                 norm_alpha: float = 0.0):
        super().__init__(n_regions, initial_belief)
        self.gamma = float(np.clip(gamma, 1e-9, 1.0))
        self.payoff_normalization = (
            max(float(payoff_normalization), 1.0)
            if payoff_normalization is not None and payoff_normalization > 0
            else None
        )
        if eta is None:
            eta = self.gamma / n_regions if self.payoff_normalization is not None else 0.02
        self.eta = eta
        self._norm = max(
            self.payoff_normalization if self.payoff_normalization is not None else initial_belief,
            1e-10,
        )
        self.weights = np.ones(n_regions, dtype=float)
        self._last_p = np.ones(n_regions, dtype=float) / n_regions  # p at most recent choose()
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
        del current_region  # EXP3 samples directly from its mixed strategy.
        gamma_t = self._current_gamma()
        p = (1 - gamma_t) * self.weights / self.weights.sum() + gamma_t / len(self.weights)
        self._last_p = p
        return int(np.random.choice(len(p), p=p))

    def update(self, region_id: int, reward: float):
        if self.norm_alpha > 0.0:
            self._norm = (1 - self.norm_alpha) * self._norm + self.norm_alpha * max(reward, 1e-10)
        normalized_reward = reward / self._norm
        if self.payoff_normalization is not None:
            normalized_reward = float(np.clip(normalized_reward, 0.0, 1.0))
        gain_hat = normalized_reward / max(float(self._last_p[region_id]), 1e-12)
        self.weights[region_id] *= np.exp(self.eta * gain_hat)
        max_weight = np.max(self.weights)
        if not np.isfinite(max_weight) or max_weight > 1e100:
            self.weights = self.weights / max_weight
        self.beliefs[region_id] = reward  # scalar tracking for chosen arm only
        self._step += 1

    def get_name(self) -> str:
        if self.gamma_schedule != "static":
            return f"EXP3({self.gamma_schedule})"
        return "EXP3"


class PropagationModel(ABC):
    @abstractmethod
    def receives(self, region_id: int, source_id: int, tx: Transaction, delta: float) -> bool:
        pass

    @abstractmethod
    def reception_prob(self, region_id: int, source_id: int, remaining_time: float) -> float:
        """Return P(latency <= remaining_time) for a builder in region_id from source_id."""
        pass

    def receive_probabilities(self, source_id: int, remaining_times: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [self.reception_prob(region_id, source_id, remaining_time) for remaining_time in remaining_times]
                for region_id in range(self.n_regions)
            ],
            dtype=float,
        )

class LatencyPropagationModel(PropagationModel):
    """
    Accepts raw empirical latency_mean and latency_std (seconds) and converts
    them to lognormal parameters.
    """

    def __init__(self, latency_mean: np.ndarray, latency_std: np.ndarray):
        self.n_regions = latency_mean.shape[0]
        sigma_ln = np.sqrt(np.log(1 + (latency_std / latency_mean) ** 2))
        self._mu_ln = np.log(latency_mean) - sigma_ln ** 2 / 2
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
        return float(ndtr((np.log(remaining_time) - mu) / sigma))

    def receive_probabilities(self, source_id: int, remaining_times: np.ndarray) -> np.ndarray:
        mu_ln = self._mu_ln[:, source_id][:, None]
        sigma_ln = self._sigma_ln[:, source_id][:, None]
        z = (np.log(remaining_times)[None, :] - mu_ln) / sigma_ln
        return ndtr(z)


class FixedLatencyPropagationModel(PropagationModel):
    """
    Deterministic propagation: a transaction is received iff emission_time + latency <= delta.
    Use for synthetic experiments to eliminate stochastic propagation noise.
    """
    def __init__(self, latency_mean: np.ndarray):
        self.n_regions = latency_mean.shape[0]
        self.latency_mean = latency_mean

    def receives(self, region_id: int, source_id: int, tx: Transaction, delta: float) -> bool:
        return tx.emission_time + self.latency_mean[region_id, source_id] <= delta

    def reception_prob(self, region_id: int, source_id: int, remaining_time: float) -> float:
        return 1.0 if self.latency_mean[region_id, source_id] <= remaining_time else 0.0

    def receive_probabilities(self, source_id: int, remaining_times: np.ndarray) -> np.ndarray:
        return (self.latency_mean[:, source_id][:, None] <= remaining_times[None, :]).astype(float)


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
        pass


class StochasticTransactionGenerator(TransactionGenerator):
    """Poisson count, lognormal value, uniform emission time."""

    def generate(self, source: Source, delta: float) -> List[Transaction]:
        n = np.random.poisson(source.lambda_rate * delta)
        emission_times = np.random.uniform(0, delta, size=n)
        values = np.random.lognormal(source.mu_val, source.sigma_val, size=n)

        return [
            Transaction(source_id=source.id, emission_time=emission_times[i], value=values[i])
            for i in range(n)
        ]


@dataclass
class RoundOutcome:
    all_tx_values: Dict[int, float]
    actual_tx_receivers: Dict[int, List[int]]
    tx_receiving_regions: Dict[int, List[int]]
    rewards: Dict[int, float]
    tx_emitted_count: int
    tx_received_count: int
    builder_region_receipts: Optional[Dict[int, np.ndarray]] = None


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
    over time based on observed rewards, using a configured policy or better-response dynamic.
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

        self._expected_env_cache: Dict[int, Dict[str, np.ndarray]] = {}

        self._initialize_builder_distribution()

        self.region_counts_history: List[np.ndarray] = []
        self.reward_history: List[List[float]] = []
        self.welfare_history: List[float] = []
        self.builder_distribution_history: List[np.ndarray] = []
        self.region_reward_pairs_history: List[List[tuple]] = []
        self.tx_emitted_history: List[float] = []
        self.tx_received_history: List[float] = []
        self.abr_adaptation_steps: int = 0
        self.abr_update_mode: str = "async"
        self.abr_converged: bool = False
        self.abr_final_profile: Optional[List[int]] = None
        self.abr_max_profitable_deviation: float = 0.0
        self.abr_cycle_detected: bool = False
        self.abr_cycle_length: Optional[int] = None
        self.cce_gap_over_time = np.array([], dtype=float)
        self.cce_gap_by_builder = np.array([], dtype=float)
        self.cce_best_deviation_regions = np.array([], dtype=int)

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
        distribution = np.zeros(self.n_regions)
        for builder in self.builders:
            distribution[builder.current_region] += 1
        return distribution

    def _current_profile(self) -> List[int]:
        return [builder.current_region for builder in self.builders]

    def _record_state(
        self,
        region_counts: np.ndarray,
        slot_rewards: List[float],
        welfare: float,
        tx_emitted: float,
        tx_received: float,
    ):
        self.region_counts_history.append(region_counts.copy())
        self.reward_history.append(list(slot_rewards))
        self.welfare_history.append(float(welfare))
        self.builder_distribution_history.append(self._get_builder_distribution())
        self.region_reward_pairs_history.append(
            [(builder.current_region, slot_rewards[builder.id]) for builder in self.builders]
        )
        self.tx_emitted_history.append(float(tx_emitted))
        self.tx_received_history.append(float(tx_received))

    def _clear_history(self):
        self.region_counts_history.clear()
        self.reward_history.clear()
        self.welfare_history.clear()
        self.builder_distribution_history.clear()
        self.region_reward_pairs_history.clear()
        self.tx_emitted_history.clear()
        self.tx_received_history.clear()

    def _set_profile(self, profile: List[int]):
        for builder, region_id in zip(self.builders, profile):
            builder.set_region(region_id)

    def _simulate_round_for_profile(
        self,
        builder_selected_regions: Dict[int, int],
        evaluate_all_regions: bool = False,
    ) -> RoundOutcome:
        actual_tx_receivers: Dict[int, List[int]] = {}
        tx_receiving_regions: Dict[int, List[int]] = {}
        all_tx_values: Dict[int, float] = {}
        builder_region_receipts: Optional[Dict[int, np.ndarray]] = {} if evaluate_all_regions else None
        tx_emitted_counter = 0
        tx_received_counter = 0

        for source in self.sources:
            txs = self.tx_generator.generate(source, self.delta)
            for tx in txs:
                tx_id = tx_emitted_counter
                all_tx_values[tx_id] = tx.value

                actual_receivers = []
                receiving_regions = set()

                if evaluate_all_regions:
                    candidate_receipts = np.zeros((self.n_builders, self.n_regions), dtype=bool)
                    for builder_id in range(self.n_builders):
                        for region_id in range(self.n_regions):
                            candidate_receipts[builder_id, region_id] = self.propagation_model.receives(
                                region_id,
                                source.id,
                                tx,
                                self.delta,
                            )

                    builder_region_receipts[tx_id] = candidate_receipts

                    for builder_id, region_id in builder_selected_regions.items():
                        if candidate_receipts[builder_id, region_id]:
                            actual_receivers.append(builder_id)
                            receiving_regions.add(region_id)
                else:
                    for builder_id, region_id in builder_selected_regions.items():
                        if self.propagation_model.receives(region_id, source.id, tx, self.delta):
                            actual_receivers.append(builder_id)
                            receiving_regions.add(region_id)

                tx_receiving_regions[tx_id] = sorted(receiving_regions)
                if actual_receivers:
                    actual_tx_receivers[tx_id] = actual_receivers
                    tx_received_counter += 1

                tx_emitted_counter += 1

        captured_tx_values = {tx_id: all_tx_values[tx_id] for tx_id in actual_tx_receivers}
        rewards = self.sharing_rule.compute_rewards(
            tx_values=captured_tx_values,
            tx_receivers=actual_tx_receivers,
        )

        return RoundOutcome(
            all_tx_values=all_tx_values,
            actual_tx_receivers=actual_tx_receivers,
            tx_receiving_regions=tx_receiving_regions,
            rewards=rewards,
            tx_emitted_count=tx_emitted_counter,
            tx_received_count=tx_received_counter,
            builder_region_receipts=builder_region_receipts,
        )

    def run_round(self):
        builder_selected_regions = {builder.id: builder.choose_region() for builder in self.builders}
        outcome = self._simulate_round_for_profile(builder_selected_regions, evaluate_all_regions=False)

        slot_rewards = []
        for builder in self.builders:
            reward = outcome.rewards.get(builder.id, 0.0)
            builder.update(builder_selected_regions[builder.id], reward)
            slot_rewards.append(reward)

        region_counts = np.zeros(self.n_regions)
        for region_id in builder_selected_regions.values():
            region_counts[region_id] += 1

        self._record_state(
            region_counts=region_counts,
            slot_rewards=slot_rewards,
            welfare=float(sum(slot_rewards)),
            tx_emitted=outcome.tx_emitted_count,
            tx_received=outcome.tx_received_count,
        )

    def _get_expected_environment(self, n_time_steps: int) -> Dict[str, np.ndarray]:
        cached = self._expected_env_cache.get(n_time_steps)
        if cached is not None:
            return cached

        t = np.linspace(0, self.delta, n_time_steps + 1)[:-1]
        remaining = self.delta - t

        q_by_source = np.zeros((self.n_sources, self.n_regions, n_time_steps))
        for source_idx in range(self.n_sources):
            q_by_source[source_idx] = self.propagation_model.receive_probabilities(source_idx, remaining)

        source_value_weights = np.array(
            [
                source.lambda_rate * self.delta * np.exp(source.mu_val + source.sigma_val ** 2 / 2)
                for source in self.sources
            ],
            dtype=float,
        )
        source_tx_weights = np.array(
            [source.lambda_rate * self.delta for source in self.sources],
            dtype=float,
        )

        cached = {
            "q_by_source": q_by_source,
            "source_value_weights": source_value_weights,
            "source_tx_weights": source_tx_weights,
        }
        self._expected_env_cache[n_time_steps] = cached
        return cached

    def _expected_utility_from_candidate_region(
        self,
        candidate_region: int,
        counts_other: np.ndarray,
        q_by_source: np.ndarray,
        source_value_weights: np.ndarray,
    ) -> float:
        total_utility = 0.0
        n_time_steps = q_by_source.shape[2]
        total_other_count = int(np.sum(counts_other))

        for source_idx in range(self.n_sources):
            q_source = q_by_source[source_idx]
            q_self = q_source[candidate_region]

            distribution = np.zeros((n_time_steps, total_other_count + 1))
            distribution[:, 0] = 1.0

            for region_idx, count in enumerate(counts_other):
                if count <= 0:
                    continue
                q_other = q_source[region_idx]
                for _ in range(int(count)):
                    updated = distribution * (1.0 - q_other)[:, None]
                    updated[:, 1:] += distribution[:, :-1] * q_other[:, None]
                    distribution = updated

            expected_share_factor = distribution @ (1.0 / (1.0 + np.arange(total_other_count + 1)))
            total_utility += source_value_weights[source_idx] * np.mean(q_self * expected_share_factor)

        return float(total_utility)

    def compute_candidate_utilities_for_builder(
        self,
        builder_id: int,
        profile: Optional[List[int]] = None,
        n_time_steps: int = 200,
    ) -> np.ndarray:
        if profile is None:
            profile = self._current_profile()

        counts = np.bincount(profile, minlength=self.n_regions).astype(int)
        current_region = profile[builder_id]
        counts_other = counts.copy()
        counts_other[current_region] -= 1

        env = self._get_expected_environment(n_time_steps)
        q_by_source = env["q_by_source"]
        source_value_weights = env["source_value_weights"]

        return np.array(
            [
                self._expected_utility_from_candidate_region(
                    candidate_region=region_id,
                    counts_other=counts_other,
                    q_by_source=q_by_source,
                    source_value_weights=source_value_weights,
                )
                for region_id in range(self.n_regions)
            ]
        )

    def _select_async_response_region(
        self,
        candidate_utilities: np.ndarray,
        current_region: int,
        improvement_threshold_pct: float,
        response_rule: str,
    ) -> Optional[int]:
        current_utility = float(candidate_utilities[current_region])
        improvement_threshold = improvement_threshold_pct * current_utility
        improving_regions = np.flatnonzero(candidate_utilities > current_utility + improvement_threshold)
        improving_regions = improving_regions[improving_regions != current_region]

        if len(improving_regions) == 0:
            return None

        if response_rule == "best":
            return int(improving_regions[np.argmax(candidate_utilities[improving_regions])])
        if response_rule == "better":
            return int(np.random.choice(improving_regions))
        raise ValueError(
            f"Unknown ABR response rule: {response_rule!r}. "
            f"Expected one of {sorted(ABR_RESPONSE_RULES)}."
        )

    def compute_expected_builder_utilities(
        self,
        profile: Optional[List[int]] = None,
        n_time_steps: int = 200,
    ) -> np.ndarray:
        if profile is None:
            profile = self._current_profile()

        counts = np.bincount(profile, minlength=self.n_regions).astype(int)
        env = self._get_expected_environment(n_time_steps)
        q_by_source = env["q_by_source"]
        source_value_weights = env["source_value_weights"]

        utilities = np.zeros(self.n_builders)
        cache: Dict[tuple, float] = {}

        for builder_id, current_region in enumerate(profile):
            counts_other = counts.copy()
            counts_other[current_region] -= 1
            cache_key = (current_region, tuple(counts_other.tolist()))
            if cache_key not in cache:
                cache[cache_key] = self._expected_utility_from_candidate_region(
                    candidate_region=current_region,
                    counts_other=counts_other,
                    q_by_source=q_by_source,
                    source_value_weights=source_value_weights,
                )
            utilities[builder_id] = cache[cache_key]

        return utilities

    def compute_expected_welfare(
        self,
        profile: Optional[List[int]] = None,
        n_time_steps: int = 200,
    ) -> float:
        if profile is None:
            profile = self._current_profile()
        return float(np.sum(self.compute_expected_builder_utilities(profile=profile, n_time_steps=n_time_steps)))

    def compute_expected_covered_transactions(
        self,
        profile: Optional[List[int]] = None,
        n_time_steps: int = 200,
    ) -> float:
        if profile is None:
            profile = self._current_profile()

        counts = np.bincount(profile, minlength=self.n_regions).astype(int)
        env = self._get_expected_environment(n_time_steps)
        q_by_source = env["q_by_source"]
        source_tx_weights = env["source_tx_weights"]

        expected_captured = 0.0
        for source_idx in range(self.n_sources):
            q_source = q_by_source[source_idx]
            no_coverage = np.ones(n_time_steps)
            for region_idx, count in enumerate(counts):
                if count > 0:
                    no_coverage *= np.power(1.0 - q_source[region_idx], count)
            expected_captured += source_tx_weights[source_idx] * np.mean(1.0 - no_coverage)

        return float(expected_captured)

    def evaluate_fixed_profile(self, n_slots: int, profile: Optional[List[int]] = None):
        if profile is None:
            profile = self._current_profile()
        else:
            self._set_profile(profile)

        builder_selected_regions = {builder.id: builder.current_region for builder in self.builders}
        for _ in range(n_slots):
            outcome = self._simulate_round_for_profile(
                builder_selected_regions=builder_selected_regions,
                evaluate_all_regions=False,
            )
            region_counts = self._get_builder_distribution()
            slot_rewards = [outcome.rewards.get(builder.id, 0.0) for builder in self.builders]
            self._record_state(
                region_counts=region_counts,
                slot_rewards=slot_rewards,
                welfare=float(sum(slot_rewards)),
                tx_emitted=outcome.tx_emitted_count,
                tx_received=outcome.tx_received_count,
            )

    def verify_pure_nash_equilibrium(
        self,
        profile: Optional[List[int]] = None,
        n_time_steps: int = 200,
        tolerance: float = 1e-12,
    ) -> Dict[str, object]:
        if profile is None:
            profile = self._current_profile()

        profile = list(profile)
        gains = np.zeros((self.n_builders, self.n_regions))
        profitable_deviations = []
        max_gain = -np.inf

        for builder_id in range(self.n_builders):
            candidate_utilities = self.compute_candidate_utilities_for_builder(
                builder_id=builder_id,
                profile=profile,
                n_time_steps=n_time_steps,
            )
            current_region = profile[builder_id]
            current_utility = candidate_utilities[current_region]
            builder_gains = candidate_utilities - current_utility
            gains[builder_id] = builder_gains

            best_region = int(np.argmax(candidate_utilities))
            best_gain = float(builder_gains[best_region])
            max_gain = max(max_gain, best_gain)
            if best_gain > tolerance:
                profitable_deviations.append(
                    {
                        "builder_id": builder_id,
                        "current_region": current_region,
                        "best_region": best_region,
                        "gain": best_gain,
                    }
                )

        if max_gain == -np.inf:
            max_gain = 0.0

        return {
            "is_pure_ne": len(profitable_deviations) == 0,
            "max_gain": float(max_gain),
            "gains": gains,
            "profitable_deviations": profitable_deviations,
        }

    def run_async_better_response(
        self,
        n_slots: int,
        improvement_threshold_pct: float = 0.001,
        n_time_steps: int = 200,
        max_updates: Optional[int] = None,
        response_rule: str = "best",
    ):
        response_rule = response_rule.lower()
        if response_rule not in ABR_RESPONSE_RULES:
            raise ValueError(
                f"Unknown ABR response rule: {response_rule!r}. "
                f"Expected one of {sorted(ABR_RESPONSE_RULES)}."
            )

        if max_updates is None:
            max_updates = max(n_slots, self.n_builders)

        updates = 0
        converged = False

        while updates < max_updates:
            moved_in_pass = False
            for builder_id in np.random.permutation(self.n_builders):
                profile = self._current_profile()
                candidate_utilities = self.compute_candidate_utilities_for_builder(
                    builder_id=builder_id,
                    profile=profile,
                    n_time_steps=n_time_steps,
                )

                current_region = profile[builder_id]
                next_region = self._select_async_response_region(
                    candidate_utilities=candidate_utilities,
                    current_region=current_region,
                    improvement_threshold_pct=improvement_threshold_pct,
                    response_rule=response_rule,
                )

                if next_region is not None:
                    self.builders[builder_id].set_region(next_region)
                    updates += 1
                    moved_in_pass = True
                    if updates >= max_updates:
                        break

            if not moved_in_pass:
                converged = True
                break

        final_profile = self._current_profile()
        ne_check = self.verify_pure_nash_equilibrium(
            profile=final_profile,
            n_time_steps=n_time_steps,
            tolerance=1e-12,
        )

        self.abr_adaptation_steps = updates
        self.abr_update_mode = "async"
        self.abr_converged = converged and bool(ne_check["is_pure_ne"])
        self.abr_final_profile = list(final_profile)
        self.abr_max_profitable_deviation = float(ne_check["max_gain"])
        self.abr_cycle_detected = False
        self.abr_cycle_length = None

        self._clear_history()
        self.evaluate_fixed_profile(n_slots=n_slots, profile=final_profile)

    def run_simultaneous_better_response(
        self,
        n_slots: int,
        improvement_threshold_pct: float = 0.001,
        n_time_steps: int = 200,
        max_rounds: Optional[int] = None,
        response_rule: str = "best",
        detect_cycles: bool = True,
    ):
        """Run synchronous exact-response dynamics.

        In each adaptation round, every builder computes its response against the
        same pre-update profile. All selected moves are then committed together.
        A fixed point is a pure Nash equilibrium; simultaneous updates may also
        enter cycles, so repeated profiles are tracked separately from convergence.
        """
        response_rule = response_rule.lower()
        if response_rule not in ABR_RESPONSE_RULES:
            raise ValueError(
                f"Unknown ABR response rule: {response_rule!r}. "
                f"Expected one of {sorted(ABR_RESPONSE_RULES)}."
            )

        if max_rounds is None:
            max_rounds = max(n_slots, self.n_builders)

        rounds = 0
        converged = False
        cycle_detected = False
        cycle_length = None
        seen_profiles = {tuple(self._current_profile()): 0} if detect_cycles else {}

        while rounds < max_rounds:
            profile = self._current_profile()
            next_profile = list(profile)
            moved_in_round = False

            for builder_id in range(self.n_builders):
                candidate_utilities = self.compute_candidate_utilities_for_builder(
                    builder_id=builder_id,
                    profile=profile,
                    n_time_steps=n_time_steps,
                )
                current_region = profile[builder_id]
                next_region = self._select_async_response_region(
                    candidate_utilities=candidate_utilities,
                    current_region=current_region,
                    improvement_threshold_pct=improvement_threshold_pct,
                    response_rule=response_rule,
                )
                if next_region is not None:
                    next_profile[builder_id] = next_region
                    moved_in_round = True

            if not moved_in_round:
                converged = True
                break

            rounds += 1
            self._set_profile(next_profile)

            if detect_cycles:
                profile_key = tuple(next_profile)
                previous_round = seen_profiles.get(profile_key)
                if previous_round is not None:
                    cycle_detected = True
                    cycle_length = rounds - previous_round
                    break
                seen_profiles[profile_key] = rounds

        final_profile = self._current_profile()
        ne_check = self.verify_pure_nash_equilibrium(
            profile=final_profile,
            n_time_steps=n_time_steps,
            tolerance=1e-12,
        )

        self.abr_adaptation_steps = rounds
        self.abr_update_mode = "simultaneous"
        self.abr_converged = converged and bool(ne_check["is_pure_ne"])
        self.abr_final_profile = list(final_profile)
        self.abr_max_profitable_deviation = float(ne_check["max_gain"])
        self.abr_cycle_detected = cycle_detected
        self.abr_cycle_length = cycle_length

        self._clear_history()
        self.evaluate_fixed_profile(n_slots=n_slots, profile=final_profile)

    def _compute_mwu_counterfactual_payoffs(
        self,
        builder_selected_regions: Dict[int, int],
        outcome: RoundOutcome,
    ) -> np.ndarray:
        if outcome.builder_region_receipts is None:
            raise ValueError("MWU counterfactual payoffs require builder-level receipt samples.")

        payoffs = np.zeros((self.n_builders, self.n_regions))
        del builder_selected_regions

        for tx_id, candidate_receipts in outcome.builder_region_receipts.items():
            value = outcome.all_tx_values[tx_id]
            actual_receivers = outcome.actual_tx_receivers.get(tx_id, [])
            actual_receiver_count = len(actual_receivers)
            actual_receiver_mask = np.zeros(self.n_builders, dtype=bool)
            actual_receiver_mask[actual_receivers] = True

            for builder_id in range(self.n_builders):
                other_receivers = actual_receiver_count - int(actual_receiver_mask[builder_id])
                payoffs[builder_id, candidate_receipts[builder_id]] += value / (1 + other_receivers)

        return payoffs

    def run_mwu(
        self,
        n_slots: int,
        eta: float = 0.1,
        payoff_normalization: Optional[float] = None,
    ):
        weights = np.ones((self.n_builders, self.n_regions), dtype=float)
        if payoff_normalization is None or payoff_normalization <= 0:
            payoff_normalization = float(
                sum(
                    source.lambda_rate * self.delta * np.exp(source.mu_val + source.sigma_val ** 2 / 2)
                    for source in self.sources
                )
            )
        payoff_normalization = max(payoff_normalization, 1.0)

        for _ in range(n_slots):
            probabilities = weights / weights.sum(axis=1, keepdims=True)
            builder_selected_regions = {}
            for builder in self.builders:
                chosen_region = int(np.random.choice(self.n_regions, p=probabilities[builder.id]))
                builder.set_region(chosen_region)
                builder_selected_regions[builder.id] = chosen_region

            outcome = self._simulate_round_for_profile(builder_selected_regions, evaluate_all_regions=True)
            counterfactual_payoffs = self._compute_mwu_counterfactual_payoffs(builder_selected_regions, outcome)
            normalized_payoffs = np.clip(counterfactual_payoffs / payoff_normalization, 0.0, 1.0)
            weights *= np.exp(eta * normalized_payoffs)

            slot_rewards = [outcome.rewards.get(builder.id, 0.0) for builder in self.builders]
            region_counts = self._get_builder_distribution()
            self._record_state(
                region_counts=region_counts,
                slot_rewards=slot_rewards,
                welfare=float(sum(slot_rewards)),
                tx_emitted=outcome.tx_emitted_count,
                tx_received=outcome.tx_received_count,
            )

    def run(self, n_slots: int):
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

    def compute_empirical_cce_gap(self, n_time_steps: int = 200) -> Dict[str, np.ndarray | float]:
        n_slots = len(self.region_reward_pairs_history)
        if n_slots == 0:
            zero_builder_gaps = np.zeros(self.n_builders, dtype=float)
            zero_regions = np.zeros(self.n_builders, dtype=int)
            zero_gap_series = np.zeros(0, dtype=float)
            self.cce_gap_over_time = zero_gap_series
            self.cce_gap_by_builder = zero_builder_gaps
            self.cce_best_deviation_regions = zero_regions
            return {
                "cce_gap": 0.0,
                "cce_gap_by_builder": zero_builder_gaps,
                "cce_best_deviation_regions": zero_regions,
                "cce_gap_over_time": zero_gap_series,
            }

        env = self._get_expected_environment(n_time_steps)
        q_by_source = env["q_by_source"]
        source_value_weights = env["source_value_weights"]

        cumulative_regrets = np.zeros((self.n_builders, self.n_regions), dtype=float)
        gap_over_time = np.zeros(n_slots, dtype=float)
        candidate_utility_cache: Dict[tuple, np.ndarray] = {}

        for slot_idx, slot_profile in enumerate(self.region_reward_pairs_history):
            profile = [region_id for region_id, _ in slot_profile]
            counts = np.bincount(profile, minlength=self.n_regions).astype(int)

            for builder_id, current_region in enumerate(profile):
                counts_other = counts.copy()
                counts_other[current_region] -= 1
                cache_key = (current_region, tuple(counts_other.tolist()))
                candidate_utilities = candidate_utility_cache.get(cache_key)
                if candidate_utilities is None:
                    candidate_utilities = np.array(
                        [
                            self._expected_utility_from_candidate_region(
                                candidate_region=region_id,
                                counts_other=counts_other,
                                q_by_source=q_by_source,
                                source_value_weights=source_value_weights,
                            )
                            for region_id in range(self.n_regions)
                        ],
                        dtype=float,
                    )
                    candidate_utility_cache[cache_key] = candidate_utilities

                played_utility = candidate_utilities[current_region]
                cumulative_regrets[builder_id] += candidate_utilities - played_utility

            gap_over_time[slot_idx] = max(0.0, float(np.max(cumulative_regrets / (slot_idx + 1))))

        average_regrets = cumulative_regrets / n_slots
        cce_gap_by_builder = np.maximum(0.0, np.max(average_regrets, axis=1))
        best_deviation_regions = np.argmax(average_regrets, axis=1).astype(int)
        cce_gap = float(np.max(cce_gap_by_builder)) if len(cce_gap_by_builder) > 0 else 0.0

        self.cce_gap_over_time = gap_over_time
        self.cce_gap_by_builder = cce_gap_by_builder
        self.cce_best_deviation_regions = best_deviation_regions

        return {
            "cce_gap": cce_gap,
            "cce_gap_by_builder": cce_gap_by_builder,
            "cce_best_deviation_regions": best_deviation_regions,
            "cce_gap_over_time": gap_over_time,
        }

    def get_statistics(self) -> Dict:
        cce_stats = self.compute_empirical_cce_gap(n_time_steps=50)
        region_counts = np.array(self.region_counts_history)
        builder_distribution = np.array(self.builder_distribution_history)

        avg_region_counts = np.mean(region_counts, axis=0) if len(region_counts) > 0 else np.zeros(self.n_regions)
        avg_builder_distribution = (
            np.mean(builder_distribution, axis=0) if len(builder_distribution) > 0 else np.zeros(self.n_regions)
        )

        all_rewards = [reward for slot_rewards in self.reward_history for reward in slot_rewards]
        avg_reward = float(np.mean(all_rewards)) if all_rewards else 0.0

        welfare = np.array(self.welfare_history)
        mean_txs_emitted = float(np.mean(self.tx_emitted_history)) if self.tx_emitted_history else 0.0
        mean_txs_received = float(np.mean(self.tx_received_history)) if self.tx_received_history else 0.0
        coverage_per_round = [
            received / emitted if emitted > 0 else 0.0
            for emitted, received in zip(self.tx_emitted_history, self.tx_received_history)
        ]
        mean_coverage_ratio = float(np.mean(coverage_per_round)) if coverage_per_round else 0.0
        mean_txs_per_builder = mean_txs_received / self.n_builders if self.n_builders > 0 else 0.0
        all_slot_rewards = [sum(slot_rewards) / len(slot_rewards) for slot_rewards in self.reward_history if slot_rewards]
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
            "avg_region_counts": avg_region_counts,
            "avg_builder_distribution": avg_builder_distribution,
            "avg_reward": avg_reward,
            "region_gini": gini(avg_region_counts),
            "builder_dist_gini": gini(avg_builder_distribution),
            "region_entropy": entropy(avg_region_counts),
            "builder_dist_entropy": entropy(avg_builder_distribution),
            "location_gini": gini(last_builder_dist),
            "location_entropy": entropy(last_builder_dist),
            "location_hhi": hhi(last_builder_dist),
            "utility_gini": gini(utilities),
            "utility_entropy": entropy(utilities),
            "utility_hhi": hhi(utilities),
            "total_slots": len(self.region_counts_history),
            "mean_welfare": float(np.mean(welfare)) if len(welfare) > 0 else 0.0,
            "mean_txs_emitted_per_round": mean_txs_emitted,
            "mean_txs_received_per_round": mean_txs_received,
            "mean_coverage_ratio": mean_coverage_ratio,
            "mean_txs_received_per_builder": mean_txs_per_builder,
            "mean_value_per_builder": mean_value_per_builder,
            "abr_adaptation_steps": self.abr_adaptation_steps,
            "abr_update_mode": self.abr_update_mode,
            "abr_converged": self.abr_converged,
            "abr_final_profile": self.abr_final_profile,
            "abr_max_profitable_deviation": self.abr_max_profitable_deviation,
            "abr_cycle_detected": self.abr_cycle_detected,
            "abr_cycle_length": self.abr_cycle_length,
            "cce_gap": cce_stats["cce_gap"],
            "cce_gap_by_builder": cce_stats["cce_gap_by_builder"],
            "cce_best_deviation_regions": cce_stats["cce_best_deviation_regions"],
        }
