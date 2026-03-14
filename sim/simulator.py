#!/usr/bin/env python3
"""
Decentralized Building Simulator (db-sims) - Core simulation engine.

Supported dynamics:
  (A) EMA + softmax
  (B) Individual UCB bandit
  (C) Individual EXP3 bandit
  (D) Asynchronous strict better response with exact utility evaluation
  (E) Full-information multiplicative weights update (MWU)
"""
from collections import defaultdict

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

from scipy.special import ndtr


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


class EXP3Policy(LearningPolicy):
    """Policy C: individual EXP3 bandit with clipped rewards."""

    def __init__(
        self,
        n_regions: int,
        gamma: float = 0.07,
        payoff_normalization: float = 1.0,
        initial_belief: float = 0.0,
    ):
        super().__init__(n_regions, initial_belief)
        self.gamma = float(np.clip(gamma, 1e-9, 1.0))
        self.payoff_normalization = max(float(payoff_normalization), 1.0)
        self.weights = np.ones(len(self.beliefs), dtype=float)
        self.last_probabilities = np.ones(len(self.beliefs), dtype=float) / len(self.beliefs)
        self.last_action: Optional[int] = None

    def choose(self, current_region: int) -> int:
        del current_region  # EXP3 samples directly from its mixed strategy.

        weight_sum = np.sum(self.weights)
        if weight_sum <= 0:
            base_probs = np.ones(len(self.weights), dtype=float) / len(self.weights)
        else:
            base_probs = self.weights / weight_sum

        n_regions = len(self.weights)
        self.last_probabilities = (1.0 - self.gamma) * base_probs + self.gamma / n_regions
        region_id = int(np.random.choice(n_regions, p=self.last_probabilities))
        self.last_action = region_id
        return region_id

    def update(self, region_id: int, reward: float):
        if self.last_action is None:
            return

        del region_id
        action = self.last_action
        normalized_reward = float(np.clip(reward / self.payoff_normalization, 0.0, 1.0))
        probability = max(float(self.last_probabilities[action]), 1e-12)
        estimated_reward = normalized_reward / probability
        self.weights[action] *= np.exp(self.gamma * estimated_reward / len(self.weights))
        self.beliefs[action] = estimated_reward
        self.last_action = None

    def get_name(self) -> str:
        return "EXP3"


class FixedPolicy(LearningPolicy):
    """Policy that never moves."""

    def choose(self, current_region: int) -> int:
        return current_region

    def update(self, region_id: int, reward: float):
        pass

    def get_name(self) -> str:
        return "Fixed"


class PropagationModel(ABC):
    @abstractmethod
    def receives(self, region_id: int, source_id: int, tx: Transaction, delta: float) -> bool:
        pass


class LatencyPropagationModel(PropagationModel):
    """
    Accepts raw empirical latency_mean and latency_std (seconds) and converts
    them to lognormal parameters.
    """

    def __init__(self, latency_mean: np.ndarray, latency_std: np.ndarray):
        sigma_ln = np.sqrt(np.log(1 + (latency_std / latency_mean) ** 2))
        self._mu_ln = np.log(latency_mean) - sigma_ln ** 2 / 2
        self._sigma_ln = sigma_ln

    def receives(self, region_id: int, source_id: int, tx: Transaction, delta: float) -> bool:
        d = np.random.lognormal(self._mu_ln[region_id, source_id], self._sigma_ln[region_id, source_id])
        return tx.emission_time + d <= delta

    def receive_probabilities(self, source_id: int, remaining_times: np.ndarray) -> np.ndarray:
        mu_ln = self._mu_ln[:, source_id][:, None]
        sigma_ln = self._sigma_ln[:, source_id][:, None]
        z = (np.log(remaining_times)[None, :] - mu_ln) / sigma_ln
        return ndtr(z)


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


class LocationGamesSimulator:
    """Core simulator for studying location choice in decentralized block building."""

    def __init__(
        self,
        regions: List[Region],
        sources: List[Source],
        builders: List[Builder],
        tx_generator: TransactionGenerator,
        propagation_model: PropagationModel,
        sharing_rule: SharingRule,
        delta: float,
        seed: int = 42,
    ):
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
        self.abr_converged: bool = False
        self.abr_final_profile: Optional[List[int]] = None
        self.abr_max_profitable_deviation: float = 0.0

    def _initialize_builder_distribution(self):
        for i, builder in enumerate(self.builders):
            builder.set_region(i % self.n_regions)

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
        tx_emitted_counter = 0
        tx_received_counter = 0

        region_to_builders: Dict[int, List[int]] = defaultdict(list)
        for builder_id, region_id in builder_selected_regions.items():
            region_to_builders[region_id].append(builder_id)

        sampled_regions = range(self.n_regions) if evaluate_all_regions else region_to_builders.keys()

        for source in self.sources:
            txs = self.tx_generator.generate(source, self.delta)
            for tx in txs:
                tx_id = tx_emitted_counter
                all_tx_values[tx_id] = tx.value

                receiving_regions = []
                actual_receivers = []
                for region_id in sampled_regions:
                    if self.propagation_model.receives(region_id, source.id, tx, self.delta):
                        receiving_regions.append(region_id)
                        if region_id in region_to_builders:
                            actual_receivers.extend(region_to_builders[region_id])

                tx_receiving_regions[tx_id] = receiving_regions
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

    def _require_latency_model(self) -> LatencyPropagationModel:
        if not isinstance(self.propagation_model, LatencyPropagationModel):
            raise TypeError("Exact utility evaluation currently requires LatencyPropagationModel.")
        return self.propagation_model

    def _get_expected_environment(self, n_time_steps: int) -> Dict[str, np.ndarray]:
        cached = self._expected_env_cache.get(n_time_steps)
        if cached is not None:
            return cached

        model = self._require_latency_model()
        t = np.linspace(0, self.delta, n_time_steps + 1)[:-1]
        remaining = self.delta - t

        q_by_source = np.zeros((self.n_sources, self.n_regions, n_time_steps))
        for source_idx in range(self.n_sources):
            q_by_source[source_idx] = model.receive_probabilities(source_idx, remaining)

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
        same_region_others = int(counts_other[candidate_region])
        outside_other_count = int(np.sum(counts_other) - same_region_others)

        for source_idx in range(self.n_sources):
            q_source = q_by_source[source_idx]
            q_self = q_source[candidate_region]

            distribution = np.zeros((n_time_steps, outside_other_count + 1))
            distribution[:, 0] = 1.0

            for region_idx, count in enumerate(counts_other):
                if region_idx == candidate_region or count <= 0:
                    continue
                q_other = q_source[region_idx]
                updated = distribution * (1.0 - q_other)[:, None]
                updated[:, count:] += distribution[:, :-count] * q_other[:, None]
                distribution = updated

            denominators = 1.0 + same_region_others + np.arange(outside_other_count + 1)
            expected_share_factor = distribution @ (1.0 / denominators)
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
    ):
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
                current_utility = candidate_utilities[current_region]
                improvement_threshold = improvement_threshold_pct * current_utility
                best_region = int(np.argmax(candidate_utilities))
                best_utility = candidate_utilities[best_region]

                if best_region != current_region and best_utility > current_utility + improvement_threshold:
                    self.builders[builder_id].set_region(best_region)
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
        self.abr_converged = converged and bool(ne_check["is_pure_ne"])
        self.abr_final_profile = list(final_profile)
        self.abr_max_profitable_deviation = float(ne_check["max_gain"])

        self._clear_history()
        self.evaluate_fixed_profile(n_slots=n_slots, profile=final_profile)

    def _compute_mwu_counterfactual_payoffs(
        self,
        builder_selected_regions: Dict[int, int],
        outcome: RoundOutcome,
    ) -> np.ndarray:
        payoffs = np.zeros((self.n_builders, self.n_regions))
        region_counts = np.zeros(self.n_regions, dtype=int)
        selected_region_array = np.zeros(self.n_builders, dtype=int)

        for builder_id, region_id in builder_selected_regions.items():
            region_counts[region_id] += 1
            selected_region_array[builder_id] = region_id

        for tx_id, receiving_regions in outcome.tx_receiving_regions.items():
            if not receiving_regions:
                continue

            value = outcome.all_tx_values[tx_id]
            actual_receiving_builder_count = int(np.sum(region_counts[receiving_regions]))
            receiving_region_mask = np.zeros(self.n_regions, dtype=bool)
            receiving_region_mask[receiving_regions] = True

            for builder_id in range(self.n_builders):
                current_region = selected_region_array[builder_id]
                other_receivers = actual_receiving_builder_count - int(receiving_region_mask[current_region])
                share = value / (1 + other_receivers)
                payoffs[builder_id, receiving_region_mask] += share

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

    def get_statistics(self) -> Dict:
        region_counts = np.array(self.region_counts_history)
        builder_distribution = np.array(self.builder_distribution_history)

        avg_region_counts = np.mean(region_counts, axis=0) if len(region_counts) > 0 else np.zeros(self.n_regions)
        avg_builder_distribution = (
            np.mean(builder_distribution, axis=0) if len(builder_distribution) > 0 else np.zeros(self.n_regions)
        )

        all_rewards = [reward for slot_rewards in self.reward_history for reward in slot_rewards]
        avg_reward = float(np.mean(all_rewards)) if all_rewards else 0.0

        def gini(x: np.ndarray) -> float:
            total = np.sum(x)
            if total == 0:
                return 0.0
            sorted_x = np.sort(x)
            n = len(x)
            cumsum = np.cumsum(sorted_x)
            return float((2 * np.sum((np.arange(1, n + 1)) * sorted_x)) / (n * cumsum[-1]) - (n + 1) / n)

        def entropy(counts: np.ndarray) -> float:
            total = np.sum(counts)
            if total == 0 or len(counts) <= 1:
                return 0.0
            probs = counts / total
            probs = probs[probs > 0]
            return float(-np.sum(probs * np.log(probs)) / np.log(len(counts)))

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

        return {
            "avg_region_counts": avg_region_counts,
            "avg_builder_distribution": avg_builder_distribution,
            "avg_reward": avg_reward,
            "region_gini": gini(avg_region_counts),
            "builder_dist_gini": gini(avg_builder_distribution),
            "region_entropy": entropy(avg_region_counts),
            "builder_dist_entropy": entropy(avg_builder_distribution),
            "total_slots": len(self.region_counts_history),
            "mean_welfare": float(np.mean(welfare)) if len(welfare) > 0 else 0.0,
            "mean_txs_emitted_per_round": mean_txs_emitted,
            "mean_txs_received_per_round": mean_txs_received,
            "mean_coverage_ratio": mean_coverage_ratio,
            "mean_txs_received_per_builder": mean_txs_per_builder,
            "mean_value_per_builder": mean_value_per_builder,
            "abr_adaptation_steps": self.abr_adaptation_steps,
            "abr_converged": self.abr_converged,
            "abr_final_profile": self.abr_final_profile,
            "abr_max_profitable_deviation": self.abr_max_profitable_deviation,
        }
