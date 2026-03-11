#!/usr/bin/env python3
"""
Decentralized Building Simulator (db-sims) - Core simulation engine

Inspired by GeoDec research, generalized for studying location choice in various
distributed and decentralized block building regimes.

Implements two interchangeable learning policies:
  (A) EMA + softmax
  (B) Individual UCB bandit
"""
from collections import defaultdict

import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod

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


class EMASoftmaxPolicy(LearningPolicy):
    """
    Policy A: EMA + softmax.
    Individual scoreboard per builder.
    """

    def __init__(self, n_regions: int, eta: float = 0.1,
                 beta: float = 2.0,
                 cost: float = 0.0,
                 initial_belief: float = 0.0):
        """
        Args:
            eta: EMA learning rate
            beta: Temperature for region selection
            cost: Migration cost (subtracted from region score)
        """
        super().__init__(n_regions, initial_belief)
        self.eta = eta
        self.beta = beta
        self.cost = cost

    def choose(self, current_region: int) -> int:
        """Two-stage softmax selection."""
        # Stage 1: Choose region
        U_reg = self.beliefs

        # Softmax over regions
        # we subtract the largest value to ensure no overflow
        shifted = self.beta * (U_reg - np.max(U_reg))
        exp_scores = np.exp(shifted)
        probs_reg = exp_scores / np.sum(exp_scores)
        region_id = np.random.choice(len(self.beliefs), p=probs_reg)

        if U_reg[region_id] - U_reg[current_region] <= self.cost:
            return current_region

        return region_id

    def update(self, region_id: int, reward: float):
        """EMA update: beliefs(r) <- (1-eta)*beliefs(r) + eta*R."""
        self.beliefs[region_id] = (
            (1 - self.eta) * self.beliefs[region_id] +
            self.eta * reward
        )

    def get_name(self) -> str:
        return "EMA-Softmax"


class UCBPolicy(LearningPolicy):
    """
    Policy B: Individual UCB bandit.
    Arms are regions.
    """

    def __init__(self, n_regions: int, alpha: float = 1.0, cost: float = 0.0,
                 initial_belief: float = 0.0):
        """
        Args:
            alpha: Exploration parameter
            cost: Migration cost
        """
        super().__init__(n_regions, initial_belief)
        self.alpha = alpha
        self.cost = cost

        self.N = np.zeros(len(self.beliefs))
        self.t = 0  # Local clock

    def choose(self, current_region: int) -> int:
        """UCB selection: argmax [beliefs(r) + alpha*sqrt(log(1+t)/(1+N(r,I)))]."""
        # Compute UCB scores
        exploration_bonus = self.alpha * np.sqrt(
            np.log(1 + self.t) / (1 + self.N)
        )
        ucb_scores = self.beliefs + exploration_bonus

        region_id = np.argmax(ucb_scores)
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
    """Policy that never moves. We use this for welfare estimation at fixed profiles."""
    def choose(self, current_region: int) -> int:
        return current_region

    def update(self, region_id: int, reward: float):
        pass

    def get_name(self) -> str:
        return "Fixed"


class PropagationModel(ABC):
    @abstractmethod
    def receives(self, region_id: int, source_id: int, tx: Transaction, delta: float) -> bool:
        """Determine if a builder in region_id receives transaction tx from source_id within delta."""
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


class LocationGamesSimulator:
    """
    Core simulator for studying location choice in decentralized block building.

    Implements reward sharing among builders based on their chosen regions and the transactions they capture
    which are generated stochastically from information sources. Builders learn and adapt their region choices 
    over time based on observed rewards, using either an EMA-softmax policy or a UCB bandit policy.
    """

    def __init__(self,
                 regions: List[Region],
                 sources: List[Source],
                 builders: List[Builder],
                 tx_generator: TransactionGenerator,
                 propagation_model: PropagationModel,
                 sharing_rule: SharingRule,
                 delta: float,
                 seed: int = 42):
        """
        Args:
            regions: List of regions
            sources: List of sources
            builders: List of builders (agents)
            tx_generator: Transaction generator
            propagation_model: Propagation model
            sharing_rule: Sharing rule
            delta: Delta parameter
            seed: Random seed
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

        np.random.seed(seed)

        # Initialize builders evenly across regions
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
        """Initialize builders evenly across regions."""
        for i, builder in enumerate(self.builders):
            # Distribute evenly: builder i goes to region (i mod n_regions)
            initial_region = i % self.n_regions
            builder.set_region(initial_region)

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
            (builder_selected_regions[b.id], rewards.get(b.id, 0.0)) for b in self.builders
        ])
        self.tx_emitted_history.append(tx_emitted_counter)
        self.tx_received_history.append(tx_received_counter)


    def run(self, n_slots: int):
        """Run simulation for n_slots."""
        for _ in range(n_slots):
            self.run_round()

    def get_statistics(self) -> Dict:
        """Compute summary statistics."""
        region_counts = np.array(self.region_counts_history)
        builder_distribution = np.array(self.builder_distribution_history)

        avg_region_counts = np.mean(region_counts, axis=0) if len(region_counts) > 0 else np.zeros(self.n_regions)
        avg_builder_distribution = np.mean(builder_distribution, axis=0) if len(builder_distribution) > 0 else np.zeros(self.n_regions)

        all_rewards = [r for slot_rewards in self.reward_history for r in slot_rewards]
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
        coverage_per_round = [r / e if e > 0 else 0.0
                              for e, r in zip(self.tx_emitted_history, self.tx_received_history)]
        mean_coverage_ratio = float(np.mean(coverage_per_round)) if coverage_per_round else 0.0
        mean_txs_per_builder = mean_txs_received / self.n_builders if self.n_builders > 0 else 0.0
        all_slot_rewards = [sum(s) / len(s) for s in self.reward_history if s]
        mean_value_per_builder = float(np.mean(all_slot_rewards)) if all_slot_rewards else 0.0

        return {
            'avg_region_counts': avg_region_counts,
            'avg_builder_distribution': avg_builder_distribution,
            'avg_reward': avg_reward,
            'region_gini': gini(avg_region_counts),
            'builder_dist_gini': gini(avg_builder_distribution),
            'region_entropy': entropy(avg_region_counts),
            'builder_dist_entropy': entropy(avg_builder_distribution),
            'total_slots': len(self.region_counts_history),
            'mean_welfare': float(np.mean(welfare)) if len(welfare) > 0 else 0.0,
            'mean_txs_emitted_per_round': mean_txs_emitted,
            'mean_txs_received_per_round': mean_txs_received,
            'mean_coverage_ratio': mean_coverage_ratio,
            'mean_txs_received_per_builder': mean_txs_per_builder,
            'mean_value_per_builder': mean_value_per_builder,
        }
