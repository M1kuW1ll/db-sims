from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
from sim.config import ExperimentConfig
from sim.metrics import gini, entropy, hhi
from sim.simulator import LocationGamesSimulator


class ExperimentResult:
    """Store results from a single experiment."""

    def __init__(self, config: ExperimentConfig, simulator: LocationGamesSimulator):
        self.config = config
        self.simulator = simulator
        self.stats = simulator.get_statistics()

        # Time series data
        self.region_counts = np.array(simulator.region_counts_history)
        self.builder_distribution = np.array(simulator.builder_distribution_history)
        self.rewards = [np.mean(r) if r else 0.0 for r in simulator.reward_history]
        self.region_reward_pairs = simulator.region_reward_pairs_history
        self.welfare_history = np.array(simulator.welfare_history)
        self.tx_emitted_history = np.array(simulator.tx_emitted_history)
        self.tx_received_history = np.array(simulator.tx_received_history)
        self.builder_rewards_history = np.array(simulator.reward_history)

        self.poa_stats = None  # populated by compute_poa_stats if requested

        # Compute time-series metrics
        self._compute_time_series_metrics()

    def _compute_time_series_metrics(self):
        """Compute location and utility concentration metrics over time."""

        n_builders = self.builder_rewards_history.shape[1] if self.builder_rewards_history.ndim == 2 else self.config.n_builders
        n_regions = self.config.n_regions

        self.location_gini_over_time = []
        self.location_entropy_over_time = []
        self.location_hhi_over_time = []

        def top_k_concentration(counts, k):
            """Top-k concentration: sum of top k shares"""
            total = np.sum(counts)
            if total == 0:
                return 0.0
            shares = counts / total
            sorted_shares = np.sort(shares)[::-1]  # Descending order
            return np.sum(sorted_shares[:k])

        def l1_distance(p1, p2):
            """L1 distance between two distributions"""
            return np.sum(np.abs(p1 - p2))

        self.utility_gini_over_time = []
        self.utility_entropy_over_time = []
        self.utility_hhi_over_time = []

        # Value-capture metrics
        self.value_capture_by_region = []
        self.value_share_distribution = []
        self.value_share_hhi_over_time = []
        self.value_share_entropy_over_time = []
        self.value_share_top1_over_time = []
        self.value_share_top3_over_time = []

        # Volatility metrics
        self.region_volatility_over_time = []
        self.builder_dist_volatility_over_time = []
        self.value_share_volatility_over_time = []

        prev_region_shares = None
        prev_builder_shares = None
        prev_value_shares = None

        for t in range(len(self.region_counts)):
            # Location metrics
            builder_dist_t = self.builder_distribution[t]
            self.location_gini_over_time.append(gini(builder_dist_t))
            self.location_entropy_over_time.append(entropy(builder_dist_t))
            self.location_hhi_over_time.append(hhi(builder_dist_t))

            # Utility metrics from observed rewards for each builder
            rewards_t = self.builder_rewards_history[t]
            total_reward = float(np.sum(rewards_t))
            if total_reward == 0:
                self.utility_hhi_over_time.append(1.0 / n_builders)
                self.utility_gini_over_time.append(0.0)
                self.utility_entropy_over_time.append(1.0)
            else:
                self.utility_hhi_over_time.append(hhi(rewards_t))
                self.utility_gini_over_time.append(gini(rewards_t))
                self.utility_entropy_over_time.append(entropy(rewards_t))

            value_by_region = np.zeros(n_regions)
            for region_id, reward in self.region_reward_pairs[t]:
                value_by_region[region_id] += reward

            self.value_capture_by_region.append(value_by_region.copy())

            total_value = np.sum(value_by_region)
            value_shares = value_by_region / total_value if total_value > 0 else np.zeros(n_regions)

            self.value_share_distribution.append(value_shares.copy())
            self.value_share_hhi_over_time.append(hhi(value_by_region))
            self.value_share_entropy_over_time.append(entropy(value_by_region))
            self.value_share_top1_over_time.append(top_k_concentration(value_by_region, 1))
            self.value_share_top3_over_time.append(top_k_concentration(value_by_region, min(3, n_regions)))

            region_total = np.sum(self.region_counts[t])
            current_region_shares = self.region_counts[t] / region_total if region_total > 0 else np.zeros(n_regions)

            builder_total = np.sum(builder_dist_t)
            current_builder_shares = builder_dist_t / builder_total if builder_total > 0 else np.zeros(n_regions)

            self.region_volatility_over_time.append(l1_distance(current_region_shares, prev_region_shares) if prev_region_shares is not None else 0.0)
            self.builder_dist_volatility_over_time.append(l1_distance(current_builder_shares, prev_builder_shares) if prev_builder_shares is not None else 0.0)
            self.value_share_volatility_over_time.append(l1_distance(value_shares, prev_value_shares) if prev_value_shares is not None else 0.0)

            prev_region_shares = current_region_shares
            prev_builder_shares = current_builder_shares
            prev_value_shares = value_shares

        self.value_capture_by_region = np.array(self.value_capture_by_region)
        self.value_share_distribution = np.array(self.value_share_distribution)

    def compute_average_volatility(self, metric_name: str, window: int = 2000) -> float:
        """
        Compute average volatility over the last 'window' slots.

        Args:
            metric_name: One of 'region', 'builder_dist', 'value_share'
            window: Number of slots to average over (default: 2000)

        Returns:
            Average volatility over the window
        """
        if metric_name == 'region':
            data = self.region_volatility_over_time
        elif metric_name == 'builder_dist':
            data = self.builder_dist_volatility_over_time
        elif metric_name == 'value_share':
            data = self.value_share_volatility_over_time
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

        if len(data) == 0:
            return 0.0

        # Take last 'window' points or all if fewer
        recent_data = data[-window:]
        return np.mean(recent_data)

    def compute_time_to_convergence(self, metric_series: np.ndarray,
                                    window: int = 500,
                                    epsilon: float = 0.01) -> int:
        """
        Compute time-to-convergence for a metric series.

        Convergence is defined as the first time n after which the metric stays
        within epsilon of its rolling mean for the next 'window' slots.

        T = min{n : |m(t) - mean(m[n:n+W])| <= epsilon for all t in [n, n+W]}

        Args:
            metric_series: Time series of metric values
            window: Rolling window size W (default: 500)
            epsilon: Convergence threshold (default: 0.01)

        Returns:
            Convergence time T, or -1 if never converges
        """
        n_slots = len(metric_series)

        if n_slots < window:
            return -1  # Not enough data

        # Check each potential convergence point
        for n in range(n_slots - window):
            # Compute rolling mean over window [n, n+W]
            window_data = metric_series[n:n+window]
            rolling_mean = np.mean(window_data)

            # Check if all points in window are within epsilon of mean
            deviations = np.abs(window_data - rolling_mean)
            if np.all(deviations <= epsilon):
                return n

        return -1  # Never converged

    def save(self, filepath: Optional[str] = None):
        """Save results to disk."""
        if filepath is None:
            results_dir = Path(self.config.results_dir)
            results_dir.mkdir(exist_ok=True)
            filepath = results_dir / f"{self.config.name}_results.npz"

        np.savez(
            filepath,
            region_counts=self.region_counts,
            builder_distribution=self.builder_distribution,
            rewards=np.array(self.rewards),
            welfare_history=self.welfare_history,
            tx_emitted_history=self.tx_emitted_history,
            tx_received_history=self.tx_received_history,
            builder_rewards_history=self.builder_rewards_history,
            location_gini_over_time=np.array(self.location_gini_over_time),
            location_entropy_over_time=np.array(self.location_entropy_over_time),
            location_hhi_over_time=np.array(self.location_hhi_over_time),
            utility_gini_over_time=np.array(self.utility_gini_over_time),
            utility_entropy_over_time=np.array(self.utility_entropy_over_time),
            utility_hhi_over_time=np.array(self.utility_hhi_over_time),
            value_capture_by_region=self.value_capture_by_region,
            value_share_distribution=self.value_share_distribution,
            value_share_hhi_over_time=np.array(self.value_share_hhi_over_time),
            value_share_entropy_over_time=np.array(self.value_share_entropy_over_time),
            value_share_top1_over_time=np.array(self.value_share_top1_over_time),
            value_share_top3_over_time=np.array(self.value_share_top3_over_time),
            region_volatility_over_time=np.array(self.region_volatility_over_time),
            builder_dist_volatility_over_time=np.array(self.builder_dist_volatility_over_time),
            value_share_volatility_over_time=np.array(self.value_share_volatility_over_time),
            config=np.array([asdict(self.config)], dtype=object),
            stats=np.array([self.stats], dtype=object)
        )

        print(f"Results saved to: {filepath}")
        return filepath

    @staticmethod
    def load(filepath: str) -> 'ExperimentResult':
        """Load results from disk."""
        data = np.load(filepath, allow_pickle=True)

        # Reconstruct config
        config_dict = data['config'].item()
        config = ExperimentConfig(**config_dict)

        result = object.__new__(ExperimentResult)
        result.config = config
        result.stats = data['stats'].item()
        result.region_counts = data['region_counts']
        result.builder_distribution = data['builder_distribution']
        result.rewards = list(data['rewards'])
        result.welfare_history = data.get('welfare_history', np.array([]))
        result.tx_emitted_history = data.get('tx_emitted_history', np.array([]))
        result.tx_received_history = data.get('tx_received_history', np.array([]))
        result.builder_rewards_history = data.get('builder_rewards_history', np.array([]))
        result.region_reward_pairs = []
        result.poa_stats = None

        # Location metrics (support old key names for backward compat)
        result.location_gini_over_time = list(
            data.get('location_gini_over_time', data.get('builder_dist_gini_over_time', []))
        )
        result.location_entropy_over_time = list(
            data.get('location_entropy_over_time', data.get('builder_dist_entropy_over_time', []))
        )
        result.location_hhi_over_time = list(
            data.get('location_hhi_over_time', data.get('builder_dist_hhi_over_time', []))
        )
        result.utility_gini_over_time = list(data.get('utility_gini_over_time', []))
        result.utility_entropy_over_time = list(data.get('utility_entropy_over_time', []))
        result.utility_hhi_over_time = list(data.get('utility_hhi_over_time', []))
        result.value_capture_by_region = data.get('value_capture_by_region', np.array([]))
        result.value_share_distribution = data.get('value_share_distribution', np.array([]))
        result.value_share_hhi_over_time = list(data.get('value_share_hhi_over_time', []))
        result.value_share_entropy_over_time = list(data.get('value_share_entropy_over_time', []))
        result.value_share_top1_over_time = list(data.get('value_share_top1_over_time', []))
        result.value_share_top3_over_time = list(data.get('value_share_top3_over_time', []))
        result.region_volatility_over_time = list(data.get('region_volatility_over_time', []))
        result.builder_dist_volatility_over_time = list(data.get('builder_dist_volatility_over_time', []))
        result.value_share_volatility_over_time = list(data.get('value_share_volatility_over_time', []))

        return result
