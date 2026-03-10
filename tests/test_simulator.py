import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sim.simulator import (
    Builder, EqualSplitSharingRule, EMASoftmaxPolicy,
    LatencyPropagationModel, LocationGamesSimulator,
    Region, Source, StochasticTransactionGenerator, Transaction,
)
from experiment_runner import ExperimentConfig, create_scenario_from_config


class TestEqualSplitSharing(unittest.TestCase):

    def setUp(self):
        self.rule = EqualSplitSharingRule()

    def test_two_builders_share_equally(self):
        rewards = self.rule.compute_rewards(
            tx_values={0: 2.0},
            tx_receivers={0: [0, 1]},
        )
        self.assertAlmostEqual(rewards[0], 1.0)
        self.assertAlmostEqual(rewards[1], 1.0)

    def test_solo_builder_keeps_full_value(self):
        rewards = self.rule.compute_rewards(
            tx_values={0: 3.0},
            tx_receivers={0: [5]},
        )
        self.assertAlmostEqual(rewards[5], 3.0)

    def test_multiple_transactions_accumulate(self):
        # builder 0 gets half of tx 0 (2.0) plus all of tx 1 (6.0)
        rewards = self.rule.compute_rewards(
            tx_values={0: 4.0, 1: 6.0},
            tx_receivers={0: [0, 1], 1: [0]},
        )
        self.assertAlmostEqual(rewards[0], 8.0)
        self.assertAlmostEqual(rewards[1], 2.0)


class TestPropagation(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)

    def test_colocated_vs_distant_builder(self):
        # Region 0 has near-zero latency, region 1 has impossibly high latency
        latency_mean = np.array([[0.01], [1000.0]])
        latency_std = np.array([[0.001], [1.0]])
        model = LatencyPropagationModel(latency_mean, latency_std)

        delta = 12.0
        n_trials = 500

        near_received = sum(
            model.receives(0, 0, Transaction(0, 0.0, 1.0), delta)
            for _ in range(n_trials)
        )
        far_received = sum(
            model.receives(1, 0, Transaction(0, 0.0, 1.0), delta)
            for _ in range(n_trials)
        )

        self.assertGreater(near_received, 490)
        self.assertEqual(far_received, 0)

    def test_receives_when_emission_time_is_early(self):
        # latency ~ 5s, emission_time=0, delta=12 -> arrives at ~5 <= 12
        latency_mean = np.array([[5.0]])
        latency_std = np.array([[0.0001]])
        model = LatencyPropagationModel(latency_mean, latency_std)

        results = [
            model.receives(0, 0, Transaction(0, 0.0, 1.0), 12.0)
            for _ in range(100)
        ]
        self.assertTrue(all(results))

    def test_does_not_receive_when_emission_time_is_too_late(self):
        # latency ~ 5s, emission_time=8, delta=12 -> arrives at ~13 > 12
        latency_mean = np.array([[5.0]])
        latency_std = np.array([[0.0001]])
        model = LatencyPropagationModel(latency_mean, latency_std)

        results = [
            model.receives(0, 0, Transaction(0, 8.0, 1.0), 12.0)
            for _ in range(100)
        ]
        self.assertFalse(any(results))


class TestWelfare(unittest.TestCase):

    def test_welfare_zero_when_no_receivers(self):
        np.random.seed(0)
        regions = [Region(0, "R0"), Region(1, "R1")]
        sources = [Source(0, "S0", 0, 10.0, 1.0, 0.5)]

        latency_mean = np.array([[1e6], [1e6]])
        latency_std = np.array([[1.0], [1.0]])
        model = LatencyPropagationModel(latency_mean, latency_std)

        builders = [Builder(i, EMASoftmaxPolicy(2)) for i in range(4)]
        sim = LocationGamesSimulator(
            regions=regions,
            sources=sources,
            builders=builders,
            tx_generator=StochasticTransactionGenerator(),
            propagation_model=model,
            sharing_rule=EqualSplitSharingRule(),
            delta=12.0,
            seed=0,
        )
        sim.run_round()

        self.assertEqual(sim.welfare_history[0], 0.0)
        self.assertEqual(sim.tx_received_history[0], 0)


class TestLatencySlicing(unittest.TestCase):

    def test_slices_correct_columns_for_non_sequential_source_regions(self):
        full_matrix = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ])
        config = ExperimentConfig(
            name="test",
            n_regions=3,
            region_names=["A", "B", "C"],
            sources_config=[("S", 2, 5.0, 1.0, 0.5)],
            latency_mean=full_matrix,
            latency_std=full_matrix * 0.1,
            save_results=False,
        )
        _, _, lat_mean, lat_std = create_scenario_from_config(config)

        self.assertEqual(lat_mean.shape, (3, 1))
        np.testing.assert_array_almost_equal(lat_mean[:, 0], [0.3, 0.6, 0.9])
        np.testing.assert_array_almost_equal(lat_std[:, 0], [0.03, 0.06, 0.09])


if __name__ == '__main__':
    unittest.main()
