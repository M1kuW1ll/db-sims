import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.experiment_runner import ExperimentConfig
from sim.config import create_scenario_from_config
from sim.simulator import (
    Builder,
    EqualSplitSharingRule,
    FixedPolicy,
    LatencyPropagationModel,
    LocationGamesSimulator,
    Region,
    Source,
    StochasticTransactionGenerator,
    Transaction,
)


class SingleTransactionGenerator:
    def generate(self, source: Source, delta: float):
        del source, delta
        return [Transaction(0, 0.0, 1.0)]


class SequencePropagationModel:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.index = 0

    def receives(self, region_id: int, source_id: int, tx: Transaction, delta: float) -> bool:
        del region_id, source_id, tx, delta
        result = self.outcomes[self.index]
        self.index += 1
        return result


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
        latency_mean = np.array([[5.0]])
        latency_std = np.array([[0.0001]])
        model = LatencyPropagationModel(latency_mean, latency_std)

        results = [
            model.receives(0, 0, Transaction(0, 0.0, 1.0), 12.0)
            for _ in range(100)
        ]
        self.assertTrue(all(results))

    def test_does_not_receive_when_emission_time_is_too_late(self):
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

        builders = [Builder(i, FixedPolicy(2)) for i in range(4)]
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
        self.assertEqual(sim.tx_received_history[0], 0.0)


class TestIndependentBuilderReception(unittest.TestCase):
    def test_colocated_builders_are_sampled_independently(self):
        regions = [Region(0, "R0")]
        sources = [Source(0, "S0", 0, 1.0, 0.0, 0.0)]
        builders = [Builder(i, FixedPolicy(1)) for i in range(2)]

        sim = LocationGamesSimulator(
            regions=regions,
            sources=sources,
            builders=builders,
            tx_generator=SingleTransactionGenerator(),
            propagation_model=SequencePropagationModel([True, False]),
            sharing_rule=EqualSplitSharingRule(),
            delta=1.0,
            seed=0,
        )

        outcome = sim._simulate_round_for_profile({0: 0, 1: 0}, evaluate_all_regions=False)

        self.assertEqual(outcome.actual_tx_receivers, {0: [0]})
        self.assertEqual(outcome.tx_receiving_regions, {0: [0]})


class TestCCEDiagnostics(unittest.TestCase):
    def test_empirical_cce_gap_is_zero_when_no_deviation_exists(self):
        regions = [Region(0, "Only")]
        sources = [Source(0, "S0", 0, 1.0, 0.0, 0.01)]
        latency_mean = np.array([[0.01]])
        latency_std = np.array([[0.001]])
        builders = [Builder(0, FixedPolicy(1))]

        sim = LocationGamesSimulator(
            regions=regions,
            sources=sources,
            builders=builders,
            tx_generator=StochasticTransactionGenerator(),
            propagation_model=LatencyPropagationModel(latency_mean, latency_std),
            sharing_rule=EqualSplitSharingRule(),
            delta=1.0,
            seed=0,
        )
        sim.region_reward_pairs_history = [[(0, 0.0)], [(0, 0.0)]]

        cce_stats = sim.compute_empirical_cce_gap(n_time_steps=50)

        self.assertAlmostEqual(cce_stats["cce_gap"], 0.0)
        np.testing.assert_allclose(cce_stats["cce_gap_by_builder"], [0.0])
        np.testing.assert_array_equal(cce_stats["cce_best_deviation_regions"], [0])

    def test_empirical_cce_gap_is_positive_when_fixed_deviation_would_help(self):
        regions = [Region(0, "Fast"), Region(1, "Slow")]
        sources = [Source(0, "S0", 0, 2.0, 0.0, 0.01)]
        latency_mean = np.array([[0.01], [100.0]])
        latency_std = np.array([[0.001], [1.0]])
        builders = [Builder(0, FixedPolicy(2))]

        sim = LocationGamesSimulator(
            regions=regions,
            sources=sources,
            builders=builders,
            tx_generator=StochasticTransactionGenerator(),
            propagation_model=LatencyPropagationModel(latency_mean, latency_std),
            sharing_rule=EqualSplitSharingRule(),
            delta=1.0,
            seed=0,
        )
        sim.region_reward_pairs_history = [[(1, 0.0)], [(1, 0.0)], [(1, 0.0)]]

        cce_stats = sim.compute_empirical_cce_gap(n_time_steps=50)

        self.assertGreater(cce_stats["cce_gap"], 0.0)
        np.testing.assert_array_equal(cce_stats["cce_best_deviation_regions"], [0])
        self.assertEqual(len(cce_stats["cce_gap_over_time"]), 3)


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


class TestAsynchronousBetterResponse(unittest.TestCase):
    def _build_simple_exact_sim(self):
        regions = [Region(0, "Fast"), Region(1, "Slow")]
        sources = [Source(0, "S0", 0, 2.0, 0.0, 0.01)]
        latency_mean = np.array([[0.01], [100.0]])
        latency_std = np.array([[0.001], [1.0]])

        builders = [Builder(i, FixedPolicy(2)) for i in range(2)]
        sim = LocationGamesSimulator(
            regions=regions,
            sources=sources,
            builders=builders,
            tx_generator=StochasticTransactionGenerator(),
            propagation_model=LatencyPropagationModel(latency_mean, latency_std),
            sharing_rule=EqualSplitSharingRule(),
            delta=1.0,
            seed=0,
        )
        return sim

    def _build_three_region_exact_sim(self):
        regions = [Region(0, "R0"), Region(1, "R1"), Region(2, "R2")]
        sources = [Source(0, "S0", 0, 1.0, 0.0, 0.01)]
        latency_mean = np.array([[0.01], [0.01], [0.01]])
        latency_std = np.array([[0.001], [0.001], [0.001]])

        sim = LocationGamesSimulator(
            regions=regions,
            sources=sources,
            builders=[Builder(0, FixedPolicy(3))],
            tx_generator=StochasticTransactionGenerator(),
            propagation_model=LatencyPropagationModel(latency_mean, latency_std),
            sharing_rule=EqualSplitSharingRule(),
            delta=1.0,
            seed=0,
        )
        return sim

    def test_exact_utilities_favor_fast_region(self):
        sim = self._build_simple_exact_sim()
        sim.builders[0].set_region(0)
        sim.builders[1].set_region(1)

        utilities = sim.compute_expected_builder_utilities(n_time_steps=50)

        self.assertGreater(utilities[0], utilities[1])
        self.assertAlmostEqual(sim.compute_expected_welfare(n_time_steps=50), float(np.sum(utilities)))

    def test_async_better_response_moves_to_improving_region(self):
        sim = self._build_simple_exact_sim()
        for builder in sim.builders:
            builder.set_region(1)

        sim.run_async_better_response(
            n_slots=3,
            improvement_threshold_pct=0.001,
            n_time_steps=50,
            max_updates=10,
        )

        final_profile = [builder.current_region for builder in sim.builders]
        self.assertIn(0, final_profile)
        self.assertEqual(len(sim.welfare_history), 3)
        self.assertGreaterEqual(sim.tx_emitted_history[0], 0.0)
        ne_check = sim.verify_pure_nash_equilibrium(n_time_steps=50)
        self.assertTrue(ne_check["is_pure_ne"])
        self.assertTrue(sim.abr_converged)
        self.assertLessEqual(sim.abr_max_profitable_deviation, 0.001)

    def test_async_response_rule_best_uses_argmax_move(self):
        sim = self._build_three_region_exact_sim()

        def fake_candidate_utilities(builder_id: int, profile=None, n_time_steps: int = 200):
            del builder_id, n_time_steps
            current_region = profile[0]
            if current_region == 0:
                return np.array([1.0, 2.0, 3.0])
            if current_region == 1:
                return np.array([1.0, 2.0, 0.5])
            return np.array([1.0, 2.0, 3.0])

        sim.compute_candidate_utilities_for_builder = fake_candidate_utilities
        sim.builders[0].set_region(0)

        sim.run_async_better_response(
            n_slots=0,
            improvement_threshold_pct=0.0,
            n_time_steps=10,
            max_updates=5,
            response_rule="best",
        )

        self.assertEqual(sim.builders[0].current_region, 2)
        self.assertTrue(sim.abr_converged)

    def test_async_response_rule_better_can_choose_non_best_improving_move(self):
        sim = self._build_three_region_exact_sim()

        def fake_candidate_utilities(builder_id: int, profile=None, n_time_steps: int = 200):
            del builder_id, n_time_steps
            current_region = profile[0]
            if current_region == 0:
                return np.array([1.0, 2.0, 3.0])
            if current_region == 1:
                return np.array([1.0, 2.0, 0.5])
            return np.array([1.0, 2.0, 3.0])

        sim.compute_candidate_utilities_for_builder = fake_candidate_utilities
        sim.builders[0].set_region(0)
        np.random.seed(0)

        sim.run_async_better_response(
            n_slots=0,
            improvement_threshold_pct=0.0,
            n_time_steps=10,
            max_updates=5,
            response_rule="better",
        )

        self.assertEqual(sim.builders[0].current_region, 1)
        self.assertTrue(sim.abr_converged)

if __name__ == "__main__":
    unittest.main()
