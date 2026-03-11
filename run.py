#!/usr/bin/env python3
"""
CLI entrypoint for running experiments

Usage:
    python run.py configs/  # run all YAML files in a directory
    python run.py configs/ema_baseline.yaml  # run a single config file
    python run.py configs/ema_baseline.yaml --poa  # also compute PoA (default: brute-force method)
    python run.py configs/ema_baseline.yaml --poa --poa-method greedy  # use greedy method for PoA
"""

import argparse
import sys
from pathlib import Path
from analysis.experiment_runner import run_experiment
from sim.config import load_config
from analysis.plotting import compare_experiments, print_comparison_table, plot_experiment_details, plot_network_setup


def main():
    parser = argparse.ArgumentParser(
        description="Run db-sims experiments",
        usage="python run.py <config.yaml> [config2.yaml ...] | <configs_dir/> [--poa] [--poa-method {brute_force,greedy}]",
    )
    parser.add_argument("configs", nargs="+", help="Config YAML files or directories")
    parser.add_argument("--poa", action="store_true", help="Compute Price of Anarchy stats (expensive)")
    parser.add_argument("--poa-method", choices=["brute_force", "greedy"], default="brute_force",
                        help="PoA computation method (default: brute_force)")
    args = parser.parse_args()

    paths = []
    for arg in args.configs:
        p = Path(arg)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.yaml")))
        elif p.is_file():
            paths.append(p)
        else:
            print(f"Error: {arg} is not a file or directory")
            sys.exit(1)

    if not paths:
        print("No config files found.")
        sys.exit(1)

    configs = [load_config(p) for p in paths]

    print(f"Dataset: {len(configs[0].region_names)} regions, "
          f"{len(configs[0].sources_config)} sources")
    print(f"Running {len(configs)} experiment(s) * {configs[0].n_slots} slots\n")

    plot_network_setup(configs[0], save_plots=True)

    results = []
    for config in configs:
        result = run_experiment(config, verbose=True, compute_poa=args.poa, poa_method=args.poa_method)
        results.append(result)

    print_comparison_table(results)

    if len(results) > 1:
        compare_experiments(
            results,
            metrics=["builder_dist_hhi", "builder_dist_entropy", "value_share_hhi",
                     "value_share_top1", "region_volatility", "reward", "welfare"],
            save_plots=True,
        )

    for result in results:
        plot_experiment_details(result, save_plots=True)

    print(f"\nResults saved to: results/")


if __name__ == "__main__":
    main()
