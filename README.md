# Decentralized Building Simulator (db-sims)

Simulator for studying geographic location choice incentives in decentralized block building.

Agents (builders) can adapt their locations using `EMA`-Softmax, `UCB`, asynchronous strict better response (`ABR`), or multiplicative weights update (`MWU`). Rewards are shared among builders who cover a source using a configurable sharing rule (e.g., `equal_split`). The simulator tracks how the population distributes across regions over time and how efficiently sources are covered.

## Quick Start

```bash
pip install -r requirements.txt
python run.py configs/ema_baseline.yaml
python run.py configs/abr_baseline.yaml
python run.py configs/mwu_baseline.yaml
```

To run all configs in a directory:

```bash
python run.py configs/
```

## Usage

```
python run.py <config.yaml> [config2.yaml ...] | <configs_dir/> [--poa] [--poa-method {brute_force,greedy}]
```

### PoA Analysis

```bash
# Exact optimal welfare (feasible for small numbers of regions or builders)
python run.py configs/ema_baseline.yaml --poa

# Greedy approximation (faster for large numbers of regions or builders)
python run.py configs/ema_baseline.yaml --poa --poa-method greedy
```

Results and plots are saved to `results/`.

## Key Parameters

| Parameter | Description |
|---|---|
| `policy_type` | `"EMA"`, `"UCB"`, `"ABR"`, or `"MWU"` |
| `eta`, `beta_reg` | EMA learning rate and softmax temperature |
| `alpha` | UCB exploration bonus |
| `improvement_threshold_pct` | Relative improvement threshold for strict better response |
| `utility_eval_time_steps` | Deterministic integration grid size for exact ABR utility evaluation |
| `mwu_eta` | MWU learning rate |
| `payoff_normalization` | Optional MWU payoff scale before clipping to `[0, 1]` |
| `cost_c` | Migration cost |
| `n_builders` | Number of concurrent builders per slot |
| `n_slots` | Number of simulation slots |

## Dynamics

- `ABR`: each iteration picks one builder uniformly at random and computes exact expected payoffs for all regions under the current pure profile. The builder moves only if the best region improves utility by more than `improvement_threshold_pct` times its current utility. Ties stay put.
- `MWU`: each builder maintains a mixed strategy over regions, samples an action every round, then updates all region weights using full-information counterfactual realized payoffs from that same stochastic round.

## Metrics

- **Inequality**: Gini, entropy, HHI across regions/sources/population
- **Value-capture**: per-region share, top-1/top-3 concentration
- **Volatility**: L1 change in distributions between slots
- **Price of Anarchy**: `optimal_welfare / actual_welfare` (≥ 1; 1 = socially optimal). Optimal welfare assumes one builder per source.

All metrics are time series; pass any subset to `compare_experiments(results, metrics=[...])`.

## Project Structure

```
sim/
  simulator.py         — core simulator (policies, propagation model, tracking)
  config.py            — ExperimentConfig, load_config
  datasets.py          — GCP latency data loading
analysis/
  experiment_runner.py — runner, plots
  result.py            — ExperimentResult
  plotting.py          — comparison and detail plots
  poa.py               — Price of Anarchy computation
configs/               — YAML experiment configs
run.py                 — CLI entrypoint
```

## References

- Paper: [arXiv:2509.21475v2](https://arxiv.org/pdf/2509.21475v2)
- Original repo: [geographical-decentralization-simulation](https://github.com/syang-ng/geographical-decentralization-simulation)
