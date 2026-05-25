# Decentralized Block Building Simulator

Simulator for studying geographic location choice incentives in decentralized block building. Builders compete to include transactions from information sources distributed across regions. Rewards are split among all builders that include the same transaction. Builders choose regions selfishly to maximize their own capture, and the simulator measures the resulting inefficiencies via Price of Anarchy and inequality metrics.

## Quick Start

```bash
pip install -r requirements.txt
python run.py configs/ABR/sym_linear_equal.yaml
```

## Usage

```
python run.py <config.yaml> [config2.yaml ...] | <configs_dir/> [--poa] [--poa-method {brute_force,greedy}]
```

### Asynchronous Better Response (ABR)

The main dynamics used in this project are **Asynchronous Better Response (ABR)**. The location game is a potential game, so ABR is guaranteed to converge to a Nash equilibrium. In each round, one builder (round-robin) evaluates all regions analytically and migrates to the first that strictly improves its expected reward.

Set `policy_type: ABR` in the config to use ABR. Example:

```yaml
policy_type: ABR
n_builders: 5
n_slots: 500
delta: 0.05  # 50ms slot time
```

Run with PoA computation:

```bash
# Exact optimal (feasible for small numbers of regions or builders)
python run.py configs/ABR/sym_linear_equal.yaml --poa

# Greedy approximation (faster for large numbers of regions or builders)
python run.py configs/ABR/sym_linear_equal.yaml --poa --poa-method greedy
```

Results and plots are saved to `results/`.

### Learning Algorithms

The EXP3 bandit policy is also available via `policy_type: EXP3`. It is useful for studying out-of-equilibrium dynamics but does not have the same convergence guarantees as ABR.

## Experiments

The `scripts/` directory contains experiment scripts that sweep over key parameters and produce figures:

| Script | Description |
|---|---|
| `exp1_value_asymmetry.py` | Sweeps value ratio (high-value / peripheral information sources); measures welfare, welfare ratio, geographic HHI, utility HHI, and cluster coverage |
| `exp2_builder_count.py` | Sweeps builder count K; measures welfare ratio, geographic HHI, utility HHI, cluster coverage, and mean pairwise distance |
| `exp3_value_asymmetry_vs_slot_duration.py` | Heatmaps over (value ratio × delta); welfare ratio, geographic HHI, utility HHI, mean pairwise distance, and cluster coverage |
| `exp4_slot_duration.py` | Sweeps slot duration delta (10ms–12s) at fixed value ratio 10x; same metrics as exp1 |
| `exp5_builders_vs_slot_duration.py` | Heatmaps over (builder count K × delta) at fixed value ratio; same metrics as exp3 |

All scripts use ABR with GCP empirical latency data and randomized information source layouts, reporting median + IQR over multiple instances and seeds.

## Key Parameters

| Parameter | Description |
|---|---|
| `policy_type` | `"ABR"` or `"EXP3"` |
| `n_builders` | Number of builders |
| `n_slots` | Number of simulation slots |
| `delta` | Slot duration (seconds) |
| `eta` | EXP3 learning rate |
| `gamma`, `gamma_schedule` | EXP3 exploration rate (uniform mixture weight) and decay schedule |

## Metrics

**ABR experiment scripts (exp1–5):**
- **Price of Anarchy**: W* / W_ABR (≥ 1; 1 = socially optimal), where W* is the welfare-maximizing planner allocation and W_ABR is the welfare at the ABR-converged profile
- **Geographic / utility HHI**: builder concentration across regions and payoff inequality
- **Cluster coverage**: fraction of expected value captured from each information source cluster
- **Mean pairwise distance**: geographic spread of the converged builder placement

**Analysis pipeline (`run.py` + `analysis/`):**
- Gini, entropy, HHI of location and utility distributions (time series)
- Value-capture HHI, top-1/top-3 concentration
- Volatility: L1 change in region and value-share distributions between slots

## Project Structure

```
sim/
  simulator.py         - core simulator (ABR, learning policies, propagation, sharing rule)
  config.py            - ExperimentConfig, load_config
  datasets.py          - GCP latency data loading
analysis/
  experiment_runner.py - runner, plots
  result.py            - ExperimentResult
  plotting.py          - comparison and detail plots
  poa.py               - Price of Anarchy computation
scripts/               - experiment scripts (exp1–exp5, ...)
configs/               - YAML experiment configs
run.py                 - CLI entrypoint
```

## References

- Paper: [arXiv:2509.21475v2](https://arxiv.org/pdf/2509.21475v2)
- Original repo: [geographical-decentralization-simulation](https://github.com/syang-ng/geographical-decentralization-simulation)
