# Decentralized Block Building Simulator

Simulator for studying geographic location choice incentives in decentralized block building. Builders compete to include transactions from information sources distributed across regions, split rewards with other builders that include the same transaction, and choose regions selfishly to maximize capture. The simulator supports `EMA`-Softmax, `UCB`, `EXP3`, asynchronous strict better response (`ABR`), and multiplicative weights update (`MWU`), and reports Price of Anarchy, coverage, concentration, and inequality metrics.

## Quick Start

```bash
pip install -r requirements.txt
python run.py configs/ema_baseline.yaml
python run.py configs/exp3_baseline.yaml
python run.py configs/abr_baseline.yaml
python run.py configs/mwu_baseline.yaml
python run.py configs/ABR/sym_linear_equal.yaml
```

To run all configs in a directory:

```bash
python run.py configs/
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

There are also stochastic learning policies (EXP3, EMA-Softmax, UCB) available via `policy_type`. These are useful for studying out-of-equilibrium dynamics that may not have the same convergence guarantees.

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
| `policy_type` | `"EMA"`, `"UCB"`, `"EXP3"`, `"ABR"`, or `"MWU"` |
| `n_builders` | Number of builders |
| `n_slots` | Number of simulation slots |
| `delta` | Slot duration (seconds) |
| `eta`, `beta_reg` | EMA learning rate and softmax temperature |
| `alpha` | UCB exploration bonus |
| `gamma` / `exp3_gamma` | EXP3 exploration parameter |
| `gamma_schedule` | EXP3 exploration schedule (`static`, `exponential`, `sqrt_decay`, or `linear`) |
| `abr_response_rule` | ABR move rule: `"best"` for argmax best response, `"better"` for a uniformly random improving move |
| `improvement_threshold_pct` | Relative improvement threshold for ABR moves (use `0.0` for exact pure-NE search) |
| `utility_eval_time_steps` | Deterministic integration grid size for exact ABR utility evaluation |
| `abr_max_updates` | Maximum ABR unilateral moves during adaptation before frozen evaluation |
| `mwu_eta` | MWU learning rate |
| `payoff_normalization` | Optional EXP3/MWU payoff scale before clipping to `[0, 1]` |
| `cost_c` | Migration cost |

## Dynamics

- `EXP3`: each builder maintains a mixed strategy over regions, samples one region per round, and applies an importance-weighted update using only the realized reward of the sampled region.
- `ABR`: first runs a pure-profile adaptation phase under exact expected builder utility, updating builders asynchronously with either argmax best responses (`abr_response_rule: "best"`, the default) or uniformly random better responses (`abr_response_rule: "better"`), until no improving move remains or `abr_max_updates` is reached. It then freezes the resulting profile and evaluates it over stochastic rounds.
- `MWU`: each builder maintains a mixed strategy over regions, samples an action every round, then updates all region weights using full-information counterfactual realized payoffs from that same stochastic round.

## Metrics

**ABR experiment scripts (exp1–5):**
- **Price of Anarchy**: W* / W_ABR (≥ 1; 1 = socially optimal), where W* is the welfare-maximizing planner allocation and W_ABR is the welfare at the ABR-converged profile
- **Geographic / utility HHI**: builder concentration across regions and payoff inequality
- **Cluster coverage**: fraction of expected value captured from each information source cluster
- **Mean pairwise distance**: geographic spread of the converged builder placement

**Analysis pipeline (`run.py` + `analysis/`), used with learning algorithms:**
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
