from typing import List

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from analysis.result import ExperimentResult

def compare_experiments(results: List[ExperimentResult],
                        metrics: List[str] = None,
                        save_plots: bool = True):
    """
    Compare multiple experiments by plotting metrics over time.

    Args:
        results: List of ExperimentResult objects to compare.
        metrics: List of metrics to plot. Options:
                 Location concentration (over builder counts per region):
                   'location_gini', 'location_entropy', 'location_hhi'
                 Utility concentration (over per-builder rewards):
                   'utility_gini', 'utility_entropy', 'utility_hhi'
                 Value-capture:
                   'value_share_hhi', 'value_share_entropy'
                   'value_share_top1', 'value_share_top3'
                 Volatility:
                   'region_volatility', 'builder_dist_volatility', 'value_share_volatility'
                 Other:
                   'reward', 'welfare'
                 If None, plots the default set
        save_plots: Whether to save plots to disk
    """
    if metrics is None:
        metrics = [
            'location_hhi', 'utility_hhi',
            'location_entropy', 'utility_entropy',
            'value_share_hhi', 'region_volatility',
            'reward', 'welfare',
        ]

    n_metrics = len(metrics)
    n_rows = 3
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15))
    axes = axes.flatten()

    exp_names = ", ".join([r.config.name for r in results])
    if len(exp_names) > 80:
        exp_names = ", ".join([r.config.name for r in results[:3]])
        if len(results) > 3:
            exp_names += f", +{len(results)-3} more"
    fig.suptitle(f'Experiment Comparison: {exp_names}', fontsize=14, fontweight='bold')

    # Color scheme and line styles for better distinction
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    line_styles = ['-', '--', '-.', ':']  # Solid, dashed, dash-dot, dotted

    for metric_idx, metric in enumerate(metrics):
        if metric_idx >= len(axes):
            print(f"Warning: More metrics ({n_metrics}) than available subplots ({len(axes)})")
            break
        ax = axes[metric_idx]

        for result_idx, result in enumerate(results):
            metric_map = {
                'location_gini': (result.location_gini_over_time, 'Location Gini'),
                'location_entropy': (result.location_entropy_over_time, 'Location Entropy'),
                'location_hhi': (result.location_hhi_over_time, 'Location HHI'),
                'utility_gini': (result.utility_gini_over_time, 'Utility Gini'),
                'utility_entropy': (result.utility_entropy_over_time, 'Utility Entropy'),
                'utility_hhi': (result.utility_hhi_over_time, 'Utility HHI'),
                'value_share_hhi': (result.value_share_hhi_over_time, 'Value-Capture HHI'),
                'value_share_entropy': (result.value_share_entropy_over_time, 'Value-Capture Entropy'),
                'value_share_top1': (result.value_share_top1_over_time, 'Value-Capture Top-1 Concentration'),
                'value_share_top3': (result.value_share_top3_over_time, 'Value-Capture Top-3 Concentration'),
                'region_volatility': (result.region_volatility_over_time, 'Region Selection Volatility (L1 change)'),
                'builder_dist_volatility': (result.builder_dist_volatility_over_time, 'Population Distribution Volatility (L1 change)'),
                'value_share_volatility': (result.value_share_volatility_over_time, 'Value-Share Volatility (L1 change)'),
                'reward': (result.rewards, 'Average Reward per Builder'),
                'welfare': (list(result.welfare_history), 'Total Welfare per Slot'),
            }

            if metric not in metric_map:
                print(f"Unknown metric: {metric}")
                continue

            data, ylabel = metric_map[metric]

            # Smooth with moving average (NaN-safe for PoA)
            window = min(100, len(data) // 10)
            if window > 1:
                smoothed = np.convolve(np.nan_to_num(data, nan=1.0),
                                       np.ones(window) / window, mode='valid')
                slots = np.arange(window - 1, len(data))
            else:
                smoothed = data
                slots = np.arange(len(data))

            # Plot with distinct line style
            label = f"{result.config.name} ({result.config.policy_type})"
            linestyle = line_styles[result_idx % len(line_styles)]
            ax.plot(slots, smoothed, label=label, linewidth=2.5,
                    color=colors[result_idx], linestyle=linestyle, alpha=0.9)

        ax.set_xlabel('Slot', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{metric.replace("_", " ").title()} Over Time', fontsize=12, fontweight='bold')
        ax.legend(loc='best', framealpha=0.95, fontsize=10, edgecolor='black', fancybox=True)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave 3% space at top for suptitle

    if save_plots:
        # Create descriptive filename with experiment names
        exp_names = "_vs_".join([r.config.name[:15] for r in results[:3]])  # Limit to first 3 names
        if len(results) > 3:
            exp_names += f"_and_{len(results)-3}_more"
        results_dir = Path(results[0].config.results_dir)
        results_dir.mkdir(exist_ok=True)
        filename = results_dir / f'comparison_{exp_names}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\nComparison plot saved to: {filename}")

    plt.close(fig)


def plot_experiment_details(result: ExperimentResult, save_plots: bool = True):
    """
    Plot detailed time-series for a single experiment.

    Each concentration metric (Gini, Entropy, HHI) shows two lines:
      - Location (deterministic, based on builder counts per region)
      - Utility (stochastic, based on observed per-builder rewards)
    """
    print(f"\n[DEBUG] Plotting details for experiment: {result.config.name}")

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 18))

    # Define grid: 4 rows, 3 columns (row 3 reserved for welfare spanning full width)
    gs = fig.add_gridspec(4, 3, hspace=0.55, wspace=0.3)

    slots = np.arange(len(result.region_counts))

    # Continent aggregation helper
    _continent_map = {
        'africa': 'Africa', 'asia': 'Asia', 'australia': 'Oceania',
        'europe': 'Europe', 'us': 'North America',
        'northamerica': 'North America', 'southamerica': 'South America',
        'me': 'Middle East',
    }
    def _continent(name):
        return _continent_map.get(name.split('-')[0], name.split('-')[0].title())

    continent_names = sorted(set(_continent(r) for r in result.config.region_names))
    continent_indices = {c: [i for i, r in enumerate(result.config.region_names) if _continent(r) == c]
                         for c in continent_names}
    continent_counts = np.array([
        result.region_counts[:, continent_indices[c]].sum(axis=1)
        for c in continent_names
    ]).T  # (n_slots, n_continents)
    colors_continents = plt.cm.tab10(np.linspace(0, 1, len(continent_names)))

    # Row 1
    # 1. Region selection stacked area
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.stackplot(slots, *continent_counts.T,
                  labels=continent_names, colors=colors_continents, alpha=0.7)
    ax1.set_xlabel('Slot', fontsize=10)
    ax1.set_ylabel('Number of builders', fontsize=10)
    ax1.set_title('Region Selection Per Slot', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=5, fontsize=7, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # 2. Region occupancy heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    region_counts_matrix = np.array(result.region_counts)
    active_mask = region_counts_matrix.max(axis=0) > 0
    active_indices = np.where(active_mask)[0]
    active_matrix = region_counts_matrix[:, active_indices].T
    region_labels = ([result.config.region_names[i] for i in active_indices]
                     if result.config.region_names else [str(i) for i in active_indices])
    im = ax2.imshow(active_matrix, aspect='auto', interpolation='nearest', cmap='YlOrRd',
                    extent=[0, len(slots), len(active_indices), 0])
    ax2.set_yticks(np.arange(len(active_indices)) + 0.5)
    ax2.set_yticklabels(region_labels, fontsize=6)
    ax2.set_xlabel('Slot', fontsize=10)
    ax2.set_title('Region Occupancy Heatmap', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax2, label='Builders', shrink=0.8)

    # 3. Transactions per round + coverage
    ax3 = fig.add_subplot(gs[0, 2])
    slots_txs = np.arange(len(result.tx_emitted_history))
    ax3.plot(slots_txs, result.tx_emitted_history, linewidth=2, color='lightgray', alpha=0.9, label='Emitted')
    ax3.plot(slots_txs, result.tx_received_history, linewidth=2, color='steelblue', alpha=0.8, label='Received')
    ax3.set_xlabel('Slot', fontsize=10)
    ax3.set_ylabel('Transactions', fontsize=10)
    ax3.set_title('Transactions per Round', fontsize=11, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax3_cov = ax3.twinx()
    coverage_series = np.where(result.tx_emitted_history > 0,
                               result.tx_received_history / result.tx_emitted_history, 0.0)
    ax3_cov.plot(slots_txs, coverage_series, linewidth=1.5, color='green', alpha=0.7,
                 linestyle='--', label='Coverage')
    ax3_cov.set_ylabel('Coverage ratio', fontsize=9, color='green')
    ax3_cov.set_ylim(0, 1)
    ax3_cov.tick_params(axis='y', labelcolor='green')
    ax3_cov.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax3.grid(True, alpha=0.3)

    # Row 2
    # 4. Gini: location + utility on same panel
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(slots, result.location_gini_over_time,
             label='Location', linewidth=2, alpha=0.9, color='steelblue')
    ax4.plot(slots, result.utility_gini_over_time,
             label='Utility', linewidth=1.5, alpha=0.7, color='darkorange', linestyle='--')
    ax4.set_xlabel('Slot', fontsize=10)
    ax4.set_ylabel('Gini Coefficient', fontsize=10)
    ax4.set_title('Gini (Inequality) Over Time', fontsize=11, fontweight='bold')
    ax4.legend(loc='best', fontsize=9, framealpha=0.9)
    ax4.grid(True, alpha=0.3)

    # 5. Entropy: location + utility on same panel
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(slots, result.location_entropy_over_time,
             label='Location', linewidth=2, alpha=0.9, color='steelblue')
    ax5.plot(slots, result.utility_entropy_over_time,
             label='Utility', linewidth=1.5, alpha=0.7, color='darkorange', linestyle='--')
    ax5.set_xlabel('Slot', fontsize=10)
    ax5.set_ylabel('Normalised Entropy', fontsize=10)
    ax5.set_title('Entropy (Diversity) Over Time', fontsize=11, fontweight='bold')
    ax5.legend(loc='best', fontsize=9, framealpha=0.9)
    ax5.grid(True, alpha=0.3)

    # 6. Average reward
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(slots, result.rewards, linewidth=2, color='darkgreen', alpha=0.8)
    ax6.set_xlabel('Slot', fontsize=10)
    ax6.set_ylabel('Average Reward', fontsize=10)
    ax6.set_title('Average Reward Per Builder', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # Row 3
    # 7. HHI: location + utility on same panel
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(slots, result.location_hhi_over_time,
             label='Location', linewidth=2, alpha=0.9, color='steelblue')
    ax7.plot(slots, result.utility_hhi_over_time,
             label='Utility', linewidth=1.5, alpha=0.7, color='darkorange', linestyle='--')
    ax7.set_xlabel('Slot', fontsize=10)
    ax7.set_ylabel('HHI', fontsize=10)
    ax7.set_title('HHI (Concentration) Over Time', fontsize=11, fontweight='bold')
    ax7.legend(loc='best', fontsize=9, framealpha=0.9)
    ax7.grid(True, alpha=0.3)

    # 8. Value-Capture Concentration
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(slots, result.value_share_hhi_over_time,  label='HHI',   linewidth=2, alpha=0.8, color='darkred')
    ax8.plot(slots, result.value_share_top1_over_time, label='Top-1', linewidth=2, alpha=0.8, color='orange')
    ax8.plot(slots, result.value_share_top3_over_time, label='Top-3', linewidth=2, alpha=0.8, color='gold')
    ax8.set_xlabel('Slot', fontsize=10)
    ax8.set_ylabel('Concentration', fontsize=10)
    ax8.set_title('Value-Capture Concentration', fontsize=11, fontweight='bold')
    ax8.legend(loc='best', fontsize=9, framealpha=0.9)
    ax8.grid(True, alpha=0.3)

    # 9. Volatility
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(slots, result.builder_dist_volatility_over_time,
             label='Location Distribution', linewidth=2, alpha=0.8)
    ax9.plot(slots, result.value_share_volatility_over_time,
             label='Value-Capture', linewidth=2, alpha=0.8)
    ax9.set_xlabel('Slot', fontsize=10)
    ax9.set_ylabel('L1 Change', fontsize=10)
    ax9.set_title('Volatility (Distribution Churn) Over Time', fontsize=11, fontweight='bold')
    ax9.legend(loc='best', fontsize=9, framealpha=0.9)
    ax9.grid(True, alpha=0.3)

    # Row 4
    # 10. Welfare (full-width)
    ax10 = fig.add_subplot(gs[3, :])
    slots_welfare = np.arange(len(result.welfare_history))
    ax10.plot(slots_welfare, result.welfare_history, linewidth=2, color='steelblue', alpha=0.9, label='Welfare')
    mean_w = result.stats['mean_welfare']
    ax10.axhline(y=mean_w, color='green', linestyle='--', linewidth=1.5, alpha=0.8,
                 label=f'Mean welfare = {mean_w:.2f}')
    ax10.set_xlabel('Slot', fontsize=10)
    ax10.set_ylabel('Total Welfare', fontsize=10)
    ax10.set_title(
        f'Total Welfare Per Slot  |  Mean = {mean_w:.2f}  |  '
        f'Mean txs/round = {result.stats["mean_txs_emitted_per_round"]:.1f}  |  '
        f'Coverage = {result.stats["mean_coverage_ratio"]:.2%}',
        fontsize=11, fontweight='bold'
    )
    ax10.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax10.grid(True, alpha=0.3)

    # Main title
    policy_info = result.config.policy_type

    fig.suptitle(f'Experiment: {result.config.name} | {policy_info}',
                 fontsize=13, fontweight='bold')

    if save_plots:
        results_dir = Path(result.config.results_dir)
        results_dir.mkdir(exist_ok=True)
        filename = results_dir / f"{result.config.name}_details.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Detail plot saved to: {filename}")

    plt.close(fig)


def print_comparison_table(results: List[ExperimentResult]):
    """Print a comparison table of final metrics across experiments."""
    col_w = 12
    header_cols = ['Experiment', 'Policy', 'Reward', 'Welfare',
                   'Loc.Gini', 'Loc.Entr', 'Loc.HHI',
                   'Util.Gini', 'Util.Entr', 'Util.HHI',
                   'Coverage']
    widths = [25, 8] + [col_w] * (len(header_cols) - 2)
    total_width = sum(widths) + len(widths) - 1

    print(f"\n{'='*total_width}")
    print("EXPERIMENT COMPARISON TABLE")
    print(f"{'='*total_width}")
    header = (f"{'Experiment':<25} {'Policy':<8} "
              f"{'Reward':<{col_w}} {'Welfare':<{col_w}} "
              f"{'Loc.Gini':<{col_w}} {'Loc.Entr':<{col_w}} {'Loc.HHI':<{col_w}} "
              f"{'Util.Gini':<{col_w}} {'Util.Entr':<{col_w}} {'Util.HHI':<{col_w}} "
              f"{'Coverage':<{col_w}}")
    print(header)
    print("-" * total_width)

    for result in results:
        s = result.stats
        print(
            f"{result.config.name:<25} {result.config.policy_type:<8} "
            f"{s['avg_reward']:<{col_w}.4f} {s['mean_welfare']:<{col_w}.4f} "
            f"{s['location_gini']:<{col_w}.4f} {s['location_entropy']:<{col_w}.4f} {s['location_hhi']:<{col_w}.4f} "
            f"{s['utility_gini']:<{col_w}.4f} {s['utility_entropy']:<{col_w}.4f} {s['utility_hhi']:<{col_w}.4f} "
            f"{s['mean_coverage_ratio']:<{col_w}.4f}"
        )

    print("=" * total_width)
