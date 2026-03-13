from typing import List

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sim.config import ExperimentConfig
from analysis.result import ExperimentResult

def compare_experiments(results: List[ExperimentResult],
                       metrics: List[str] = None,
                       save_plots: bool = True):
    """
    Compare multiple experiments by plotting metrics over time.

    Args:
        results: List of ExperimentResult objects to compare
        metrics: List of metrics to plot. Options:
                 - 'region_gini', 'region_entropy', 'region_hhi'
                 - 'source_gini', 'source_entropy', 'source_hhi'
                 - 'builder_dist_gini', 'builder_dist_entropy', 'builder_dist_hhi'
                 - 'value_share_hhi', 'value_share_entropy'
                 - 'value_share_top1', 'value_share_top3'
                 - 'region_volatility', 'builder_dist_volatility', 'value_share_volatility'
                 - 'reward'
                 If None, plots default set of metrics
        save_plots: Whether to save plots to disk
    """
    if metrics is None:
        metrics = ['builder_dist_gini', 'builder_dist_entropy', 'builder_dist_hhi',
                   'region_volatility', 'value_share_hhi', 'reward', 'welfare']

    print(f"\n[DEBUG] compare_experiments called with {len(results)} results:")
    for i, r in enumerate(results):
        print(f"  [{i}] {r.config.name} ({r.config.policy_type})")

    n_metrics = len(metrics)
    # Use 3x3 grid layout (supports up to 9 metrics; extras are hidden)
    n_rows = 3
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15))
    axes = axes.flatten()  # Flatten to 1D array for easier indexing

    # Create descriptive title with experiment names
    exp_names = ", ".join([r.config.name for r in results])
    if len(exp_names) > 80:
        exp_names = ", ".join([r.config.name for r in results[:3]])
        if len(results) > 3:
            exp_names += f", +{len(results)-3} more"
    fig.suptitle(f'Experiment Comparison: {exp_names}', fontsize=14, fontweight='bold')

    # Color scheme and line styles for better distinction
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    line_styles = ['-', '--', '-.', ':']  # Solid, dashed, dash-dot, dotted
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']  # Different markers

    for metric_idx, metric in enumerate(metrics):
        if metric_idx >= len(axes):
            print(f"Warning: More metrics ({n_metrics}) than available subplots ({len(axes)})")
            break
        ax = axes[metric_idx]

        for result_idx, result in enumerate(results):
            metric_map = {
                'region_gini': (result.region_gini_over_time, 'Region Gini (lower = more diverse)'),
                'region_entropy': (result.region_entropy_over_time, 'Region Entropy (higher = more diverse)'),
                'region_hhi': (result.region_hhi_over_time, 'Region HHI (lower = more diverse)'),
                'builder_dist_gini': (result.builder_dist_gini_over_time, 'Builder Distribution Gini (lower = more equal)'),
                'builder_dist_entropy': (result.builder_dist_entropy_over_time, 'Builder Distribution Entropy (higher = more equal)'),
                'builder_dist_hhi': (result.builder_dist_hhi_over_time, 'Builder Distribution HHI (lower = more equal)'),
                'value_share_hhi': (result.value_share_hhi_over_time, 'Value-Capture HHI (lower = more equal)'),
                'value_share_entropy': (result.value_share_entropy_over_time, 'Value-Capture Entropy (higher = more equal)'),
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
            print(f"    Plotting {result.config.name}: {len(smoothed)} points, color idx {result_idx}, style {linestyle}")
            ax.plot(slots, smoothed, label=label, linewidth=2.5,
                   color=colors[result_idx], linestyle=linestyle, alpha=0.9)

        ax.set_xlabel('Slot', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{metric.replace("_", " ").title()} Over Time', fontsize=12, fontweight='bold')
        ax.legend(loc='best', framealpha=0.95, fontsize=10, edgecolor='black', fancybox=True)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        print(f"  [DEBUG] Plotted {len(results)} lines for metric '{metric}'")

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave 3% space at top for suptitle

    if save_plots:
        # Create descriptive filename with experiment names
        exp_names = "_vs_".join([r.config.name[:15] for r in results[:3]])  # Limit to first 3 names
        if len(results) > 3:
            exp_names += f"_and_{len(results)-3}_more"
        results_dir = Path(result.config.results_dir)
        results_dir.mkdir(exist_ok=True)
        filename = results_dir / f'comparison_{exp_names}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\nComparison plot saved to: {filename}")

    plt.close(fig)


def plot_experiment_details(result: ExperimentResult, save_plots: bool = True):
    """
    Plot detailed time-series for a single experiment.

    Creates separate plots for:
    - Region selection over time (stacked area or line plot)
    - Source selection over time
    - builder distribution over time
    - Diversity metrics (Gini, Entropy)
    - Rewards

    Args:
        result: ExperimentResult object
        save_plots: Whether to save plots to disk
    """
    print(f"\n[DEBUG] Plotting details for experiment: {result.config.name}")

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 18))

    # Define grid: 4 rows, 3 columns (row 3 reserved for welfare spanning full width)
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

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

    # Row 1: Distributions
    # 1. Region Selection Over Time (stacked area, aggregated by continent)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.stackplot(slots, *continent_counts.T,
                  labels=continent_names,
                  colors=colors_continents, alpha=0.7)
    ax1.set_xlabel('Slot', fontsize=10)
    ax1.set_ylabel('Number of builders', fontsize=10)
    ax1.set_title('Region Selection Per Slot', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # 2. Transactions per Round + Coverage Ratio
    ax2 = fig.add_subplot(gs[0, 1])
    window = min(100, len(result.tx_emitted_history) // 10) if len(result.tx_emitted_history) > 10 else 1
    if window > 1:
        smooth_emitted = np.convolve(result.tx_emitted_history, np.ones(window) / window, mode='valid')
        smooth_received = np.convolve(result.tx_received_history, np.ones(window) / window, mode='valid')
        slots_txs = np.arange(window - 1, len(result.tx_emitted_history))
    else:
        smooth_emitted = result.tx_emitted_history
        smooth_received = result.tx_received_history
        slots_txs = np.arange(len(result.tx_emitted_history))

    ax2.plot(slots_txs, smooth_emitted, linewidth=2, color='lightgray', alpha=0.9, label='Emitted')
    ax2.plot(slots_txs, smooth_received, linewidth=2, color='steelblue', alpha=0.8, label='Received')
    ax2.set_xlabel('Slot', fontsize=10)
    ax2.set_ylabel('Transactions', fontsize=10)
    ax2.set_title('Transactions per Round', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)

    ax2_cov = ax2.twinx()
    coverage_series = np.where(result.tx_emitted_history > 0,
                               result.tx_received_history / result.tx_emitted_history, 0.0)
    if window > 1:
        smooth_cov = np.convolve(coverage_series, np.ones(window) / window, mode='valid')
    else:
        smooth_cov = coverage_series
    ax2_cov.plot(slots_txs, smooth_cov, linewidth=1.5, color='green', alpha=0.7, linestyle='--', label='Coverage')
    ax2_cov.set_ylabel('Coverage ratio', fontsize=9, color='green')
    ax2_cov.set_ylim(0, 1)
    ax2_cov.tick_params(axis='y', labelcolor='green')
    ax2_cov.legend(loc='lower right', fontsize=8, framealpha=0.9)

    ax2.grid(True, alpha=0.3)

    # Row 2: Traditional Metrics
    # 4. Diversity Metrics - Gini
    ax4 = fig.add_subplot(gs[1, 0])

    window = min(100, len(result.builder_dist_gini_over_time) // 10)
    if window > 1:
        smooth_prop_gini = np.convolve(result.builder_dist_gini_over_time,
                                       np.ones(window)/window, mode='valid')
        smooth_region_gini = np.convolve(result.region_gini_over_time,
                                         np.ones(window)/window, mode='valid')
        slots_smooth = np.arange(window-1, len(slots))
    else:
        smooth_prop_gini = result.builder_dist_gini_over_time
        smooth_region_gini = result.region_gini_over_time
        slots_smooth = slots

    ax4.plot(slots_smooth, smooth_prop_gini, label='Builder Distribution', linewidth=2, alpha=0.8)
    ax4.plot(slots_smooth, smooth_region_gini, label='Region Selection', linewidth=2, alpha=0.8)
    ax4.set_xlabel('Slot', fontsize=10)
    ax4.set_ylabel('Gini Coefficient', fontsize=10)
    ax4.set_title('Gini (Inequality) Over Time', fontsize=11, fontweight='bold')
    ax4.legend(loc='best', fontsize=9, framealpha=0.9)
    ax4.grid(True, alpha=0.3)

    # 5. Diversity Metrics - Entropy
    ax5 = fig.add_subplot(gs[1, 1])

    if window > 1:
        smooth_prop_ent = np.convolve(result.builder_dist_entropy_over_time,
                                      np.ones(window)/window, mode='valid')
        smooth_region_ent = np.convolve(result.region_entropy_over_time,
                                        np.ones(window)/window, mode='valid')
    else:
        smooth_prop_ent = result.builder_dist_entropy_over_time
        smooth_region_ent = result.region_entropy_over_time

    ax5.plot(slots_smooth, smooth_prop_ent, label='Builder Distribution', linewidth=2, alpha=0.8)
    ax5.plot(slots_smooth, smooth_region_ent, label='Region Selection', linewidth=2, alpha=0.8)
    ax5.set_xlabel('Slot', fontsize=10)
    ax5.set_ylabel('Normalized Entropy', fontsize=10)
    ax5.set_title('Entropy (Diversity) Over Time', fontsize=11, fontweight='bold')
    ax5.legend(loc='best', fontsize=9, framealpha=0.9)
    ax5.grid(True, alpha=0.3)

    # 6. Average Reward Over Time
    ax6 = fig.add_subplot(gs[1, 2])

    if window > 1:
        smooth_rewards = np.convolve(result.rewards, np.ones(window)/window, mode='valid')
    else:
        smooth_rewards = result.rewards

    ax6.plot(slots_smooth, smooth_rewards, linewidth=2, color='darkgreen', alpha=0.8)
    ax6.set_xlabel('Slot', fontsize=10)
    ax6.set_ylabel('Average Reward', fontsize=10)
    ax6.set_title('Average Reward Per Builder', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # Row 3: New Metrics
    # 7. HHI Metrics Over Time
    ax7 = fig.add_subplot(gs[2, 0])

    if window > 1:
        smooth_region_hhi = np.convolve(result.region_hhi_over_time,
                                        np.ones(window)/window, mode='valid')
        smooth_prop_hhi = np.convolve(result.builder_dist_hhi_over_time,
                                      np.ones(window)/window, mode='valid')
    else:
        smooth_region_hhi = result.region_hhi_over_time
        smooth_prop_hhi = result.builder_dist_hhi_over_time

    ax7.plot(slots_smooth, smooth_region_hhi, label='Region Selection', linewidth=2, alpha=0.8)
    ax7.plot(slots_smooth, smooth_prop_hhi, label='Population Distribution', linewidth=2, alpha=0.8)
    ax7.set_xlabel('Slot', fontsize=10)
    ax7.set_ylabel('HHI', fontsize=10)
    ax7.set_title('HHI (Concentration) Over Time', fontsize=11, fontweight='bold')
    ax7.legend(loc='best', fontsize=9, framealpha=0.9)
    ax7.grid(True, alpha=0.3)

    # 8. Value-Capture Concentration
    ax8 = fig.add_subplot(gs[2, 1])

    if window > 1:
        smooth_value_hhi = np.convolve(result.value_share_hhi_over_time,
                                       np.ones(window)/window, mode='valid')
        smooth_value_top1 = np.convolve(result.value_share_top1_over_time,
                                        np.ones(window)/window, mode='valid')
        smooth_value_top3 = np.convolve(result.value_share_top3_over_time,
                                        np.ones(window)/window, mode='valid')
    else:
        smooth_value_hhi = result.value_share_hhi_over_time
        smooth_value_top1 = result.value_share_top1_over_time
        smooth_value_top3 = result.value_share_top3_over_time

    ax8.plot(slots_smooth, smooth_value_hhi, label='HHI', linewidth=2, alpha=0.8, color='darkred')
    ax8.plot(slots_smooth, smooth_value_top1, label='Top-1', linewidth=2, alpha=0.8, color='orange')
    ax8.plot(slots_smooth, smooth_value_top3, label='Top-3', linewidth=2, alpha=0.8, color='gold')
    ax8.set_xlabel('Slot', fontsize=10)
    ax8.set_ylabel('Concentration', fontsize=10)
    ax8.set_title('Value-Capture Concentration', fontsize=11, fontweight='bold')
    ax8.legend(loc='best', fontsize=9, framealpha=0.9)
    ax8.grid(True, alpha=0.3)

    # 9. Volatility (L1 Change) Metrics
    ax9 = fig.add_subplot(gs[2, 2])

    if window > 1:
        smooth_region_vol = np.convolve(result.region_volatility_over_time,
                                        np.ones(window)/window, mode='valid')
        smooth_prop_vol = np.convolve(result.builder_dist_volatility_over_time,
                                      np.ones(window)/window, mode='valid')
        smooth_value_vol = np.convolve(result.value_share_volatility_over_time,
                                       np.ones(window)/window, mode='valid')
    else:
        smooth_region_vol = result.region_volatility_over_time
        smooth_prop_vol = result.builder_dist_volatility_over_time
        smooth_value_vol = result.value_share_volatility_over_time

    ax9.plot(slots_smooth, smooth_prop_vol, label='Population Distribution', linewidth=2, alpha=0.8)
    ax9.plot(slots_smooth, smooth_value_vol, label='Value-Capture', linewidth=2, alpha=0.8)
    ax9.set_xlabel('Slot', fontsize=10)
    ax9.set_ylabel('L1 Change', fontsize=10)
    ax9.set_title('Volatility (Distribution Churn) Over Time', fontsize=11, fontweight='bold')
    ax9.legend(loc='best', fontsize=9, framealpha=0.9)
    ax9.grid(True, alpha=0.3)

    # 10. Welfare Over Time (full-width bottom row)
    ax10 = fig.add_subplot(gs[3, :])

    if window > 1 and len(result.welfare_history) > 0:
        smooth_welfare = np.convolve(result.welfare_history, np.ones(window) / window, mode='valid')
        slots_welfare = np.arange(window - 1, len(result.welfare_history))
    else:
        smooth_welfare = result.welfare_history
        slots_welfare = np.arange(len(result.welfare_history))

    ax10.plot(slots_welfare, smooth_welfare, linewidth=2, color='steelblue', alpha=0.9, label='Welfare (smoothed)')
    mean_w = result.stats['mean_welfare']
    ax10.axhline(y=mean_w, color='green', linestyle='--', linewidth=1.5, alpha=0.8, label=f'Mean welfare = {mean_w:.2f}')
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
    if result.config.policy_type == "EMA":
        policy_info += f" (η={result.config.eta}, β_reg={result.config.beta_reg}, c={result.config.cost_c})"
    elif result.config.policy_type == "UCB":
        policy_info += f" (α={result.config.alpha})"
    elif result.config.policy_type == "ABR":
        policy_info += (
            f" (threshold={100 * result.config.improvement_threshold_pct:.3f}%, "
            f"grid={result.config.utility_eval_time_steps})"
        )
    elif result.config.policy_type == "MWU":
        norm = result.config.payoff_normalization
        norm_str = f"{norm:.3f}" if norm is not None else "auto"
        policy_info += f" (η={result.config.mwu_eta}, norm={norm_str})"
    else:
        policy_info += " (custom)"

    fig.suptitle(f'Experiment: {result.config.name} | {policy_info}',
                 fontsize=13, fontweight='bold')

    if save_plots:
        results_dir = Path(result.config.results_dir)
        results_dir.mkdir(exist_ok=True)
        filename = results_dir / f"{result.config.name}_details.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Detail plot saved to: {filename}")

    plt.close(fig)


def plot_network_setup(config: ExperimentConfig, save_plots: bool = True):
    """
    Visualize the initial network setup showing regions, sources, and topology.

    Args:
        config: ExperimentConfig object
        save_plots: Whether to save the plot to disk
    """
    fig, ax = plt.subplots(figsize=(16, 8))

    n_regions = config.n_regions
    region_names = config.region_names if config.region_names else [f"R{i}" for i in range(n_regions)]

    # Position regions horizontally (equally spaced)
    region_positions = np.linspace(0, 10, n_regions)
    region_y = 5  # Y position for regions

    # Draw regions as circles
    region_radius = 0.4
    for i, (x, name) in enumerate(zip(region_positions, region_names)):
        circle = plt.Circle((x, region_y), region_radius, color='lightblue',
                           ec='darkblue', linewidth=2, alpha=0.7, zorder=2)
        ax.add_patch(circle)
        ax.text(x, region_y, f"{i}", ha='center', va='center',
               fontsize=14, fontweight='bold', zorder=3)
        ax.text(x, region_y - 0.8, name, ha='center', va='top',
               fontsize=11, fontweight='bold')

    # Draw sources at their initial regions
    source_y = 7.5  # Y position for sources (above regions)
    source_radius = 0.35

    if config.sources_config:
        for src_name, region, lambda_rate, mu_val, sigma_val in config.sources_config:
            src_x = region_positions[region]

            circle = plt.Circle((src_x, source_y), source_radius, color='gold',
                              ec='darkorange', linewidth=2.5, alpha=0.9, zorder=2)
            ax.add_patch(circle)

            ax.text(src_x, source_y, "S", ha='center', va='center',
                   fontsize=12, fontweight='bold', zorder=3)
            ax.text(src_x, source_y + 0.6, f"{src_name}", ha='center', va='bottom',
                   fontsize=10, fontweight='bold', color='darkorange')
            ax.text(src_x, source_y + 0.9, f"λ={lambda_rate}", ha='center', va='bottom',
                   fontsize=9, color='darkred')

            ax.annotate('', xy=(src_x, region_y + region_radius),
                       xytext=(src_x, source_y - source_radius),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.6))

    # Draw distance indicators between adjacent regions
    distance_y = 3.5  # Y position for distance labels
    for i in range(n_regions - 1):
        x1, x2 = region_positions[i], region_positions[i + 1]
        mid_x = (x1 + x2) / 2

        # Draw line showing distance
        ax.plot([x1 + region_radius, x2 - region_radius],
               [distance_y, distance_y],
               'k-', linewidth=1, alpha=0.3)

        # Distance label
        ax.text(mid_x, distance_y - 0.3, f"d={1}", ha='center', va='top',
               fontsize=9, style='italic', color='gray')

    # Add legend/info box
    info_text = (
        f"Configuration: {config.name}\n"
        f"Regions: {n_regions}  |  Sources: {len(config.sources_config) if config.sources_config else 0}  |  "
        f"Builders: {config.n_builders}  |  Delta: {config.delta}s\n"
        f"Equal-split sharing  |  Migration cost: c={config.cost_c if hasattr(config, 'cost_c') else 'N/A'}\n"
        f"Initial distribution: Uniform ({config.n_builders // n_regions} builders per region)"
    )

    ax.text(0.5, 0.08, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', horizontalalignment='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Add legend at bottom left, next to config box
    legend_text = (
        "Legend:\n"
        "● Region (numbered by ID)\n"
        "● Information Source (with value V)\n"
        "d = distance units between regions"
    )
    ax.text(0.02, 0.08, legend_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', horizontalalignment='left',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    # Set axis properties
    ax.set_xlim(-1, 11)
    ax.set_ylim(2, 9)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    policy_info = config.policy_type if hasattr(config, 'policy_type') else "N/A"
    ax.set_title(f'Network Setup: {config.name} | Policy: {policy_info}',
                fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_plots:
        results_dir = Path(config.results_dir if hasattr(config, 'results_dir') else 'results')
        results_dir.mkdir(exist_ok=True)
        filename = results_dir / f"{config.name}_network_setup.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Network setup plot saved to: {filename}")

    plt.close(fig)


def print_comparison_table(results: List[ExperimentResult]):
    """Print a comparison table of final metrics."""
    print(f"\n{'='*120}")
    print("EXPERIMENT COMPARISON TABLE")
    print(f"{'='*120}")

    # Header
    print(f"\n{'Experiment':<25} {'Policy':<8} {'Reward':<12} {'Welfare':<12} "
          f"{'BuilderDist':<12} {'BuilderDist':<12} {'Region':<12} {'Region':<12} {'Coverage':<12}")
    print(f"{'':25} {'':8} {'':12} {'Mean':<12} "
          f"{'Gini':<12} {'Entropy':<12} {'Gini':<12} {'Entropy':<12} {'Ratio':<12}")
    print("-" * 120)

    for result in results:
        stats = result.stats
        print(f"{result.config.name:<25} {result.config.policy_type:<8} "
              f"{stats['avg_reward']:<12.4f} {stats['mean_welfare']:<12.4f} "
              f"{stats['builder_dist_gini']:<12.4f} {stats['builder_dist_entropy']:<12.4f} "
              f"{stats['region_gini']:<12.4f} {stats['region_entropy']:<12.4f} "
              f"{stats['mean_coverage_ratio']:<12.4f}")

    print("="*120)
