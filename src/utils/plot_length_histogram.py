#!/usr/bin/env python3
"""
Plot histogram of token lengths from sampled MDS dataset.

Usage:
    python src/utils/plot_length_histogram.py \
        --input_dir temp_sampling/sampled \
        --output histogram.png
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.mds_helpers import discover_subset_folders, load_mds_subset


def collect_lengths(input_path: Path) -> dict:
    """
    Collect all lengths from the sampled dataset.
    
    Returns:
        Dictionary with 'lengths' list and 'bin_labels' dict mapping bin -> lengths
    """
    lengths = []
    bin_lengths = defaultdict(list)
    
    subset_folders = discover_subset_folders(input_path)
    if not subset_folders:
        # Try loading directly if it's a single subset
        subset_folders = [input_path]
    
    for subset_folder in subset_folders:
        try:
            for batch in load_mds_subset(subset_folder, batch_size=1000):
                for sample in batch:
                    length = int(sample.get('len', 0))
                    bin_label = sample.get('bin_label', 'unknown')
                    if isinstance(bin_label, bytes):
                        bin_label = bin_label.decode('utf-8')
                    
                    lengths.append(length)
                    bin_lengths[bin_label].append(length)
        except Exception as e:
            print(f"Warning: Could not load {subset_folder}: {e}")
            continue
    
    return {
        'lengths': lengths,
        'bin_lengths': dict(bin_lengths)
    }


def plot_histogram(
    data: dict,
    output_path: Path = None,
    title: str = "Token Length Distribution",
    bins: int = 50,
    show_bins: bool = True
):
    """
    Plot histogram of token lengths.
    
    Args:
        data: Dictionary from collect_lengths
        output_path: Path to save the plot (None to display)
        title: Plot title
        bins: Number of histogram bins
        show_bins: Whether to show bin labels in legend
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        raise ImportError("matplotlib is required. Install with: pip install matplotlib")
    
    lengths = data['lengths']
    bin_lengths = data['bin_lengths']
    
    if not lengths:
        print("No data to plot!")
        return
    
    # Set up the figure with a modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Color palette
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c']
    
    # --- Left plot: Overall histogram ---
    ax1 = axes[0]
    n, bins_edges, patches = ax1.hist(
        lengths, 
        bins=bins, 
        color='#3498db', 
        alpha=0.7,
        edgecolor='white',
        linewidth=0.5
    )
    
    ax1.set_xlabel('Token Length', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title(f'{title}\n(Total: {len(lengths):,} samples)', fontsize=12, fontweight='bold')
    
    # Add statistics
    mean_len = np.mean(lengths)
    median_len = np.median(lengths)
    ax1.axvline(mean_len, color='#e74c3c', linestyle='--', linewidth=2, label=f'Mean: {mean_len:.0f}')
    ax1.axvline(median_len, color='#2ecc71', linestyle='--', linewidth=2, label=f'Median: {median_len:.0f}')
    ax1.legend(fontsize=9)
    
    # --- Right plot: Stacked histogram by bin ---
    ax2 = axes[1]
    
    if show_bins and bin_lengths:
        # Sort bins by their lower bound
        def parse_bin_lower(label):
            try:
                return int(label.split(',')[0].replace('(', ''))
            except:
                return 0
        
        sorted_bins = sorted(bin_lengths.keys(), key=parse_bin_lower)
        
        # Create stacked histogram data
        bin_data = [bin_lengths[b] for b in sorted_bins]
        
        ax2.hist(
            bin_data,
            bins=bins,
            stacked=True,
            color=colors[:len(sorted_bins)],
            alpha=0.8,
            edgecolor='white',
            linewidth=0.5,
            label=sorted_bins
        )
        
        ax2.set_xlabel('Token Length', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('Distribution by Bin', fontsize=12, fontweight='bold')
        ax2.legend(title='Bins', fontsize=9, title_fontsize=10)
    else:
        ax2.hist(lengths, bins=bins, color='#3498db', alpha=0.7, edgecolor='white')
        ax2.set_xlabel('Token Length', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('Distribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved histogram to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Print statistics
    print()
    print("=" * 50)
    print("Statistics")
    print("=" * 50)
    print(f"Total samples: {len(lengths):,}")
    print(f"Min length: {min(lengths):,}")
    print(f"Max length: {max(lengths):,}")
    print(f"Mean length: {mean_len:.2f}")
    print(f"Median length: {median_len:.2f}")
    print(f"Std dev: {np.std(lengths):.2f}")
    print()
    
    if bin_lengths:
        print("Samples per bin:")
        for label in sorted(bin_lengths.keys(), key=parse_bin_lower):
            count = len(bin_lengths[label])
            avg = np.mean(bin_lengths[label]) if bin_lengths[label] else 0
            print(f"  {label}: {count:,} samples (avg: {avg:.1f})")


def main():
    parser = argparse.ArgumentParser(
        description="Plot histogram of token lengths from sampled MDS dataset"
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Path to sampled MDS dataset'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for histogram image (default: display)'
    )
    parser.add_argument(
        '--title',
        type=str,
        default='Token Length Distribution',
        help='Plot title'
    )
    parser.add_argument(
        '--bins',
        type=int,
        default=50,
        help='Number of histogram bins (default: 50)'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_path}")
        return 1
    
    output_path = Path(args.output) if args.output else None
    
    print(f"Reading data from: {input_path}")
    data = collect_lengths(input_path)
    
    if not data['lengths']:
        print("Error: No data found!")
        return 1
    
    plot_histogram(
        data=data,
        output_path=output_path,
        title=args.title,
        bins=args.bins
    )
    
    return 0


if __name__ == "__main__":
    exit(main())



