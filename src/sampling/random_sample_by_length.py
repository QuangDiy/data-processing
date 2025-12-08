#!/usr/bin/env python3
"""
Random sampling script for tokenized MDS datasets by token length bins.

This script samples a specified number of samples from each token length bin
using reservoir sampling for memory efficiency.

Usage:
    # Sample from HuggingFace and upload to another repo
    python src/sampling/random_sample_by_length.py \
        --hf_repo QuangDuy/FineWiki-mds-tokenized \
        --output_repo QuangDuy/FineWiki-mds-tokenized-samples \
        --samples_per_bin 500 \
        --bins 0,512,1024,2048,4096

    # Sample from HuggingFace and save locally
    python src/sampling/random_sample_by_length.py \
        --hf_repo QuangDuy/FineWiki-mds-tokenized \
        --output_dir ./sampled_data \
        --samples_per_bin 500

    # Sample from local directory
    python src/sampling/random_sample_by_length.py \
        --input_dir ./data/FineWiki-mds-tokenized \
        --output_dir ./sampled_data \
        --samples_per_bin 500
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(Path(__file__).parent.parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.mds_helpers import (
    discover_subset_folders,
    load_mds_subset,
    save_mds_subset,
    download_from_hf,
    upload_to_hf,
    load_index_file,
)


MDS_COLS_SAMPLED = {
    'input_ids': 'ndarray:uint16',
    'len': 'int',
    'bin_label': 'str',
}


def get_bin_label(length: int, bin_edges: List[int]) -> Optional[str]:
    """
    Get the bin label for a given token length.
    
    Args:
        length: Token length
        bin_edges: List of bin edges (e.g., [0, 512, 1024, 2048, 4096])
        
    Returns:
        Bin label string (e.g., '(512,1024]') or None if outside all bins
    """
    for i in range(len(bin_edges) - 1):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        if lower < length <= upper:
            return f"({lower},{upper}]"
    return None


def reservoir_sample(
    input_path: Path,
    bin_edges: List[int],
    samples_per_bin: int,
    seed: int = 42,
    batch_size: int = 1000
) -> Dict[str, List[Dict]]:
    """
    Perform reservoir sampling to select samples from each bin.
    
    Uses reservoir sampling algorithm (Algorithm R) to efficiently sample
    without loading all data into memory.
    
    Args:
        input_path: Path to input dataset
        bin_edges: List of bin edges
        samples_per_bin: Number of samples to collect per bin
        seed: Random seed for reproducibility
        batch_size: Batch size for reading data
        
    Returns:
        Dictionary mapping bin labels to lists of sampled data
    """
    random.seed(seed)
    np.random.seed(seed)
    
    bin_labels = []
    for i in range(len(bin_edges) - 1):
        label = f"({bin_edges[i]},{bin_edges[i+1]}]"
        bin_labels.append(label)
    
    reservoirs: Dict[str, List[Dict]] = {label: [] for label in bin_labels}
    bin_counts: Dict[str, int] = {label: 0 for label in bin_labels}
    
    subset_folders = discover_subset_folders(input_path)
    if not subset_folders:
        raise ValueError(f"No subset folders found in {input_path}")
    
    print(f"Found {len(subset_folders)} subset folders")
    print(f"Bin edges: {bin_edges}")
    print(f"Samples per bin: {samples_per_bin}")
    print()
    
    total_samples = 0
    for subset_folder in subset_folders:
        index_file = subset_folder / "index.json"
        if index_file.exists():
            index_data = load_index_file(index_file)
            for shard in index_data.get('shards', []):
                total_samples += shard.get('samples', 0)
    
    print(f"Total samples to process: {total_samples:,}")
    print()
    
    with tqdm(total=total_samples, desc="Sampling", unit="samples") as pbar:
        for subset_folder in subset_folders:
            for batch in load_mds_subset(subset_folder, batch_size=batch_size):
                for sample in batch:
                    if 'len' in sample:
                        length = int(sample['len'])
                    elif 'input_ids' in sample:
                        input_ids = sample['input_ids']
                        if isinstance(input_ids, np.ndarray):
                            length = len(input_ids)
                        else:
                            length = len(input_ids)
                    else:
                        pbar.update(1)
                        continue
                    
                    bin_label = get_bin_label(length, bin_edges)
                    if bin_label is None:
                        pbar.update(1)
                        continue
                    
                    bin_counts[bin_label] += 1
                    n = bin_counts[bin_label]
                    
                    input_ids = sample['input_ids']
                    if not isinstance(input_ids, np.ndarray):
                        input_ids = np.array(input_ids, dtype=np.uint16)
                    
                    sample_data = {
                        'input_ids': input_ids.astype(np.uint16),
                        'len': length,
                        'bin_label': bin_label,
                    }
                    
                    if n <= samples_per_bin:
                        reservoirs[bin_label].append(sample_data)
                    else:
                        j = random.randint(1, n)
                        if j <= samples_per_bin:
                            reservoirs[bin_label][j - 1] = sample_data
                    
                    pbar.update(1)
    
    print()
    print("=" * 60)
    print("Sampling Statistics")
    print("=" * 60)
    for label in bin_labels:
        collected = len(reservoirs[label])
        total_in_bin = bin_counts[label]
        print(f"  {label}: {collected}/{samples_per_bin} samples "
              f"(from {total_in_bin:,} total in bin)")
    
    total_collected = sum(len(v) for v in reservoirs.values())
    print(f"\nTotal samples collected: {total_collected}")
    print("=" * 60)
    
    return reservoirs


def save_samples_to_mds(
    reservoirs: Dict[str, List[Dict]],
    output_path: Path,
    compression: str = 'zstd'
) -> Tuple[int, int]:
    """
    Save sampled data to MDS format.
    
    Args:
        reservoirs: Dictionary mapping bin labels to sample lists
        output_path: Output directory path
        compression: Compression type
        
    Returns:
        Tuple of (total_samples, total_size_bytes)
    """
    print()
    print(f"Saving samples to MDS format: {output_path}")
    
    subset_path = output_path
    
    def samples_generator():
        """Generator that yields all samples from all bins."""
        for bin_label, samples in reservoirs.items():
            for sample in samples:
                yield sample
    
    total_samples, total_size = save_mds_subset(
        output_path=subset_path,
        columns=MDS_COLS_SAMPLED,
        samples_generator=samples_generator(),
        compression=compression
    )
    
    print(f"Saved {total_samples:,} samples ({total_size / 1024 / 1024:.2f} MB)")
    
    return total_samples, total_size


def random_sample_by_length(
    input_path: Path,
    output_path: Path,
    bin_edges: List[int],
    samples_per_bin: int = 500,
    seed: int = 42,
    batch_size: int = 1000,
    compression: str = 'zstd'
) -> Dict:
    """
    Main function to perform random sampling by token length bins.
    
    Args:
        input_path: Path to input tokenized dataset
        output_path: Path to output directory
        bin_edges: List of bin edges (e.g., [0, 512, 1024, 2048, 4096])
        samples_per_bin: Number of samples per bin
        seed: Random seed
        batch_size: Batch size for processing
        compression: Compression type for output
        
    Returns:
        Statistics dictionary
    """
    print("=" * 60)
    print("Random Sampling by Token Length Bins")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Bin edges: {bin_edges}")
    print(f"Samples per bin: {samples_per_bin}")
    print(f"Total target samples: {samples_per_bin * (len(bin_edges) - 1)}")
    print(f"Seed: {seed}")
    print()
    
    reservoirs = reservoir_sample(
        input_path=input_path,
        bin_edges=bin_edges,
        samples_per_bin=samples_per_bin,
        seed=seed,
        batch_size=batch_size
    )
    
    output_path.mkdir(parents=True, exist_ok=True)
    total_samples, total_size = save_samples_to_mds(
        reservoirs=reservoirs,
        output_path=output_path,
        compression=compression
    )
    
    stats = {
        'bin_edges': bin_edges,
        'samples_per_bin': samples_per_bin,
        'seed': seed,
        'total_samples': total_samples,
        'total_size_bytes': total_size,
        'bins': {}
    }
    
    for bin_label, samples in reservoirs.items():
        stats['bins'][bin_label] = {
            'count': len(samples),
            'lengths': [s['len'] for s in samples]
        }
    
    print()
    print("=" * 60)
    print("Sampling complete!")
    print("=" * 60)
    
    return stats


def parse_bins(bins_str: str) -> List[int]:
    """Parse comma-separated bin edges string."""
    return [int(x.strip()) for x in bins_str.split(',')]


def main():
    parser = argparse.ArgumentParser(
        description="Random sampling from tokenized MDS datasets by token length bins",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input_dir',
        type=str,
        help='Local input directory containing tokenized MDS dataset'
    )
    input_group.add_argument(
        '--hf_repo',
        type=str,
        help='HuggingFace repository ID to download (e.g., QuangDuy/FineWiki-mds-tokenized)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Local output directory for sampled dataset'
    )
    parser.add_argument(
        '--output_repo',
        type=str,
        default=None,
        help='HuggingFace repository ID for upload'
    )
    parser.add_argument(
        '--samples_per_bin',
        type=int,
        default=500,
        help='Number of samples per bin (default: 500)'
    )
    parser.add_argument(
        '--bins',
        type=str,
        default='0,512,1024,2048,4096',
        help='Comma-separated bin edges (default: 0,512,1024,2048,4096)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help='Batch size for processing (default: 1000)'
    )
    parser.add_argument(
        '--compression',
        type=str,
        default='zstd',
        choices=['zstd', 'gz', 'none'],
        help='Compression type for output (default: zstd)'
    )
    parser.add_argument(
        '--hf_token',
        type=str,
        default=None,
        help='HuggingFace API token (default: use HF_TOKEN from .env or environment variable)'
    )
    parser.add_argument(
        '--hf_cache_dir',
        type=str,
        default=None,
        help='Cache directory for HuggingFace downloads'
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Make output repository private'
    )
    parser.add_argument(
        '--temp_dir',
        type=str,
        default='./temp_sampling',
        help='Temporary directory for processing (default: ./temp_sampling)'
    )
    parser.add_argument(
        '--upload_workers',
        type=int,
        default=16,
        help='Number of parallel workers for HuggingFace upload (default: 16)'
    )
    
    args = parser.parse_args()
    
    if args.output_dir is None and args.output_repo is None:
        parser.error("Must specify either --output_dir or --output_repo")
    
    bin_edges = parse_bins(args.bins)
    if len(bin_edges) < 2:
        parser.error("Must specify at least 2 bin edges")
    
    compression = None if args.compression == 'none' else args.compression
    
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if args.output_repo and not hf_token:
        parser.error(
            "HuggingFace token required for upload. "
            "Either pass --hf_token, set HF_TOKEN in .env file, or set HF_TOKEN environment variable"
        )
    
    if args.hf_repo:
        print(f"Downloading from HuggingFace: {args.hf_repo}")
        cache_dir = Path(args.hf_cache_dir) if args.hf_cache_dir else None
        input_path = download_from_hf(
            repo_id=args.hf_repo,
            cache_dir=cache_dir,
            token=hf_token
        )
        print()
    else:
        input_path = Path(args.input_dir)
        if not input_path.exists():
            parser.error(f"Input directory does not exist: {input_path}")
    
    if args.output_repo:
        output_path = Path(args.temp_dir) / "sampled"
        output_path.mkdir(parents=True, exist_ok=True)
        upload_to_hf_after = True
    else:
        output_path = Path(args.output_dir)
        upload_to_hf_after = False
    
    try:
        stats = random_sample_by_length(
            input_path=input_path,
            output_path=output_path,
            bin_edges=bin_edges,
            samples_per_bin=args.samples_per_bin,
            seed=args.seed,
            batch_size=args.batch_size,
            compression=compression
        )
        
        if upload_to_hf_after:
            print()
            print("Uploading to HuggingFace...")
            upload_to_hf(
                local_path=output_path,
                repo_id=args.output_repo,
                token=hf_token,
                private=args.private,
                commit_message=f"Sampled dataset ({args.samples_per_bin} samples per bin) from {args.hf_repo or args.input_dir}",
                num_workers=args.upload_workers
            )
        
        return 0
        
    except Exception as e:
        print(f"\nError during sampling: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

