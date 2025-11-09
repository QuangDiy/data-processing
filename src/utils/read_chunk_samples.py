#!/usr/bin/env python3
"""
Script to read and display samples from chunk files.

Usage:
    # Read a few samples from the first subset
    python src/utils/read_chunk_samples.py \
        --chunk_dir temp_chunking/chunked-1024 \
        --num_samples 5

    # Read with tokenizer to decode text
    python src/utils/read_chunk_samples.py \
        --chunk_dir temp_chunking/chunked-1024 \
        --tokenizer_path tokenizer/HUIT-BERT \
        --num_samples 10

    # Read from specific subset
    python src/utils/read_chunk_samples.py \
        --chunk_dir temp_chunking/chunked-1024 \
        --subset 000_00000 \
        --num_samples 5

    # Read all samples in a subset
    python src/utils/read_chunk_samples.py \
        --chunk_dir temp_chunking/chunked-1024 \
        --subset 000_00000 \
        --all

    # Read all samples with tokenizer
    python src/utils/read_chunk_samples.py \
        --chunk_dir temp_chunking/chunked-1024 \
        --tokenizer_path tokenizer/HUIT-BERT \
        --all
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from streaming import StreamingDataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.mds_helpers import discover_subset_folders, load_index_file


def format_token_ids(token_ids: np.ndarray, max_display: int = 50) -> str:
    """Format token IDs for display."""
    if len(token_ids) <= max_display:
        return str(token_ids.tolist())
    else:
        first = token_ids[:max_display//2].tolist()
        last = token_ids[-max_display//2:].tolist()
        return f"{first} ... (truncated, total {len(token_ids)} tokens) ... {last}"


def decode_with_tokenizer(token_ids: np.ndarray, tokenizer) -> str:
    """Decode token IDs to text using tokenizer."""
    try:
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        text = tokenizer.decode(token_ids, skip_special_tokens=False)
        return text
    except Exception as e:
        return f"[Error decoding: {e}]"


def read_chunk_samples(
    chunk_dir: Path,
    subset_name: Optional[str] = None,
    num_samples: Optional[int] = 5,
    tokenizer_path: Optional[str] = None,
    show_stats: bool = True,
    read_all: bool = False
):
    """
    Read and display samples from chunk files.
    
    Args:
        chunk_dir: Path to directory containing chunks
        subset_name: Specific subset name (e.g., '000_00000'), None to use first subset
        num_samples: Number of samples to read (None if read_all=True)
        tokenizer_path: Path to tokenizer (optional, to decode text)
        show_stats: Show statistics about the subset
        read_all: Read all samples in the subset
    """
    chunk_dir = Path(chunk_dir)
    
    if not chunk_dir.exists():
        print(f"Error: Directory {chunk_dir} does not exist")
        return
    
    subset_folders = discover_subset_folders(chunk_dir)
    
    if not subset_folders:
        print(f"Error: No subset folders found in {chunk_dir}")
        return
    
    if subset_name:
        subset_path = chunk_dir / subset_name
        if subset_path not in subset_folders:
            print(f"Error: Subset '{subset_name}' does not exist")
            print(f"   Available subsets: {[s.name for s in subset_folders]}")
            return
    else:
        subset_path = subset_folders[0]
        subset_name = subset_path.name
    
    print("=" * 80)
    print(f"Reading samples from: {subset_path}")
    print("=" * 80)
    
    index_file = subset_path / "index.json"
    total_samples_in_subset = None
    if index_file.exists():
        index_data = load_index_file(index_file)
        total_samples_in_subset = sum(shard.get('samples', 0) for shard in index_data.get('shards', []))
        total_shards = len(index_data.get('shards', []))
        if show_stats:
            print(f"\nStatistics for subset '{subset_name}':")
            print(f"   - Total samples: {total_samples_in_subset:,}")
            print(f"   - Total shards: {total_shards}")
            print()
    
    if read_all:
        if total_samples_in_subset is not None:
            num_samples = total_samples_in_subset
            print(f"Will read all {num_samples:,} samples in subset\n")
        else:
            print("Warning: Cannot determine total number of samples, will read until end of dataset\n")
            num_samples = None
    elif num_samples is None:
        num_samples = 5
    
    tokenizer = None
    if tokenizer_path:
        try:
            from transformers import AutoTokenizer
            print(f"Loading tokenizer from {tokenizer_path}...")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            print("Tokenizer loaded successfully\n")
        except Exception as e:
            print(f"Warning: Cannot load tokenizer: {e}")
            print("   Will only display token IDs\n")
    
    try:
        dataset = StreamingDataset(
            local=str(subset_path),
            split=None,
            shuffle=False,
            batch_size=1
        )
        
        if num_samples is not None:
            print(f"Reading {num_samples:,} sample(s)...\n")
        else:
            print(f"Reading all samples...\n")
        
        sample_count = 0
        for idx, sample in enumerate(dataset):
            if num_samples is not None and sample_count >= num_samples:
                break
            
            print("-" * 80)
            print(f"Sample #{idx + 1} (index {idx})")
            print("-" * 80)
            
            if 'input_ids' in sample:
                input_ids = sample['input_ids']
                
                if not isinstance(input_ids, np.ndarray):
                    input_ids = np.array(input_ids, dtype=np.uint16)
                
                length = sample.get('len', len(input_ids))
                
                print(f"   Length: {length} tokens")
                print(f"   Shape: {input_ids.shape}")
                print(f"   Dtype: {input_ids.dtype}")
                
                print(f"\n   Token IDs (full):")
                print(f"   {input_ids.tolist()}")
                
                if tokenizer:
                    decoded_text = decode_with_tokenizer(input_ids, tokenizer)
                    print(f"\n   Decoded text (full):")
                    print(f"   {decoded_text}")
                
                print(f"\n   Token value range: [{input_ids.min()}, {input_ids.max()}]")
                
            else:
                print(f"   Sample keys: {list(sample.keys())}")
                for key, value in sample.items():
                    if isinstance(value, np.ndarray):
                        print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
                    else:
                        print(f"   {key}: {value}")
            
            print()
            sample_count += 1
        
        print("=" * 80)
        print(f"Read {sample_count} sample(s)")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error reading samples: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Read and display samples from chunk files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--chunk_dir',
        type=str,
        default='temp_chunking/chunked-1024',
        help='Path to directory containing chunks (default: temp_chunking/chunked-1024)'
    )
    
    parser.add_argument(
        '--subset',
        type=str,
        default=None,
        help='Specific subset name (e.g., 000_00000). If not specified, uses first subset'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='Number of samples to read (default: 5). Ignored if --all is used'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Read all samples in the subset'
    )
    
    parser.add_argument(
        '--tokenizer_path',
        type=str,
        default=None,
        help='Path to tokenizer for decoding text (optional)'
    )
    
    parser.add_argument(
        '--no_stats',
        action='store_true',
        help='Do not show statistics about the subset'
    )
    
    args = parser.parse_args()
    
    read_chunk_samples(
        chunk_dir=Path(args.chunk_dir),
        subset_name=args.subset,
        num_samples=args.num_samples if not args.all else None,
        tokenizer_path=args.tokenizer_path,
        show_stats=not args.no_stats,
        read_all=args.all
    )


if __name__ == '__main__':
    main()

