#!/usr/bin/env python3
"""
Tokenize MDS datasets with subset folder structure.

This script downloads MDS datasets from HuggingFace, tokenizes them using a specified
tokenizer, and uploads the tokenized result back to HuggingFace while preserving the
subset folder structure (000_00000, 000_00001, etc.).

Usage:
    # Tokenize from HuggingFace and upload
    python src/tokenization/tokenize_mds_subsets.py \
        --hf_repo QuangDuy/FineWiki-mds \
        --tokenizer_path /path/to/tokenizer \
        --output_repo QuangDuy/FineWiki-mds-tokenized \
        --batch_size 5000

    # Tokenize from local directory
    python src/tokenization/tokenize_mds_subsets.py \
        --input_dir ./data/FineWiki-mds \
        --tokenizer_path /path/to/tokenizer \
        --output_dir ./data/FineWiki-mds-tokenized
"""

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Optional, Dict
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.mds_helpers import (
    discover_subset_folders,
    load_mds_subset,
    save_mds_subset,
    download_from_hf,
    upload_to_hf,
    get_tokenizer_special_tokens,
    verify_mds_structure,
    count_total_samples
)


MDS_COLS_PRE_TOKENIZED = {
    'input_ids': 'ndarray:uint16',
    'id': 'str',
    'len': 'int'
}


def tokenize_subset(
    subset_path: Path,
    output_path: Path,
    tokenizer,
    batch_size: int = 5000,
    compression: str = 'zstd',
    resume: bool = False
) -> Dict:
    """
    Tokenize a single subset folder.
    
    Args:
        subset_path: Path to input subset folder
        output_path: Path to output subset folder
        tokenizer: HuggingFace tokenizer
        batch_size: Number of samples to process at once
        compression: Compression type for MDS
        resume: Whether to skip if output already exists
        
    Returns:
        Statistics dictionary
    """
    if resume and output_path.exists():
        index_file = output_path / "index.json"
        if index_file.exists():
            print(f"  Skipping {subset_path.name} (already exists)")
            return {'skipped': True}
    
    print(f"  Tokenizing {subset_path.name}...")
    
    stats = {
        'num_tokens': 0,
        'num_samples': 0,
        'skipped': False
    }
    
    def tokenized_samples_generator():
        """Generator that yields tokenized samples."""
        nonlocal stats
        
        for batch in load_mds_subset(subset_path, batch_size=batch_size):
            texts = [sample['text'] for sample in batch]
            ids = [sample['id'] for sample in batch]
            
            tokenized = tokenizer(
                texts,
                truncation=False,
                padding=False,
                return_tensors="np"
            )
            
            for idx, (sample_id, input_ids) in enumerate(zip(ids, tokenized['input_ids'])):
                input_ids_uint16 = input_ids.astype(np.uint16)
                seq_len = len(input_ids_uint16)
                
                yield {
                    'input_ids': input_ids_uint16,
                    'id': sample_id,
                    'len': seq_len
                }
                
                stats['num_tokens'] += seq_len
                stats['num_samples'] += 1
            
            del texts, ids, tokenized
            gc.collect()
    
    total_samples, total_size = save_mds_subset(
        output_path=output_path,
        columns=MDS_COLS_PRE_TOKENIZED,
        samples_generator=tokenized_samples_generator(),
        compression=compression
    )
    
    stats['output_samples'] = total_samples
    stats['output_size'] = total_size
    
    stats_file = output_path / "tokenization_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Tokenized {stats['num_samples']:,} samples, {stats['num_tokens']:,} tokens")
    
    return stats


def tokenize_dataset(
    input_path: Path,
    output_path: Path,
    tokenizer_path: str,
    batch_size: int = 5000,
    compression: str = 'zstd',
    resume: bool = False
) -> Dict:
    """
    Tokenize entire MDS dataset with subset structure.
    
    Args:
        input_path: Path to input dataset
        output_path: Path to output dataset
        tokenizer_path: Path to tokenizer
        batch_size: Batch size for tokenization
        compression: Compression type
        resume: Whether to resume from existing output
        
    Returns:
        Overall statistics dictionary
    """
    print("="*60)
    print(f"Tokenizing MDS dataset")
    print("="*60)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Batch size: {batch_size}")
    print()
    
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            add_prefix_space=True,
            trust_remote_code=True
        )
        print(f"Loaded tokenizer")
        
        special_tokens = get_tokenizer_special_tokens(tokenizer)
        print(f"Special tokens: {special_tokens}")
        print()
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        raise
    
    subset_folders = discover_subset_folders(input_path)
    if not subset_folders:
        raise ValueError(f"No subset folders found in {input_path}")
    
    print(f"Found {len(subset_folders)} subset folders")
    print()
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    overall_stats = {
        'total_tokens': 0,
        'total_samples': 0,
        'subset_stats': {}
    }
    
    for subset_folder in tqdm(subset_folders, desc="Processing subsets"):
        subset_name = subset_folder.name
        output_subset = output_path / subset_name
        
        try:
            stats = tokenize_subset(
                subset_path=subset_folder,
                output_path=output_subset,
                tokenizer=tokenizer,
                batch_size=batch_size,
                compression=compression,
                resume=resume
            )
            
            if not stats['skipped']:
                overall_stats['total_tokens'] += stats['num_tokens']
                overall_stats['total_samples'] += stats['num_samples']
            
            overall_stats['subset_stats'][subset_name] = stats
            
        except Exception as e:
            print(f"Failed to process {subset_name}: {e}")
            overall_stats['subset_stats'][subset_name] = {'error': str(e)}
    
    stats_file = output_path / "overall_tokenization_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(overall_stats, f, indent=2)
    
    num_tokens_file = output_path / "num_tokens.json"
    with open(num_tokens_file, 'w') as f:
        json.dump({'num_tokens': overall_stats['total_tokens']}, f, indent=2)
    
    print()
    print("="*60)
    print("Tokenization complete!")
    print("="*60)
    print(f"Total samples: {overall_stats['total_samples']:,}")
    print(f"Total tokens: {overall_stats['total_tokens']:,}")
    print(f"Output directory: {output_path.absolute()}")
    print()
    
    return overall_stats


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize MDS datasets with subset folder structure"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input_dir',
        type=str,
        help='Local input directory containing MDS dataset'
    )
    input_group.add_argument(
        '--hf_repo',
        type=str,
        help='HuggingFace repository ID to download (e.g., QuangDuy/FineWiki-mds)'
    )
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument(
        '--output_dir',
        type=str,
        help='Local output directory for tokenized dataset'
    )
    output_group.add_argument(
        '--output_repo',
        type=str,
        help='HuggingFace repository ID for upload (e.g., QuangDuy/FineWiki-mds-tokenized)'
    )
    parser.add_argument(
        '--tokenizer_path',
        type=str,
        required=True,
        help='Path to tokenizer (local or HuggingFace model name)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=5000,
        help='Batch size for tokenization (default: 5000)'
    )
    parser.add_argument(
        '--compression',
        type=str,
        default='zstd',
        choices=['zstd', 'gz', 'none'],
        help='Compression type for output (default: zstd)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing output (skip completed subsets)'
    )
    parser.add_argument(
        '--hf_token',
        type=str,
        default=None,
        help='HuggingFace API token'
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
        default='./temp_tokenization',
        help='Temporary directory for processing (default: ./temp_tokenization)'
    )
    
    args = parser.parse_args()
    
    compression = None if args.compression == 'none' else args.compression
    
    if args.hf_repo:
        print(f"Downloading from HuggingFace: {args.hf_repo}")
        cache_dir = Path(args.hf_cache_dir) if args.hf_cache_dir else None
        input_path = download_from_hf(
            repo_id=args.hf_repo,
            cache_dir=cache_dir,
            token=args.hf_token
        )
        print()
    else:
        input_path = Path(args.input_dir)
        if not input_path.exists():
            parser.error(f"Input directory does not exist: {input_path}")
    
    if args.output_repo:
        output_path = Path(args.temp_dir) / "tokenized"
        output_path.mkdir(parents=True, exist_ok=True)
        upload_to_hf_after = True
    else:
        output_path = Path(args.output_dir)
        upload_to_hf_after = False
    
    try:
        stats = tokenize_dataset(
            input_path=input_path,
            output_path=output_path,
            tokenizer_path=args.tokenizer_path,
            batch_size=args.batch_size,
            compression=compression,
            resume=args.resume
        )
        
        if upload_to_hf_after:
            print("Uploading to HuggingFace...")
            upload_to_hf(
                local_path=output_path,
                repo_id=args.output_repo,
                token=args.hf_token,
                private=args.private,
                commit_message=f"Tokenized dataset from {args.hf_repo or args.input_dir}"
            )
            print()
            print(f"Dataset available at: https://huggingface.co/datasets/{args.output_repo}")
        
        return 0
        
    except Exception as e:
        print(f"\nError during tokenization: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
