#!/usr/bin/env python3
"""
Chunk tokenized MDS datasets into fixed-length sequences.

This script implements the chunking algorithm from thuat_toan.md to split
tokenized sequences into fixed-length chunks suitable for model training.

Usage:
    # Chunk from HuggingFace and upload
    python src/sampling/chunk_tokenized_mds.py \
        --hf_repo QuangDuy/FineWiki-mds-tokenized \
        --output_repo QuangDuy/FineWiki-mds-tokenized-1024 \
        --chunk_size 1024 \
        --min_chunk_size 512 \
        --always_skip_size 128

    # Chunk from local directory
    python src/sampling/chunk_tokenized_mds.py \
        --input_dir ./data/FineWiki-mds-tokenized \
        --output_dir ./data/FineWiki-mds-tokenized-1024 \
        --chunk_size 1024
"""

import argparse
import gc
import json
import random
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Dict, Tuple, Optional
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
    verify_mds_structure,
    load_index_file,
    save_resume_state,
    load_resume_state,
    clear_resume_state,
    check_resume_status
)


MDS_COLS_CHUNKED = {
    'input_ids': 'ndarray:uint16',
    'len': 'int'
}


def enforce_prefix_space(chunk: np.ndarray, tokenizer, chunk_size: int) -> np.ndarray:
    """
    Ensure chunk starts with prefix space for proper word boundaries.
    
    Args:
        chunk: Token array
        tokenizer: HuggingFace tokenizer
        chunk_size: Maximum chunk size
        
    Returns:
        Token array with proper prefix space
    """
    try:
        max_tokens = chunk_size - 2
        
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=False)
        
        if chunk_text and chunk_text[0] == " ":
            if len(chunk) <= max_tokens:
                return chunk
            else:
                return chunk[:max_tokens]
        
        new_tokens = tokenizer.encode(
            chunk_text,
            add_special_tokens=False,
            truncation=False,
            padding=False
        )
        new_tokens = np.array(new_tokens, dtype=np.uint16)
        
        if len(new_tokens) <= max_tokens:
            return new_tokens
        else:
            return new_tokens[:max_tokens]
            
    except Exception as e:
        print(f"Warning: Could not enforce prefix space: {e}")
        max_tokens = chunk_size - 2
        if len(chunk) <= max_tokens:
            return chunk
        else:
            return chunk[:max_tokens]


def add_special_tokens(
    chunk: np.ndarray,
    bos_token: int,
    eos_token: Optional[int],
    tokenizer,
    chunk_size: int
) -> np.ndarray:
    """
    Add BOS token and ensure prefix space.
    
    Args:
        chunk: Token array
        bos_token: BOS token ID
        eos_token: EOS token ID (optional, not used currently)
        tokenizer: HuggingFace tokenizer
        chunk_size: Maximum chunk size
        
    Returns:
        Chunk with special tokens as uint16 array
    """
    chunk = enforce_prefix_space(chunk, tokenizer, chunk_size)
    
    if len(chunk) > 0 and chunk[0] != bos_token:
        chunk = np.concatenate([[bos_token], chunk])
    
    # Optionally add EOS token (currently not used per algorithm)
    # if eos_token is not None and len(chunk) > 0 and chunk[-1] != eos_token:
    #     chunk = np.concatenate([chunk, [eos_token]])
    
    # Enforce final length <= chunk_size - 1 (reserve 1 slot by convention)
    max_final_len = max(0, chunk_size - 1)
    if len(chunk) > max_final_len:
        chunk = chunk[:max_final_len]
    
    return chunk.astype(np.uint16)


def chunk_instance(
    tokens: np.ndarray,
    chunk_size: int,
    min_chunk_size: int,
    always_skip_size: int,
    backfill: bool,
    backfill_no_duplicates: bool,
    add_eos_token: bool,
    bos_token: int,
    eos_token: int,
    tokenizer,
    random_generator: random.Random = None
) -> Tuple[List[np.ndarray], int, int]:
    """
    Chunk a tokenized instance into fixed-length chunks.
    
    Implements the algorithm from thuat_toan.md lines 123-253.
    
    Args:
        tokens: Array of token IDs
        chunk_size: Target chunk size (e.g., 1024, 8192)
        min_chunk_size: Minimum chunk size to keep (e.g., 512)
        always_skip_size: Always skip chunks smaller than this (e.g., 32, 128)
        backfill: Whether to backfill short chunks
        backfill_no_duplicates: Whether backfill should avoid duplicates
        add_eos_token: Whether to add EOS token to end of sequence
        bos_token: BOS token ID
        eos_token: EOS token ID
        tokenizer: HuggingFace tokenizer for prefix space enforcement
        random_generator: Random generator for reproducibility
        
    Returns:
        Tuple of (chunks, amount_duplicated, amount_skipped)
    """
    if random_generator is None:
        random_generator = random.Random()
    
    if add_eos_token and len(tokens) > 0 and tokens[-1] != eos_token:
        tokens = np.concatenate([tokens, [eos_token]])
    
    chunk_size_for_cls_eos = chunk_size - 2
    
    if chunk_size_for_cls_eos <= 0:
        return [], 0, len(tokens)
    
    initial_chunks = []
    for i in range(0, len(tokens), chunk_size_for_cls_eos):
        chunk = tokens[i:i + chunk_size_for_cls_eos]
        initial_chunks.append(chunk)
    
    chunks = []
    amount_duplicated = 0
    amount_skipped = 0
    
    for i, chunk in enumerate(initial_chunks):
        chunk_len = len(chunk)
        
        if chunk_len < always_skip_size and i != 0:
            amount_skipped += chunk_len
            continue
        
        if chunk_len < min_chunk_size:
            
            if backfill and len(chunks) > 0:
                prev_chunk = chunks[-1]
                
                if len(prev_chunk) > 0 and prev_chunk[0] == bos_token:
                    prev_chunk_no_bos = prev_chunk[1:]
                else:
                    prev_chunk_no_bos = prev_chunk
                
                if backfill_no_duplicates:
                    max_random = min(chunk_size_for_cls_eos - chunk_len, len(prev_chunk_no_bos))
                    
                    if max_random > 0:
                        random_tokens = random_generator.randint(0, max_random)
                    else:
                        random_tokens = 0
                    
                    if random_tokens > 0:
                        backfilled_chunk = np.concatenate([
                            prev_chunk_no_bos[-random_tokens:],
                            chunk
                        ])
                        
                        prev_chunk_no_bos = prev_chunk_no_bos[:-random_tokens]
                        chunks[-1] = add_special_tokens(
                            prev_chunk_no_bos,
                            bos_token,
                            None,
                            tokenizer,
                            chunk_size
                        )
                    else:
                        backfilled_chunk = chunk
                    
                    chunks.append(add_special_tokens(
                        backfilled_chunk,
                        bos_token,
                        None,
                        tokenizer,
                        chunk_size
                    ))
                    
                else:
                    extra_needed = chunk_size_for_cls_eos - chunk_len
                    extra_needed = min(extra_needed, len(prev_chunk_no_bos))
                    
                    if extra_needed > 0:
                        backfilled_chunk = np.concatenate([
                            prev_chunk_no_bos[-extra_needed:],
                            chunk
                        ])
                        amount_duplicated += extra_needed
                    else:
                        backfilled_chunk = chunk
                    
                    chunks.append(add_special_tokens(
                        backfilled_chunk,
                        bos_token,
                        None,
                        tokenizer,
                        chunk_size
                    ))
            
            elif i == 0:
                chunks.append(add_special_tokens(
                    chunk,
                    bos_token,
                    None,
                    tokenizer,
                    chunk_size
                ))
            
            else:
                amount_skipped += chunk_len
        
        else:
            chunks.append(add_special_tokens(
                chunk,
                bos_token,
                None,
                tokenizer,
                chunk_size
            ))
    
    return chunks, amount_duplicated, amount_skipped


def count_subset_samples(subset_path: Path) -> int:
    """
    Count total samples in a subset.
    
    Args:
        subset_path: Path to subset folder
        
    Returns:
        Total number of samples
    """
    try:
        index_file = subset_path / "index.json"
        index_data = load_index_file(index_file)
        total = 0
        for shard in index_data.get('shards', []):
            total += shard.get('samples', 0)
        return total
    except Exception:
        return 0


def chunk_subset(
    subset_path: Path,
    output_path: Path,
    chunk_size: int,
    min_chunk_size: int,
    always_skip_size: int,
    backfill: bool,
    backfill_no_duplicates: bool,
    add_eos_token: bool,
    bos_token: int,
    eos_token: int,
    tokenizer,
    batch_size: int = 1000,
    compression: str = 'zstd',
    resume: bool = False,
    seed: int = 42,
    show_progress: bool = True
) -> Dict:
    """
    Chunk a single subset folder with robust resume capability.
    
    Args:
        subset_path: Path to input tokenized subset folder
        output_path: Path to output chunked subset folder
        chunk_size: Target chunk size
        min_chunk_size: Minimum chunk size to keep
        always_skip_size: Always skip chunks smaller than this
        backfill: Whether to backfill short chunks
        backfill_no_duplicates: Whether backfill should avoid duplicates
        add_eos_token: Whether to add EOS token
        bos_token: BOS token ID
        eos_token: EOS token ID
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size for processing
        compression: Compression type
        resume: Whether to resume from existing partial output
        seed: Random seed for reproducibility
        
    Returns:
        Statistics dictionary
    """
    total_samples_in_subset = count_subset_samples(subset_path)
    
    can_resume, samples_to_skip, status_msg = check_resume_status(output_path, total_samples_in_subset)
    
    if status_msg == "complete" and resume:
        print(f"  Skipping {subset_path.name} (already complete)")
        return {'skipped': True, 'reason': 'complete'}
    
    if can_resume and resume:
        print(f"  Resuming {subset_path.name}: {samples_to_skip:,}/{total_samples_in_subset:,} samples already processed")
        samples_remaining = total_samples_in_subset - samples_to_skip
    else:
        if status_msg == "possibly_corrupt":
            print(f"  Restarting {subset_path.name} (previous run possibly incomplete)")
        else:
            print(f"  Starting {subset_path.name} ({total_samples_in_subset:,} samples)...")
        samples_to_skip = 0
        samples_remaining = total_samples_in_subset
    
    stats = {
        'total_duplicated_tokens': 0,
        'total_tokens_written': 0,
        'total_tokens_skipped': 0,
        'total_input_samples': 0,
        'total_output_chunks': 0,
        'distribution': [],
        'skipped': False,
        'resumed_from': samples_to_skip
    }
    
    random_generator = random.Random(seed)
    
    if show_progress:
        pbar = tqdm(
            total=samples_remaining,
            desc=f"    Processing {subset_path.name}",
            unit="samples",
            leave=False,
            ncols=100,
            initial=0
        )
    else:
        pbar = None
    
    samples_processed_count = 0
    
    def chunked_samples_generator():
        """Generator that yields chunked samples, skipping already processed ones."""
        nonlocal stats, samples_processed_count
        
        sample_idx = 0
        
        for batch in load_mds_subset(subset_path, batch_size=batch_size):
            
            for sample in batch:
                if sample_idx < samples_to_skip:
                    sample_idx += 1
                    continue
                
                sample_idx += 1
                
                input_ids = sample['input_ids']
                
                if not isinstance(input_ids, np.ndarray):
                    input_ids = np.array(input_ids, dtype=np.uint16)
                
                chunks, duplicated, skipped = chunk_instance(
                    tokens=input_ids,
                    chunk_size=chunk_size,
                    min_chunk_size=min_chunk_size,
                    always_skip_size=always_skip_size,
                    backfill=backfill,
                    backfill_no_duplicates=backfill_no_duplicates,
                    add_eos_token=add_eos_token,
                    bos_token=bos_token,
                    eos_token=eos_token,
                    tokenizer=tokenizer,
                    random_generator=random_generator
                )
                
                stats['total_duplicated_tokens'] += duplicated
                stats['total_tokens_skipped'] += skipped
                stats['total_input_samples'] += 1
                samples_processed_count += 1
                
                if pbar:
                    pbar.update(1)
                
                if samples_processed_count % 500 == 0:
                    save_resume_state(
                        output_path,
                        samples_to_skip + samples_processed_count,
                        total_samples_in_subset
                    )
                
                for chunk in chunks:
                    chunk_len = len(chunk)
                    stats['total_tokens_written'] += chunk_len
                    stats['distribution'].append(chunk_len)
                    stats['total_output_chunks'] += 1
                    
                    yield {
                        'input_ids': chunk.astype(np.uint16),
                        'len': chunk_len
                    }
            
            gc.collect()
    
    total_samples, total_size = save_mds_subset(
        output_path=output_path,
        columns=MDS_COLS_CHUNKED,
        samples_generator=chunked_samples_generator(),
        compression=compression
    )
    
    if pbar:
        pbar.close()
    
    if stats['distribution']:
        distribution = sorted(stats['distribution'])
        percentiles = {}
        for p in range(0, 101, 10):
            idx = int(len(distribution) * p / 100)
            if idx >= len(distribution):
                idx = len(distribution) - 1
            percentiles[f'p{p}'] = int(distribution[idx])
        stats['percentiles'] = percentiles
    
    stats['output_samples'] = total_samples
    stats['output_size'] = total_size
    
    stats_file = output_path / "chunking_stats.json"
    with open(stats_file, 'w') as f:
        save_stats = {k: v for k, v in stats.items() if k != 'distribution'}
        json.dump(save_stats, f, indent=2)
    
    clear_resume_state(output_path)
    
    print(f"    ✓ Chunked {stats['total_input_samples']:,} samples → {stats['total_output_chunks']:,} chunks "
          f"(duplicated: {stats['total_duplicated_tokens']:,}, skipped: {stats['total_tokens_skipped']:,})")
    
    return stats


def chunk_subset_worker(args: Tuple) -> Tuple[str, Dict]:
    """
    Worker function for parallel subset processing.
    
    Args:
        args: Tuple containing all arguments for chunk_subset
        
    Returns:
        Tuple of (subset_name, stats_dict)
    """
    (subset_path, output_path, tokenizer_path, chunk_size, min_chunk_size,
     always_skip_size, backfill, backfill_no_duplicates, add_eos_token,
     bos_token, eos_token, batch_size, compression, resume, seed) = args
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            add_prefix_space=True,
            trust_remote_code=True
        )
    except Exception as e:
        return (subset_path.name, {'error': f"Failed to load tokenizer: {e}"})
    
    try:
        stats = chunk_subset(
            subset_path=subset_path,
            output_path=output_path,
            chunk_size=chunk_size,
            min_chunk_size=min_chunk_size,
            always_skip_size=always_skip_size,
            backfill=backfill,
            backfill_no_duplicates=backfill_no_duplicates,
            add_eos_token=add_eos_token,
            bos_token=bos_token,
            eos_token=eos_token,
            tokenizer=tokenizer,
            batch_size=batch_size,
            compression=compression,
            resume=resume,
            seed=seed,
            show_progress=False  # Disable inner progress bar for parallel processing
        )
        return (subset_path.name, stats)
    except Exception as e:
        return (subset_path.name, {'error': str(e)})


def chunk_dataset(
    input_path: Path,
    output_path: Path,
    tokenizer_path: str,
    chunk_size: int = 1024,
    min_chunk_size: int = 512,
    always_skip_size: int = 128,
    backfill: bool = True,
    backfill_no_duplicates: bool = True,
    add_eos_token: bool = False,
    batch_size: int = 1000,
    compression: str = 'zstd',
    resume: bool = False,
    seed: int = 42,
    num_workers: Optional[int] = None,
    subset_filter: Optional[str] = None
) -> Dict:
    """
    Chunk entire tokenized MDS dataset with subset structure.
    
    Args:
        input_path: Path to input tokenized dataset
        output_path: Path to output chunked dataset
        tokenizer_path: Path to tokenizer
        chunk_size: Target chunk size
        min_chunk_size: Minimum chunk size to keep
        always_skip_size: Always skip chunks smaller than this
        backfill: Whether to backfill short chunks
        backfill_no_duplicates: Whether backfill should avoid duplicates
        add_eos_token: Whether to add EOS token
        batch_size: Batch size for processing
        compression: Compression type
        resume: Whether to resume from existing output
        seed: Random seed for reproducibility
        num_workers: Number of parallel workers
        subset_filter: Process only this specific subset folder (e.g., 004_00004)
        
    Returns:
        Overall statistics dictionary
    """
    print("="*60)
    print(f"Chunking tokenized MDS dataset")
    print("="*60)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Chunk size: {chunk_size}")
    print(f"Min chunk size: {min_chunk_size}")
    print(f"Always skip size: {always_skip_size}")
    print(f"Backfill: {backfill}")
    print(f"Backfill no duplicates: {backfill_no_duplicates}")
    print(f"Add EOS token: {add_eos_token}")
    print(f"Resume mode: {'Enabled' if resume else 'Disabled'}")
    
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    print(f"Number of workers: {num_workers}")
    print()
    
    print("Loading tokenizer to get special tokens...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            add_prefix_space=True,
            trust_remote_code=True
        )
        print(f"Loaded tokenizer")
    
        bos_token = tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None else tokenizer.cls_token_id
        eos_token = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None else tokenizer.sep_token_id
        
        if bos_token is None:
            raise ValueError("Tokenizer must have BOS or CLS token")
        
        print(f"BOS token ID: {bos_token}")
        print(f"EOS token ID: {eos_token}")
        print()
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        raise
    
    subset_folders = discover_subset_folders(input_path)
    if not subset_folders:
        raise ValueError(f"No subset folders found in {input_path}")
    
    if subset_filter:
        subset_folders = [f for f in subset_folders if f.name == subset_filter]
        if not subset_folders:
            raise ValueError(f"Subset folder '{subset_filter}' not found in {input_path}")
        print(f"Processing only subset: {subset_filter}")
    else:
        print(f"Found {len(subset_folders)} subset folders")
    
    if resume:
        complete_count = 0
        resumable_count = 0
        new_count = 0
        
        for subset_folder in subset_folders:
            output_subset = output_path / subset_folder.name
            total_samples = count_subset_samples(subset_folder)
            can_resume, samples_to_skip, status_msg = check_resume_status(output_subset, total_samples)
            
            if status_msg == "complete":
                complete_count += 1
            elif can_resume:
                resumable_count += 1
            else:
                new_count += 1
        
        print(f"Resume summary: {complete_count} complete, {resumable_count} resumable, {new_count} new")
    
    print()
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    overall_stats = {
        'total_input_samples': 0,
        'total_output_chunks': 0,
        'total_tokens_written': 0,
        'total_duplicated_tokens': 0,
        'total_tokens_skipped': 0,
        'subset_stats': {},
        'config': {
            'chunk_size': chunk_size,
            'min_chunk_size': min_chunk_size,
            'always_skip_size': always_skip_size,
            'backfill': backfill,
            'backfill_no_duplicates': backfill_no_duplicates,
            'add_eos_token': add_eos_token,
            'seed': seed,
            'num_workers': num_workers
        }
    }
    
    worker_args = []
    for subset_folder in subset_folders:
        output_subset = output_path / subset_folder.name
        worker_args.append((
            subset_folder,
            output_subset,
            tokenizer_path,
            chunk_size,
            min_chunk_size,
            always_skip_size,
            backfill,
            backfill_no_duplicates,
            add_eos_token,
            bos_token,
            eos_token,
            batch_size,
            compression,
            resume,
            seed
        ))
    
    print(f"Processing {len(subset_folders)} subsets with {num_workers} workers...")
    print()
    
    if num_workers == 1:
        results = []
        for args in tqdm(worker_args, desc="Processing subsets", unit="subset"):
            results.append(chunk_subset_worker(args))
    else:
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(chunk_subset_worker, worker_args),
                total=len(worker_args),
                desc="Processing subsets",
                unit="subset"
            ))
    
    for subset_name, stats in results:
        if 'error' in stats:
            print(f"Failed to process {subset_name}: {stats['error']}")
        else:
            if not stats.get('skipped', False):
                overall_stats['total_input_samples'] += stats['total_input_samples']
                overall_stats['total_output_chunks'] += stats['total_output_chunks']
                overall_stats['total_tokens_written'] += stats['total_tokens_written']
                overall_stats['total_duplicated_tokens'] += stats['total_duplicated_tokens']
                overall_stats['total_tokens_skipped'] += stats['total_tokens_skipped']
        
        overall_stats['subset_stats'][subset_name] = {
            k: v for k, v in stats.items() if k != 'distribution'
        }
    
    stats_file = output_path / "overall_chunking_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(overall_stats, f, indent=2)
    
    print()
    print("="*60)
    print("Chunking complete!")
    print("="*60)
    print(f"Input samples: {overall_stats['total_input_samples']:,}")
    print(f"Output chunks: {overall_stats['total_output_chunks']:,}")
    print(f"Tokens written: {overall_stats['total_tokens_written']:,}")
    print(f"Tokens duplicated: {overall_stats['total_duplicated_tokens']:,}")
    print(f"Tokens skipped: {overall_stats['total_tokens_skipped']:,}")
    print(f"Output directory: {output_path.absolute()}")
    print()
    
    return overall_stats


def main():
    parser = argparse.ArgumentParser(
        description="Chunk tokenized MDS datasets into fixed-length sequences"
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
        help='HuggingFace repository ID to download'
    )
    
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument(
        '--output_dir',
        type=str,
        help='Local output directory for chunked dataset'
    )
    output_group.add_argument(
        '--output_repo',
        type=str,
        help='HuggingFace repository ID for upload'
    )
    
    parser.add_argument(
        '--tokenizer_path',
        type=str,
        required=True,
        help='Path to tokenizer (same as used for tokenization)'
    )
    
    parser.add_argument(
        '--chunk_size',
        type=int,
        required=True,
        help='Target chunk size (e.g., 1024, 8192)'
    )
    parser.add_argument(
        '--min_chunk_size',
        type=int,
        default=512,
        help='Minimum chunk size to keep (default: 512)'
    )
    parser.add_argument(
        '--always_skip_size',
        type=int,
        default=None,
        help='Always skip chunks smaller than this (default: 128 for 1K, 32 for 8K)'
    )
    parser.add_argument(
        '--no_backfill',
        action='store_true',
        help='Disable backfill of short chunks'
    )
    parser.add_argument(
        '--backfill_duplicates',
        action='store_true',
        help='Allow duplicates in backfill (not recommended)'
    )
    parser.add_argument(
        '--add_eos_token',
        action='store_true',
        help='Add EOS token to end of each sequence'
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
        '--resume',
        action='store_true',
        help='Resume from existing output (skip completed subsets)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: auto-detect CPU cores - 1)'
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
        default='./temp_chunking',
        help='Temporary directory for processing (default: ./temp_chunking)'
    )
    parser.add_argument(
        '--upload_workers',
        type=int,
        default=16,
        help='Number of parallel workers for HuggingFace upload (default: 16)'
    )
    parser.add_argument(
        '--upload_report_every',
        type=int,
        default=30,
        help='Print upload progress report every N seconds (default: 30)'
    )
    parser.add_argument(
        '--subset_filter',
        type=str,
        default=None,
        help='Process only this specific subset folder (e.g., 004_00004)'
    )
    
    args = parser.parse_args()
    
    if args.always_skip_size is None:
        if args.chunk_size >= 8192:
            args.always_skip_size = 32
        else:
            args.always_skip_size = 128
    
    compression = None if args.compression == 'none' else args.compression
    
    if args.hf_repo:
        print(f"Downloading from HuggingFace: {args.hf_repo}")
        cache_dir = Path(args.hf_cache_dir) if args.hf_cache_dir else None
        subset_folders = [args.subset_filter] if args.subset_filter else None
        input_path = download_from_hf(
            repo_id=args.hf_repo,
            cache_dir=cache_dir,
            token=args.hf_token,
            subset_folders=subset_folders
        )
        print()
    else:
        input_path = Path(args.input_dir)
        if not input_path.exists():
            parser.error(f"Input directory does not exist: {input_path}")
    
    if args.output_repo:
        output_path = Path(args.temp_dir) / f"chunked-{args.chunk_size}"
        output_path.mkdir(parents=True, exist_ok=True)
        upload_to_hf_after = True
    else:
        output_path = Path(args.output_dir)
        upload_to_hf_after = False
    
    try:
        stats = chunk_dataset(
            input_path=input_path,
            output_path=output_path,
            tokenizer_path=args.tokenizer_path,
            chunk_size=args.chunk_size,
            min_chunk_size=args.min_chunk_size,
            always_skip_size=args.always_skip_size,
            backfill=not args.no_backfill,
            backfill_no_duplicates=not args.backfill_duplicates,
            add_eos_token=args.add_eos_token,
            batch_size=args.batch_size,
            compression=compression,
            resume=args.resume,
            seed=args.seed,
            num_workers=args.num_workers,
            subset_filter=args.subset_filter
        )
        
        if upload_to_hf_after:
            upload_to_hf(
                local_path=output_path,
                repo_id=args.output_repo,
                token=args.hf_token,
                private=args.private,
                commit_message=f"Chunked dataset (chunk_size={args.chunk_size}) from {args.hf_repo or args.input_dir}",
                num_workers=args.upload_workers,
                print_report_every=args.upload_report_every
            )
        
        return 0
        
    except Exception as e:
        print(f"\nError during chunking: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

