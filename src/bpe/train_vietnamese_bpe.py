#!/usr/bin/env python3
"""
Train Vietnamese BPE Tokenizer

This script trains a BPE tokenizer optimized for Vietnamese text using the bpeasy library.
It downloads Vietnamese datasets from HuggingFace, preprocesses the data, and trains
a tokenizer with Vietnamese-specific regex patterns.

Usage:
    # Train with default settings (32k vocab, max length 16)
    python src/bpe/train_vietnamese_bpe.py
    
    # Train with custom settings
    python src/bpe/train_vietnamese_bpe.py --vocab_size 50000 --max_token_length 32
    
    # Verify existing tokenizer
    python src/bpe/train_vietnamese_bpe.py --verify_only --tokenizer_path ./output/vietnamese_bpe.json
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.bpe.vietnamese_word_tokenizer import get_vietnamese_bpe_regex
from src.bpe.bpeasy.tokenizer import BPEasyTokenizer
from src.utils.mds_helpers import (
    discover_subset_folders,
    load_mds_subset,
    count_total_samples
)


DEFAULT_DATASETS = [
    "QuangDuy/FineWiki-mds",
    "QuangDuy/FineWeb2-vie-mds"
]


def resolve_dataset_path(dataset_path: Path) -> Path:
    """
    Resolve dataset path, handling HuggingFace cache structure.
    
    If the path contains a HuggingFace cache structure (datasets--org--name/snapshots/hash/),
    return the snapshot directory. Otherwise, return the original path.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        Resolved path to actual MDS data
    """
    hf_cache_dirs = list(dataset_path.glob("datasets--*"))
    
    if hf_cache_dirs:
        cache_dir = hf_cache_dirs[0]
        snapshots_dir = cache_dir / "snapshots"
        
        if snapshots_dir.exists():
            snapshot_dirs = list(snapshots_dir.iterdir())
            if snapshot_dirs:
                print(f"  Resolved HuggingFace cache: {dataset_path.name} -> {snapshot_dirs[0].name}")
                return snapshot_dirs[0]
    
    return dataset_path


def text_iterator_from_mds(
    dataset_paths: list[Path],
    batch_size: int = 1000,
    max_samples: int = None
) -> Iterator[str]:
    """
    Create a text iterator from MDS datasets.
    
    Args:
        dataset_paths: List of paths to MDS datasets
        batch_size: Number of samples to load at once
        max_samples: Maximum number of samples to process (None = all)
        
    Yields:
        Text strings from the datasets
    """
    total_processed = 0
    
    total_to_process = max_samples if max_samples else sum(
        count_total_samples(p) for p in dataset_paths
    )
    
    pbar = tqdm(total=total_to_process, desc="Processing samples", unit="samples")
    
    for dataset_path in dataset_paths:
        subset_folders = discover_subset_folders(dataset_path)
        
        for subset_folder in subset_folders:
            pbar.set_postfix_str(f"{dataset_path.name}/{subset_folder.name}")
            
            for batch in load_mds_subset(subset_folder, batch_size=batch_size):
                for sample in batch:
                    if max_samples and total_processed >= max_samples:
                        pbar.close()
                        return
                    
                    yield sample['text']
                    total_processed += 1
                    pbar.update(1)
                
                del batch
                gc.collect()
    
    pbar.close()


def count_dataset_samples(dataset_paths: list[Path]) -> int:
    """
    Count total samples across multiple datasets.
    
    Args:
        dataset_paths: List of paths to MDS datasets
        
    Returns:
        Total number of samples
    """
    total = 0
    for dataset_path in dataset_paths:
        count = count_total_samples(dataset_path)
        total += count
        print(f"  {dataset_path.name}: {count:,} samples")
    return total


def train_vietnamese_tokenizer(
    dataset_paths: list[Path],
    vocab_size: int = 32000,
    max_token_length: int = 16,
    output_dir: Path = Path("./output"),
    batch_size: int = 1000,
    max_samples: int = None,
    name: str = "vietnamese_bpe",
    special_tokens: list[str] = None
) -> BPEasyTokenizer:
    """
    Train a Vietnamese BPE tokenizer.
    
    Args:
        dataset_paths: List of paths to MDS datasets
        vocab_size: Target vocabulary size
        max_token_length: Maximum token length
        output_dir: Directory to save the tokenizer
        batch_size: Batch size for data loading
        max_samples: Maximum samples to use for training (None = all)
        name: Name for the tokenizer
        special_tokens: List of special tokens (e.g., ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        
    Returns:
        Trained BPEasyTokenizer
    """
    if special_tokens is None:
        special_tokens = []
    
    print("=" * 70)
    print("Training Vietnamese BPE Tokenizer")
    print("=" * 70)
    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Max token length: {max_token_length}")
    print(f"Output directory: {output_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Max samples: {max_samples if max_samples else 'All'}")
    if special_tokens:
        print(f"Special tokens: {special_tokens}")
    print()
    
    regex_pattern = get_vietnamese_bpe_regex()
    print("Vietnamese BPE Regex Pattern:")
    print(regex_pattern)
    print()
    
    print("Counting samples in datasets...")
    total_samples = count_dataset_samples(dataset_paths)
    if max_samples:
        samples_to_use = min(total_samples, max_samples)
    else:
        samples_to_use = total_samples
    print(f"Total samples available: {total_samples:,}")
    print(f"Samples to use for training: {samples_to_use:,}")
    print()
    
    print("Creating text iterator...")
    text_iter = text_iterator_from_mds(
        dataset_paths=dataset_paths,
        batch_size=batch_size,
        max_samples=max_samples
    )
    
    print()
    print("=" * 70)
    print("Starting BPE Training...")
    print("=" * 70)
    start_time = time.time()
    
    tokenizer = BPEasyTokenizer.train(
        iterator=text_iter,
        vocab_size=vocab_size,
        max_token_length=max_token_length,
        regex_pattern=regex_pattern,
        special_tokens=special_tokens,
        fill_to_nearest_multiple_of_eight=False,
        name=name,
        batch_size=batch_size
    )
    
    training_time = time.time() - start_time
    
    print()
    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Vocabulary size: {len(tokenizer):,}")
    print()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    bpeasy_path = output_dir / f"{name}.json"
    tokenizer.save(str(bpeasy_path))
    print(f"Saved BPEasy format: {bpeasy_path}")
    
    hf_json_path = output_dir / f"{name}_hf.json"
    tokenizer.export_to_huggingface_format(str(hf_json_path))
    print(f"Saved HuggingFace tokenizer.json: {hf_json_path}")
    
    if special_tokens:
        try:
            from transformers import PreTrainedTokenizerFast
            special_tokens_map = {}
            if "[UNK]" in special_tokens:
                special_tokens_map["unk_token"] = "[UNK]"
            if "[CLS]" in special_tokens:
                special_tokens_map["cls_token"] = "[CLS]"
            if "[SEP]" in special_tokens:
                special_tokens_map["sep_token"] = "[SEP]"
            if "[PAD]" in special_tokens:
                special_tokens_map["pad_token"] = "[PAD]"
            if "[MASK]" in special_tokens:
                special_tokens_map["mask_token"] = "[MASK]"
            
            hf_tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(hf_json_path),
                **special_tokens_map
            )
            
            hf_tokenizer_dir = output_dir / f"{name}_hf_tokenizer"
            hf_tokenizer.save_pretrained(str(hf_tokenizer_dir))
            print(f"Saved HuggingFace PreTrainedTokenizer: {hf_tokenizer_dir}")
            
        except ImportError:
            print("transformers library not installed, skipping PreTrainedTokenizer export")
            print("  Install with: pip install transformers")
    
    metadata = {
        'name': name,
        'vocab_size': len(tokenizer),
        'max_token_length': max_token_length,
        'regex_pattern': regex_pattern,
        'training_samples': samples_to_use,
        'training_time_seconds': training_time,
        'datasets': [str(p) for p in dataset_paths],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    metadata_path = output_dir / f"{name}_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata: {metadata_path}")
    
    print()
    return tokenizer


def verify_tokenizer(tokenizer_path: Path):
    """
    Verify a trained tokenizer.
    
    Args:
        tokenizer_path: Path to the tokenizer JSON file
    """
    print("=" * 70)
    print("Verifying Vietnamese BPE Tokenizer")
    print("=" * 70)
    print(f"Tokenizer path: {tokenizer_path}")
    print()
    
    tokenizer = BPEasyTokenizer.from_file(str(tokenizer_path))
    
    print(f"Tokenizer loaded successfully")
    print(f"  Vocabulary size: {len(tokenizer):,}")
    print(f"  Regex pattern: {tokenizer.regex_pattern}")
    print()
    
    test_texts = [
        "Tp. Hồ Chí Minh     là thành phố lớn nhất Việt Nam.",
        "Đại học Quốc gia TP.HCM (ĐHQG-HCM) là một trong những trường đại học hàng đầu.",
        "ThS. Nguyễn Văn A và Dr. Trần Thị B đang nghiên cứu về trí tuệ nhân tạo.",
        "Email: contact@example.com, Website: https://example.com",
        "Số điện thoại: 028.1234.5678 hoặc 028-1234-5678",
    ]
    
    print("Testing on sample Vietnamese texts:")
    print("-" * 70)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}:")
        print(f"Input:  {text}")
        print()
        
        token_ids = tokenizer.encode(text)
        
        attention_mask = [1] * len(token_ids)
        
        encoding = {
            'input_ids': token_ids,
            'attention_mask': attention_mask
        }
        
        print(encoding)
        
        decoded = tokenizer.decode(token_ids)
        print(decoded)
        
        tokens = [tokenizer.decode([tid]) for tid in token_ids]
        print(tokens)
        
        print()
        
        if decoded == text:
            print("Roundtrip successful")
        else:
            print("Roundtrip failed!")
            print(f"  Expected: {text}")
            print(f"  Got:      {decoded}")
    
    print()
    print("=" * 70)
    print("Verification Complete!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Train Vietnamese BPE tokenizer"
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data',
        help='Directory containing downloaded datasets (default: ./data)'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=DEFAULT_DATASETS,
        help=f'Dataset names to use (default: {" ".join([d.split("/")[-1] for d in DEFAULT_DATASETS])})'
    )
    parser.add_argument(
        '--vocab_size',
        type=int,
        default=32000,
        help='Target vocabulary size (default: 32000)'
    )
    parser.add_argument(
        '--max_token_length',
        type=int,
        default=16,
        help='Maximum token length (default: 16)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output',
        help='Output directory for trained tokenizer (default: ./output)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help='Batch size for data loading (default: 1000)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum samples to use for training (default: all)'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='vietnamese_bpe',
        help='Name for the tokenizer (default: vietnamese_bpe)'
    )
    parser.add_argument(
        '--verify_only',
        action='store_true',
        help='Only verify an existing tokenizer'
    )
    parser.add_argument(
        '--tokenizer_path',
        type=str,
        default=None,
        help='Path to tokenizer for verification (required with --verify_only)'
    )
    parser.add_argument(
        '--special_tokens',
        type=str,
        nargs='*',
        default=None,
        help='Special tokens to add (e.g., --special_tokens "[UNK]" "[CLS]" "[SEP]" "[PAD]" "[MASK]")'
    )
    parser.add_argument(
        '--use_bert_tokens',
        action='store_true',
        help='Use BERT-style special tokens: [UNK], [CLS], [SEP], [PAD], [MASK]'
    )
    
    args = parser.parse_args()
    
    if args.verify_only:
        if not args.tokenizer_path:
            parser.error("--tokenizer_path is required with --verify_only")
        
        tokenizer_path = Path(args.tokenizer_path)
        if not tokenizer_path.exists():
            parser.error(f"Tokenizer file not found: {tokenizer_path}")
        
        verify_tokenizer(tokenizer_path)
        return 0
    
    data_dir = Path(args.data_dir)
    dataset_paths = []
    
    for dataset in args.datasets:
        dataset_name = dataset.split('/')[-1]
        dataset_path = data_dir / dataset_name
        
        if not dataset_path.exists():
            print(f"Error: Dataset not found: {dataset_path}")
            print(f"Please run download_datasets.py first to download the datasets.")
            return 1
        
        resolved_path = resolve_dataset_path(dataset_path)
        dataset_paths.append(resolved_path)
    
    special_tokens = None
    if args.use_bert_tokens:
        special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    elif args.special_tokens:
        special_tokens = args.special_tokens
    
    try:
        tokenizer = train_vietnamese_tokenizer(
            dataset_paths=dataset_paths,
            vocab_size=args.vocab_size,
            max_token_length=args.max_token_length,
            output_dir=Path(args.output_dir),
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            name=args.name,
            special_tokens=special_tokens
        )
        
        print()
        tokenizer_path = Path(args.output_dir) / f"{args.name}.json"
        verify_tokenizer(tokenizer_path)
        
        print()
        print("Training and verification complete!")
        print(f"  Tokenizer saved to: {Path(args.output_dir).absolute()}")
        
        return 0
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
