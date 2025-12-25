# Copyright (c) 2025, EleutherAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Train GPT-NeoX-style BPE tokenizer with consistent space splitting.
Supports both JSONL files and MDS datasets from HuggingFace.
"""

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from tokenizers.normalizers import NFC
from typing import List, Optional

from glob import glob
import os
import json
import argparse
from pathlib import Path
import sys


def load_jsonl(input_path, quiet=True) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.rstrip("\n|\r")))
    if not quiet:
        print("Loaded {} records from {}".format(len(data), input_path))
    return data


def json_iterator(input_dir, text_key="text"):
    """Iterator for reading text from JSONL files."""
    all_jsonls = glob(f"{input_dir}/*.jsonl") + glob(f"{input_dir}/*.json")
    for j in all_jsonls:
        data = load_jsonl(j)
        for doc in data:
            yield doc[text_key]


def limited_iterator(iterator, max_lines=None):
    """
    Wrapper to limit the number of lines from an iterator.
    
    Args:
        iterator: Source iterator
        max_lines: Maximum number of lines to yield (None = unlimited)
        
    Yields:
        Items from iterator up to max_lines
    """
    if max_lines is None:
        yield from iterator
    else:
        count = 0
        for item in iterator:
            if count >= max_lines:
                print(f"\nReached max_lines limit: {max_lines}")
                break
            yield item
            count += 1


def mds_text_iterator(
    dataset_paths: List[str], 
    cache_dir: str = "data", 
    text_key: str = "text",
    batch_size: int = 50000,
    progress_interval: int = 100000,
    max_lines: Optional[int] = None
):
    """
    Yield text from MDS datasets for tokenizer training.
    
    Automatically detects local MDS datasets or downloads from HuggingFace if needed.
    
    Args:
        dataset_paths: List of HuggingFace dataset repo IDs or local dataset names
                      Examples: ["QuangDuy/FineWiki-mds"] or ["FineWiki-mds"]
        cache_dir: Local cache directory for datasets
        text_key: Key to extract text from samples
        batch_size: Number of samples to load at once (default: 50000, higher = faster but more memory)
        progress_interval: Print progress every N samples (default: 100000)
        max_lines: Maximum number of lines to yield (default: None, unlimited)
        
    Yields:
        Text strings from the datasets
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.mds_helpers import download_from_hf, discover_subset_folders, NoStreamingMDSDataset
    
    print(f"Loading MDS datasets: {dataset_paths}")
    if max_lines:
        print(f"Maximum lines to process: {max_lines:,}")
    
    total_lines_yielded = 0
    
    for dataset_id in dataset_paths:
        print(f"\nProcessing dataset: {dataset_id}")
        
        local_path = None
        cache_path = Path(cache_dir)
        dataset_name = dataset_id.split('/')[-1]  
        direct_path = cache_path / dataset_name
        if direct_path.exists():
            nested_path = direct_path / f"datasets--{dataset_id.replace('/', '--')}" / "snapshots"
            if nested_path.exists():
                snapshot_dirs = [d for d in nested_path.iterdir() if d.is_dir()]
                if snapshot_dirs:
                    local_path = snapshot_dirs[0]
                    print(f"Found local dataset at: {local_path}")
        
        if local_path is None and direct_path.exists():
            subset_folders = discover_subset_folders(direct_path)
            if subset_folders:
                print(f"Found local dataset with subsets at: {direct_path}")
                local_path = direct_path
        
        if local_path is None:
            hf_cache_pattern = cache_path / f"datasets--{dataset_id.replace('/', '--')}"
            if hf_cache_pattern.exists():
                snapshots_dir = hf_cache_pattern / "snapshots"
                if snapshots_dir.exists():
                    snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
                    if snapshot_dirs:
                        local_path = snapshot_dirs[0]
                        print(f"Found cached HuggingFace dataset at: {local_path}")
        
        if local_path is None:
            print(f"Local dataset not found, downloading from HuggingFace...")
            local_path = download_from_hf(dataset_id, cache_dir=cache_path)
        
        subset_folders = discover_subset_folders(local_path)
        
        if not subset_folders:
            if (local_path / "index.json").exists():
                print(f"Using dataset root directly (no subsets): {local_path}")
                subset_folders = [local_path]
            else:
                print(f"Warning: No valid MDS subsets found in {local_path}")
                continue
        
        print(f"Found {len(subset_folders)} subset folder(s)")
        
        # For FineWeb2-vie-mds, reverse the order to start from 004_00005 going backwards
        if "FineWeb2-vie" in dataset_id or "fineweb2-vie" in dataset_id.lower():
            print(f"  Detected FineWeb2-vie dataset - reversing subset order (starting from last subset)")
            subset_folders = sorted(subset_folders, reverse=True)
            print(f"  Processing order: {subset_folders[0].name} -> ... -> {subset_folders[-1].name}")
        
        for i, subset_folder in enumerate(subset_folders):
            print(f"  Processing subset {i+1}/{len(subset_folders)}: {subset_folder.name}")
            
            try:
                from utils.mds_helpers import load_mds_subset
                
                sample_count = 0
                for batch in load_mds_subset(subset_folder, batch_size=batch_size):
                    texts = [sample[text_key] for sample in batch if text_key in sample]
                    
                    for text in texts:
                        if max_lines and total_lines_yielded >= max_lines:
                            print(f"  Reached max_lines limit: {max_lines:,}")
                            print(f"  Total lines yielded: {total_lines_yielded:,}")
                            return
                        yield text
                        total_lines_yielded += 1
                    
                    sample_count += len(texts)
                    
                    if sample_count % progress_interval == 0:
                        print(f"    Processed {sample_count} samples from {subset_folder.name} (Total: {total_lines_yielded:,})")
                
                print(f"  Completed subset {subset_folder.name}: {sample_count} samples (Total: {total_lines_yielded:,})")
                
            except Exception as e:
                print(f"  Error processing subset {subset_folder.name}: {e}")
                import traceback
                traceback.print_exc()
                continue


def generate_bert_special_tokens() -> List[str]:
    """
    Generate BERT-style special tokens only.
    
    Returns:
        List of BERT special tokens: [CLS], [MASK], [PAD], [SEP], [UNK]
    """
    return ["[CLS]", "[MASK]", "[PAD]", "[SEP]", "[UNK]"]


def add_repeated_space_tokens(tokenizer: Tokenizer, max_consecutive_spaces: int = 24):
    """
    Add repeated space tokens to tokenizer AFTER training (GPT-NeoX approach).
    
    Tokens are added with IDs starting from vocab_size, format:
    {
      "id": vocab_size + i,
      "content": " " * (i+2),  // raw spaces: "  ", "   ", "    ", ..., (24 spaces)
      "lstrip": false,
      "rstrip": false,
      "normalized": true,
      "single_word": false,
      "special": false
    }
    
    Note: Starts from 2 spaces (not 1) because single space is handled naturally by BPE
    
    Args:
        tokenizer: The trained tokenizer
        max_consecutive_spaces: Maximum number of consecutive spaces (2-24, default: 24)
    """
    space_tokens = [" " * i for i in range(2, max_consecutive_spaces + 1)]
    
    current_vocab_size = tokenizer.get_vocab_size()
    
    print(f"\nAdding repeated space tokens after training (GPT-NeoX style):")
    print(f"  Current vocab size: {current_vocab_size}")
    print(f"  Adding {len(space_tokens)} space tokens (2-{max_consecutive_spaces} spaces)")
    print(f"  New IDs: {current_vocab_size} to {current_vocab_size + len(space_tokens) - 1}")
    
    tokenizer.add_tokens(space_tokens)
    
    final_vocab_size = tokenizer.get_vocab_size()
    print(f" Successfully added space tokens")
    print(f"  Final vocab size: {final_vocab_size}")
   
    return space_tokens


def create_neox_pretokenizer():
    """
    Create GPT-NeoX-style pre-tokenizer with consistent space splitting.
    
    Uses ByteLevel pre-tokenizer with:
    - add_prefix_space=True: ensures the first word is treated consistently with subsequent words
    - This adds a space to the beginning of the first word if not present, following HuggingFace best practices
    
    Returns:
        Pre-tokenizer for consistent space handling
    """
    return pre_tokenizers.ByteLevel(add_prefix_space=True)


def save_as_hf_tokenizer(tokenizer_path: str, output_dir: str):
    """
    Convert trained tokenizer to HuggingFace format with special token mapping.
    
    Args:
        tokenizer_path: Path to trained tokenizer JSON file
        output_dir: Output directory for HuggingFace tokenizer files
    """
    from transformers import PreTrainedTokenizerFast
    
    print(f"\nConverting tokenizer to HuggingFace format...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        cls_token="[CLS]",
        mask_token="[MASK]",
        pad_token="[PAD]",
        sep_token="[SEP]",
        unk_token="[UNK]"
    )
    
    os.makedirs(output_dir, exist_ok=True)
    hf_tokenizer.save_pretrained(output_dir)
    print(f"HuggingFace tokenizer saved at {output_dir}")


def train_tokenizer(
    input_dir: Optional[str] = None,
    mds_datasets: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    tokenizer_type: str = "BPE",
    vocab_size: int = 32000,
    max_token_length: int = 64,
    max_consecutive_spaces: int = 24,
    cache_dir: str = "data",
    text_key: str = "text",
    hf_output_dir: Optional[str] = None,
    mds_batch_size: int = 2000000,
    progress_interval: int = 1000000,
    max_lines: Optional[int] = None
):
    """
    Train GPT-NeoX-style BPE tokenizer with consistent space handling.
    
    This implementation follows GPT-NeoX's approach with key improvements:
    1. Train on diverse data (supports both JSONL and MDS datasets)
    2. Consistent space splitting (ByteLevel with add_prefix_space=True)
    3. BERT-style special tokens
    4. Repeated space tokens added AFTER training (GPT-NeoX style)
    
    Args:
        input_dir: Directory containing JSONL files (optional)
        mds_datasets: List of HuggingFace MDS dataset repo IDs (optional)
        save_path: Path to save tokenizer JSON file
        tokenizer_type: Type of tokenizer to train (currently only BPE)
        vocab_size: Vocabulary size (default: 32000)
        max_token_length: Maximum token length for BPE (default: 64)
        max_consecutive_spaces: Number of space tokens to add after training (2-24, default: 24)
        cache_dir: Cache directory for MDS datasets (default: "data")
        text_key: Key to extract text from samples (default: "text")
        hf_output_dir: Optional directory to save HuggingFace format tokenizer
        mds_batch_size: Batch size for loading MDS data (default: 2000000, higher = faster but more memory)
        progress_interval: Print progress every N samples (default: 1000000)
        max_lines: Maximum number of lines to train on (default: None, unlimited)
    """
    
    if input_dir is None and mds_datasets is None:
        raise ValueError("Either input_dir or mds_datasets must be provided")
    
    print(f"\n{'='*70}")
    print(f"Training GPT-NeoX-style BPE Tokenizer (ByteLevel)")
    print(f"{'='*70}")
    print(f"Vocab size: {vocab_size}")
    print(f"Max token length: {max_token_length}")
    if max_lines:
        print(f"Max training lines: {max_lines:,}")
    if max_consecutive_spaces > 0:
        print(f"Space tokens to add: {max_consecutive_spaces - 1} (2-{max_consecutive_spaces} spaces)")
    else:
        print(f"Space tokens: Disabled (no repeated space tokens will be added)")
    print(f"{'='*70}\n")

    if tokenizer_type == "BPE":
        model = models.BPE()
    else:
        raise NotImplementedError(f"Tokenizer type {tokenizer_type} not implemented")
    
    tokenizer = Tokenizer(model)

    print("Setting up GPT-NeoX-style pre-tokenizer (ByteLevel, consistent space splitting)...")
    tokenizer.pre_tokenizer = create_neox_pretokenizer()
    
    tokenizer.decoder = decoders.ByteLevel()
    
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    
    tokenizer.normalizer = NFC()

    special_tokens = generate_bert_special_tokens()
    print(f"Special tokens for training: {len(special_tokens)} tokens")
    print(f"  - BERT tokens: [CLS], [MASK], [PAD], [SEP], [UNK]")
    if max_consecutive_spaces > 0:
        print(f"  - Space tokens ({max_consecutive_spaces - 1} tokens: 2-{max_consecutive_spaces} spaces) will be added after training")
    else:
        print(f"  - Space tokens: Disabled")

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        max_token_length=max_token_length,
        show_progress=True,
        byte_fallback=True
    )

    print("\nStarting tokenizer training...")
    if mds_datasets:
        print(f"Training from MDS datasets: {mds_datasets}")
        iterator = mds_text_iterator(
            mds_datasets, 
            cache_dir=cache_dir, 
            text_key=text_key,
            batch_size=mds_batch_size,
            progress_interval=progress_interval,
            max_lines=max_lines
        )
    else:
        print(f"Training from JSONL files in: {input_dir}")
        base_iterator = json_iterator(input_dir, text_key=text_key)
        iterator = limited_iterator(base_iterator, max_lines=max_lines)
    
    # Train with length hint for better progress tracking (multiprocessing is automatic)
    tokenizer.train_from_iterator(
        iterator, 
        trainer=trainer,
        length=max_lines if max_lines else None
    )
    
    if max_consecutive_spaces > 0:
        add_repeated_space_tokens(tokenizer, max_consecutive_spaces)

    if save_path:
        print(f"\nSaving tokenizer to: {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        tokenizer.save(save_path, pretty=True)
        print(f"Tokenizer saved successfully")
        
        if hf_output_dir:
            save_as_hf_tokenizer(save_path, hf_output_dir)
    
    print(f"\n{'='*70}")
    print(f"Training completed successfully!")
    print(f"{'='*70}\n")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Train GPT-NeoX-style BPE tokenizer with consistent space splitting. "
        "Supports JSONL files and MDS datasets from HuggingFace."
    )
    parser.add_argument(
        "--json_input_dir",
        type=str,
        default=None,
        help="Path to folder containing tokenizer training data in JSONL format",
    )
    parser.add_argument(
        "--mds_datasets",
        type=str,
        nargs="+",
        default=None,
        help="List of HuggingFace MDS dataset repo IDs (e.g., QuangDuy/FineWiki-mds)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="data",
        help="Cache directory for MDS datasets (default: data)",
    )
    parser.add_argument(
        "--text_key",
        type=str,
        default="text",
        help="Key to extract text from samples (default: text)",
    )
    parser.add_argument(
        "--tokenizer_output_path",
        type=str,
        required=True,
        help="Path to which your trained tokenizer will be saved (should end in .json)",
    )
    parser.add_argument(
        "--hf_output_dir",
        type=str,
        default=None,
        help="Optional directory to save HuggingFace format tokenizer",
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        help="Type of tokenizer to train, currently only BPE is supported",
        choices=["BPE"],
        default="BPE",
    )
    parser.add_argument(
        "-v",
        "--vocab_size",
        help="Vocabulary size of tokenizer (default: 32000)",
        type=int,
        default=32000,
    )
    parser.add_argument(
        "--max_token_length",
        type=int,
        default=64,
        help="Maximum token length for BPE (default: 64)",
    )
    parser.add_argument(
        "--max_consecutive_spaces",
        type=int,
        default=0,
        help="Maximum number of spaces for repeated space tokens (2-24, default: 0). Set to 0 to disable space tokens.",
    )
    parser.add_argument(
        "--mds_batch_size",
        type=int,
        default=2000000,
        help="Batch size for loading MDS data (default: 2000000, higher = faster but more memory).",
    )
    parser.add_argument(
        "--progress_interval",
        type=int,
        default=1000000,
        help="Print progress every N samples (default: 1000000).",
    )
    parser.add_argument(
        "--max_lines",
        type=int,
        default=None,
        help="Maximum number of lines to train on (default: None, unlimited).",
    )
    
    args_parsed = parser.parse_args(input_args)
    
    if args_parsed.json_input_dir is None and args_parsed.mds_datasets is None:
        parser.error("Either --json_input_dir or --mds_datasets must be provided")
    
    return args_parsed


def main(args):
    """Main entry point for tokenizer training."""
    train_tokenizer(
        input_dir=args.json_input_dir,
        mds_datasets=args.mds_datasets,
        save_path=args.tokenizer_output_path,
        tokenizer_type=args.tokenizer_type,
        vocab_size=args.vocab_size,
        max_token_length=args.max_token_length,
        max_consecutive_spaces=args.max_consecutive_spaces,
        cache_dir=args.cache_dir,
        text_key=args.text_key,
        hf_output_dir=args.hf_output_dir,
        mds_batch_size=args.mds_batch_size,
        progress_interval=args.progress_interval,
        max_lines=args.max_lines
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
