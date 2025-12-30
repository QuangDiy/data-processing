#!/usr/bin/env python3
"""
Format Tokenizer to BERT Style

This script formats a BPE tokenizer to match BERT's style:
1. Adds unused tokens ([unused1], [unused2], ...) to make vocab size a multiple of 64
2. Adds BERT-style TemplateProcessing for [CLS] and [SEP] tokens
3. Saves the formatted tokenizer
4. Optionally pushes to HuggingFace Hub

Usage:
    # Format tokenizer
    python src/bpe/format_bert_tokenizer.py --input_dir output/tokenizer_15m/hf_15m --output_dir output/tokenizer_15m_bert
    
    # Format and push to HuggingFace
    python src/bpe/format_bert_tokenizer.py --input_dir output/tokenizer_15m/hf_15m --output_dir output/tokenizer_15m_bert --push_to_hub --repo_id your-username/tokenizer-name
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Fix UTF-8 encoding on Windows
os.environ["PYTHONUTF8"] = "1"
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from tokenizers import Tokenizer, processors
from transformers import PreTrainedTokenizerFast


def add_unused_tokens(
    tokenizer: Tokenizer, 
    target_multiple: int = 64,
    num_unused: int = 0
) -> Tokenizer:
    """
    Add unused tokens to make vocab size a multiple of target_multiple.
    
    Args:
        tokenizer: Input tokenizer
        target_multiple: Target multiple for vocab size (default: 64)
        num_unused: Number of unused tokens to add (0 = auto-calculate to reach multiple)
        
    Returns:
        Tokenizer with added unused tokens
    """
    current_vocab_size = tokenizer.get_vocab_size()
    
    if num_unused > 0:
        # Add specific number of unused tokens
        tokens_to_add = num_unused
        new_vocab_size = current_vocab_size + tokens_to_add
        
        print(f"Current vocab size: {current_vocab_size}")
        print(f"Adding {tokens_to_add} unused tokens to reach {new_vocab_size}")
    else:
        # Calculate how many tokens to add to reach multiple
        remainder = current_vocab_size % target_multiple
        if remainder == 0:
            print(f"Vocab size {current_vocab_size} is already a multiple of {target_multiple}")
            print("No unused tokens will be added. Use --num_unused to add specific number.")
            return tokenizer
        
        tokens_to_add = target_multiple - remainder
        new_vocab_size = current_vocab_size + tokens_to_add
        
        print(f"Current vocab size: {current_vocab_size}")
        print(f"Adding {tokens_to_add} unused tokens to reach {new_vocab_size}")
    
    # Create unused tokens
    unused_tokens = [f"[unused{i}]" for i in range(1, tokens_to_add + 1)]
    
    # Add tokens to tokenizer
    num_added = tokenizer.add_special_tokens(unused_tokens)
    
    print(f"Successfully added {num_added} unused tokens")
    print(f"New vocab size: {tokenizer.get_vocab_size()}")
    
    # Verify it's a multiple of target_multiple
    final_size = tokenizer.get_vocab_size()
    if final_size % target_multiple == 0:
        print(f"✓ Vocab size {final_size} is a multiple of {target_multiple}")
    else:
        print(f"⚠ Warning: Vocab size {final_size} is NOT a multiple of {target_multiple}")
    
    return tokenizer



def add_bert_template_processing(tokenizer: Tokenizer) -> Tokenizer:
    """
    Add BERT-style template processing to tokenizer.
    
    This adds [CLS] at the beginning and [SEP] at the end of sequences,
    and handles sentence pairs with [SEP] in between.
    
    Args:
        tokenizer: Input tokenizer
        
    Returns:
        Tokenizer with BERT-style template processing
    """
    # Get token IDs for special tokens
    cls_token = "[CLS]"
    sep_token = "[SEP]"
    
    vocab = tokenizer.get_vocab()
    
    if cls_token not in vocab:
        raise ValueError(f"{cls_token} token not found in vocabulary")
    if sep_token not in vocab:
        raise ValueError(f"{sep_token} token not found in vocabulary")
    
    cls_token_id = vocab[cls_token]
    sep_token_id = vocab[sep_token]
    
    print(f"Adding BERT template processing:")
    print(f"  [CLS] token ID: {cls_token_id}")
    print(f"  [SEP] token ID: {sep_token_id}")
    
    # Create BERT-style template processor
    # Single sequence: [CLS] $A [SEP]
    # Pair of sequences: [CLS] $A [SEP] $B [SEP]
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{cls_token}:0 $A:0 {sep_token}:0",
        pair=f"{cls_token}:0 $A:0 {sep_token}:0 $B:1 {sep_token}:1",
        special_tokens=[
            (cls_token, cls_token_id),
            (sep_token, sep_token_id),
        ],
    )
    
    print("BERT template processing added successfully")
    
    return tokenizer


def format_tokenizer(
    input_dir: Path,
    output_dir: Path,
    target_multiple: int = 64,
    num_unused: int = 0,
    add_template: bool = True
) -> Path:
    """
    Format tokenizer to BERT style.
    
    Args:
        input_dir: Directory containing input tokenizer
        output_dir: Directory to save formatted tokenizer
        target_multiple: Target multiple for vocab size
        num_unused: Number of unused tokens to add (0 = auto-calculate)
        add_template: Whether to add BERT template processing
        
    Returns:
        Path to saved tokenizer directory
    """
    print("=" * 70)
    print("Formatting Tokenizer to BERT Style")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target vocab multiple: {target_multiple}")
    print(f"Add template processing: {add_template}")
    print()
    
    # Load tokenizer
    tokenizer_json = input_dir / "tokenizer.json"
    if not tokenizer_json.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_json}")
    
    print(f"Loading tokenizer from {tokenizer_json}")
    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    
    original_vocab_size = tokenizer.get_vocab_size()
    print(f"Original vocab size: {original_vocab_size}")
    print()
    
    # Add unused tokens
    tokenizer = add_unused_tokens(tokenizer, target_multiple, num_unused)
    print()
    
    # Add BERT template processing
    if add_template:
        tokenizer = add_bert_template_processing(tokenizer)
        print()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save tokenizer.json
    output_tokenizer_json = output_dir / "tokenizer.json"
    tokenizer.save(str(output_tokenizer_json))
    print(f"Saved tokenizer.json to {output_tokenizer_json}")
    
    # Load as PreTrainedTokenizerFast and save with all config files
    print("\nCreating PreTrainedTokenizerFast...")
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(output_tokenizer_json),
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        clean_up_tokenization_spaces=True,
    )
    
    # Save the full tokenizer with all config files
    hf_tokenizer.save_pretrained(str(output_dir))
    print(f"Saved PreTrainedTokenizerFast to {output_dir}")
    
    # Update tokenizer_config.json with additional settings
    config_path = output_dir / "tokenizer_config.json"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Add/update important settings
        config["clean_up_tokenization_spaces"] = True
        config["model_max_length"] = 1000000000000000019884624838656  # BERT default
        config["tokenizer_class"] = "PreTrainedTokenizerFast"
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"Updated tokenizer_config.json")
    
    print()
    print("=" * 70)
    print("Formatting Complete!")
    print("=" * 70)
    print(f"Original vocab size: {original_vocab_size}")
    print(f"New vocab size: {tokenizer.get_vocab_size()}")
    print(f"Output directory: {output_dir.absolute()}")
    print()
    
    return output_dir


def push_to_hub(tokenizer_dir: Path, repo_id: str, commit_message: Optional[str] = None):
    """
    Push tokenizer to HuggingFace Hub.
    
    Args:
        tokenizer_dir: Directory containing tokenizer files
        repo_id: Repository ID (e.g., "username/tokenizer-name")
        commit_message: Optional commit message
    """
    print("=" * 70)
    print("Pushing to HuggingFace Hub")
    print("=" * 70)
    print(f"Repository: {repo_id}")
    print(f"Tokenizer directory: {tokenizer_dir}")
    print()
    
    from transformers import PreTrainedTokenizerFast
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))
    
    if commit_message is None:
        commit_message = "Upload Vietnamese tokenizer"
    
    print(f"Uploading tokenizer...")
    tokenizer.push_to_hub(
        repo_id=repo_id,
        commit_message=commit_message,
        use_auth_token=True
    )
    
    print()
    print("=" * 70)
    print("Upload Complete!")
    print("=" * 70)
    print(f"Tokenizer available at: https://huggingface.co/{repo_id}")
    print()


def verify_tokenizer(tokenizer_dir: Path):
    """
    Verify the formatted tokenizer.
    
    Args:
        tokenizer_dir: Directory containing tokenizer files
    """
    print("=" * 70)
    print("Verifying Formatted Tokenizer")
    print("=" * 70)
    
    from transformers import PreTrainedTokenizerFast
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))
    
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Model max length: {tokenizer.model_max_length}")
    print()
    
    print("Special tokens:")
    print(f"  UNK: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
    print(f"  CLS: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")
    print(f"  SEP: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})")
    print(f"  PAD: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"  MASK: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")
    print()
    
    test_text = "Tp. Hồ Chí Minh là thành phố lớn nhất Việt Nam."
    
    print("Testing tokenization:")
    print(f"Input: {test_text}")
    print()
    
    encoded = tokenizer(test_text)
    print("Single sequence encoding:")
    print(f"  Input IDs: {encoded['input_ids'][:20]}...")
    print(f"  Tokens: {tokenizer.convert_ids_to_tokens(encoded['input_ids'][:20])}...")
    print(f"  Length: {len(encoded['input_ids'])}")
    print()
    
    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'])
    if tokens[0] == "[CLS]" and tokens[-1] == "[SEP]":
        print("✓ BERT template processing working correctly")
    else:
        print("✗ BERT template processing not working as expected")
        print(f"  First token: {tokens[0]} (expected [CLS])")
        print(f"  Last token: {tokens[-1]} (expected [SEP])")
    print()
    
    text_a = "Câu hỏi: Thủ đô của Việt Nam là gì?"
    text_b = "Trả lời: Hà Nội"
    
    encoded_pair = tokenizer(text_a, text_b)
    tokens_pair = tokenizer.convert_ids_to_tokens(encoded_pair['input_ids'])
    
    print("Pair sequence encoding:")
    print(f"  Text A: {text_a}")
    print(f"  Text B: {text_b}")
    print(f"  Tokens: {tokens_pair}")
    print()
    
    sep_count = tokens_pair.count("[SEP]")
    if tokens_pair[0] == "[CLS]" and sep_count == 2:
        print("✓ Pair template processing working correctly")
    else:
        print("✗ Pair template processing not working as expected")
    print()
    
    print("=" * 70)
    print("Verification Complete!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Format tokenizer to BERT style"
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Input tokenizer directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for formatted tokenizer'
    )
    parser.add_argument(
        '--target_multiple',
        type=int,
        default=64,
        help='Target multiple for vocab size (default: 64)'
    )
    parser.add_argument(
        '--num_unused',
        type=int,
        default=0,
        help='Number of unused tokens to add (0 = auto-calculate to reach multiple)'
    )
    parser.add_argument(
        '--no_template',
        action='store_true',
        help='Skip adding BERT template processing'
    )
    parser.add_argument(
        '--push_to_hub',
        action='store_true',
        help='Push tokenizer to HuggingFace Hub'
    )
    parser.add_argument(
        '--repo_id',
        type=str,
        default=None,
        help='HuggingFace repository ID (required with --push_to_hub)'
    )
    parser.add_argument(
        '--commit_message',
        type=str,
        default=None,
        help='Commit message for HuggingFace Hub upload'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify the formatted tokenizer after creation'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        parser.error(f"Input directory not found: {input_dir}")
    
    # Format tokenizer
    tokenizer_dir = format_tokenizer(
        input_dir=input_dir,
        output_dir=output_dir,
        target_multiple=args.target_multiple,
        num_unused=args.num_unused,
        add_template=not args.no_template
    )
    
    # Verify if requested
    if args.verify:
        verify_tokenizer(tokenizer_dir)
    
    # Push to hub if requested
    if args.push_to_hub:
        if not args.repo_id:
            parser.error("--repo_id is required with --push_to_hub")
        
        push_to_hub(
            tokenizer_dir=tokenizer_dir,
            repo_id=args.repo_id,
            commit_message=args.commit_message
        )
    
    return 0


if __name__ == "__main__":
    exit(main())
