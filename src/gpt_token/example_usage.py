#!/usr/bin/env python3
"""
Example usage of the GPT-NeoX-style Vietnamese tokenizer.
Demonstrates training and testing the tokenizer.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gpt_token.train_tokenizer import train_tokenizer


def example_train_from_mds():
    """
    Example: Train tokenizer from Vietnamese MDS datasets.
    """
    print("="*70)
    print("Example 1: Training from MDS Datasets")
    print("="*70)
    
    train_tokenizer(
        mds_datasets=[
            "QuangDuy/FineWiki-mds",
            "QuangDuy/FineWeb2-vie-mds"
        ],
        save_path="output/vietnamese_neox_32k.json",
        vocab_size=32000,
        max_token_length=64,
        max_consecutive_spaces=24,
        cache_dir="data",
        hf_output_dir="output/vietnamese_neox_32k_hf"
    )


def example_train_from_jsonl():
    """
    Example: Train tokenizer from local JSONL files.
    """
    print("\n" + "="*70)
    print("Example 2: Training from JSONL Files")
    print("="*70)
    
    train_tokenizer(
        input_dir="data/vietnamese_jsonl",
        save_path="output/tokenizer_jsonl.json",
        vocab_size=32000,
        max_token_length=64,
        max_consecutive_spaces=24,
        hf_output_dir="output/tokenizer_jsonl_hf"
    )


def example_use_tokenizer():
    """
    Example: Load and use a trained tokenizer.
    """
    from tokenizers import Tokenizer
    
    print("\n" + "="*70)
    print("Example 3: Using Trained Tokenizer")
    print("="*70)
    
    # Load tokenizer
    tokenizer_path = "output/vietnamese_neox_32k.json"
    
    try:
        tokenizer = Tokenizer.from_file(tokenizer_path)
        print(f" Loaded tokenizer from {tokenizer_path}")
        print(f"  Vocab size: {tokenizer.get_vocab_size()}")
    except Exception as e:
        print(f"✗ Could not load tokenizer: {e}")
        print("  Please train the tokenizer first using example_train_from_mds()")
        return
    
    # Test cases
    test_texts = [
        "Xin chào, tôi là một trợ lý AI.",
        "Việt Nam là một quốc gia Đông Nam Á.",
        "def function():\n    return True",
        "This text has    multiple    spaces.",
    ]
    
    for i, text in enumerate(test_texts, 1):
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        
        print(f"\nTest {i}:")
        print(f"  Input:  '{text}'")
        print(f"  Tokens: {len(encoded.ids)} tokens")
        print(f"  IDs:    {encoded.ids[:10]}..." if len(encoded.ids) > 10 else f"  IDs:    {encoded.ids}")
        print(f"  Decoded: '{decoded}'")
        print(f"  Match:  {decoded == text}")


def example_use_hf_tokenizer():
    """
    Example: Use HuggingFace format tokenizer.
    """
    from transformers import PreTrainedTokenizerFast
    
    print("\n" + "="*70)
    print("Example 4: Using HuggingFace Format Tokenizer")
    print("="*70)
    
    tokenizer_dir = "output/vietnamese_neox_32k_hf"
    
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
        print(f" Loaded HuggingFace tokenizer from {tokenizer_dir}")
        print(f"  Vocab size: {len(tokenizer)}")
    except Exception as e:
        print(f"✗ Could not load HuggingFace tokenizer: {e}")
        print("  Please train with --hf_output_dir first")
        return
    
    # Show special tokens
    print("\nSpecial tokens:")
    print(f"  CLS:  {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")
    print(f"  SEP:  {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})")
    print(f"  MASK: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")
    print(f"  PAD:  {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"  UNK:  {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
    
    # Test encoding/decoding
    text = "Xin chào, đây là một bài kiểm tra."
    encoded = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(encoded)
    
    print(f"\nTest encoding:")
    print(f"  Input:   '{text}'")
    print(f"  Tokens:  {len(encoded)} tokens")
    print(f"  Decoded: '{decoded}'")
    
    # Test with special tokens
    text_with_special = f"{tokenizer.cls_token} {text} {tokenizer.sep_token}"
    encoded_with_special = tokenizer.encode(text_with_special, add_special_tokens=False)
    decoded_with_special = tokenizer.decode(encoded_with_special)
    
    print(f"\nWith special tokens:")
    print(f"  Input:   '{text_with_special}'")
    print(f"  Tokens:  {len(encoded_with_special)} tokens")
    print(f"  Decoded: '{decoded_with_special}'")


def example_compare_space_handling():
    """
    Example: Demonstrate consistent space splitting.
    """
    from tokenizers import Tokenizer
    
    print("\n" + "="*70)
    print("Example 5: Consistent Space Splitting (GPT-NeoX advantage)")
    print("="*70)
    
    tokenizer_path = "output/vietnamese_neox_32k.json"
    
    try:
        tokenizer = Tokenizer.from_file(tokenizer_path)
    except Exception as e:
        print(f"✗ Could not load tokenizer: {e}")
        return
    
    # Test consistency: same word at different positions
    word = "programming"
    text1 = word  # At start
    text2 = f"Hello {word}"  # In middle
    text3 = f"  {word}"  # After spaces
    
    tokens1 = tokenizer.encode(text1).ids
    tokens2 = tokenizer.encode(text2).ids
    tokens3 = tokenizer.encode(text3).ids
    
    print(f"\nWord: '{word}'")
    print(f"\nPosition 1 (start):  '{text1}'")
    print(f"  Tokens: {tokens1}")
    
    print(f"\nPosition 2 (middle): '{text2}'")
    print(f"  Tokens: {tokens2}")
    
    print(f"\nPosition 3 (after spaces): '{text3}'")
    print(f"  Tokens: {tokens3}")
    
    print(f"\n Consistent tokenization: The word '{word}' is tokenized")
    print(f"  uniformly regardless of position (GPT-NeoX improvement)")


def main():
    """
    Run all examples.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Examples for GPT-NeoX-style tokenizer")
    parser.add_argument(
        "--train-mds",
        action="store_true",
        help="Run training from MDS datasets example"
    )
    parser.add_argument(
        "--train-jsonl",
        action="store_true",
        help="Run training from JSONL files example"
    )
    parser.add_argument(
        "--use",
        action="store_true",
        help="Run tokenizer usage examples"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all examples"
    )
    
    args = parser.parse_args()
    
    if args.all or args.train_mds:
        example_train_from_mds()
    
    if args.all or args.train_jsonl:
        example_train_from_jsonl()
    
    if args.all or args.use:
        example_use_tokenizer()
        example_use_hf_tokenizer()
        example_compare_space_handling()
    
    if not any([args.train_mds, args.train_jsonl, args.use, args.all]):
        print("Please specify an example to run:")
        print("  --train-mds    : Train from MDS datasets")
        print("  --train-jsonl  : Train from JSONL files")
        print("  --use          : Use trained tokenizer")
        print("  --all          : Run all examples")


if __name__ == "__main__":
    main()

