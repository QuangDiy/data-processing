#!/usr/bin/env python3
"""
Quick test script for tokenizer.

Usage:
    python src/bpe/test_tokenizer_quick.py --tokenizer_path ./output/vietnamese_bpe_bert_hf_tokenizer --text "Xin chào Việt Nam"
"""

import argparse
import sys
from pathlib import Path

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from transformers import PreTrainedTokenizerFast, AutoTokenizer


def test_tokenizer(tokenizer_path: str, text: str):
    """Test tokenizer with a single sentence."""
    path = Path(tokenizer_path)
    
    if path.exists() and path.is_dir():
        print(f"Loading from local: {path.resolve()}")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(path.resolve()))
    else:
        print(f"Loading from HuggingFace: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    encoded = tokenizer(text)
    
    decoded = tokenizer.decode(encoded['input_ids'])
    
    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'])
    
    print(dict(encoded))
    print(decoded)
    print(tokens)


def main():
    parser = argparse.ArgumentParser(description="Quick tokenizer test")
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to tokenizer')
    parser.add_argument('--text', type=str, required=True, help='Text to tokenize')
    
    args = parser.parse_args()
    test_tokenizer(args.tokenizer_path, args.text)


if __name__ == "__main__":
    main()
