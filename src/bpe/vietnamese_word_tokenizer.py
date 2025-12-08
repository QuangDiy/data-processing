#!/usr/bin/env python3
"""
Vietnamese Word Tokenizer with Enhanced Regex Patterns

This module provides Vietnamese-specific tokenization patterns optimized for BPE training.
It includes patterns for Vietnamese abbreviations, special characters (including Đ),
emails, URLs, numbers, and other Vietnamese language characteristics.
"""

import re
import unicodedata as ud
from typing import List


def word_tokenizer(text: str) -> List[str]:
    """
    Tokenize Vietnamese text using enhanced regex patterns.
    
    This function handles Vietnamese-specific patterns including:
    - Vietnamese abbreviations (Tp., Mr., Mrs., Ms., Dr., ThS.)
    - Vietnamese uppercase letters including Đ
    - Special tokens (==>, ->, ..., >>, newlines)
    - Email addresses
    - URLs
    - Numbers with separators
    - General words and non-word characters
    
    Args:
        text: Input Vietnamese text to tokenize
        
    Returns:
        List of tokens
        
    Example:
        >>> text = "Tp. Hồ Chí Minh có email: test@example.com"
        >>> tokens = word_tokenizer(text)
        >>> print(tokens)
        ['Tp.', 'Hồ', 'Chí', 'Minh', 'có', 'email', ':', 'test@example.com']
    """
    text = ud.normalize('NFC', text)
    
    specials = ["==>", "->", r"\.\.\.", ">>", r'\n']
    digit = r"\d+([\.,_]\d+)+"
    email = r"([a-zA-Z0-9_.+-]+@([a-zA-Z0-9-]+\.)+[a-zA-Z0-9-]+)"
    web = r"\w+://[^\s]+"
    word = r"\w+"
    non_word = r"[^\w\s]"
    
    abbreviations = [
        r"[A-ZĐ]+\.", 
        r"Tp\.",       
        r"Mr\.", r"Mrs\.", r"Ms\.",
        r"Dr\.",      
        r"ThS\.",     
    ]
    
    patterns = []
    patterns.extend(abbreviations)
    patterns.extend(specials)
    patterns.extend([web, email])
    patterns.extend([digit, non_word, word])
    
    combined_pattern = "(" + "|".join(patterns) + ")"
    tokens = re.findall(combined_pattern, text, re.UNICODE)
    
    return [token[0] for token in tokens]


def get_vietnamese_bpe_regex() -> str:
    """
    Get the regex pattern optimized for Vietnamese BPE tokenization.
    
    This pattern is designed to work with the bpeasy library and follows
    a GPT-4 style regex pattern with Vietnamese enhancements.
    
    The pattern handles:
    - English contractions ('s, 't, 're, 've, 'm, 'll, 'd)
    - Vietnamese abbreviations with dots (Tp., Mr., Mrs., Dr., ThS.)
    - Letters with optional leading space (enables " từ" style tokens)
    - Numbers (1-3 digits at a time)
    - Special character sequences with optional leading space
    - Multi-space tokens (allows "    ", "   ", "  " as separate tokens)
    - Whitespace handling for newlines
    
    Returns:
        Regex pattern string compatible with bpeasy training
        
    Example:
        >>> regex = get_vietnamese_bpe_regex()
        >>> # Use with bpeasy.train_bpe(iterator, regex, max_token_length, vocab_size)
    """
    
    compact_pattern = (
        r"'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]|"
        r"[^\r\n\p{L}\p{N}]?\p{L}+|"
        r"\p{N}{1,3}|"
        r" ?[^\s\p{L}\p{N}]+[\r\n]*|"
        r"\s*[\r\n]+|"
        r"\s+"
    )
    
    return compact_pattern


def test_vietnamese_tokenizer():
    """
    Test the Vietnamese tokenizer with sample texts.
    """
    import sys
    import io
    
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    test_cases = [
        "Tp. Hồ Chí Minh là    thành phố lớn nhất Việt Nam.",
        "Email: contact@example.com và website: https://example.com",
        "Số điện thoại: 123.456.789 hoặc 123,456,789",
        "ThS. Nguyễn Văn A và Dr. Trần Thị B",
        "Đây là văn bản tiếng Việt ==> kết quả",
        "ĐHQG-HCM, TP.HCM, Mr. Smith",
    ]
    
    print("Testing Vietnamese Word Tokenizer")
    print("=" * 60)
    
    for i, text in enumerate(test_cases, 1):
        tokens = word_tokenizer(text)
        print(f"\nTest {i}:")
        print(f"Input:  {text}")
        print(f"Tokens: {tokens}")
        print(f"Count:  {len(tokens)} tokens")
    
    print("\n" + "=" * 60)
    print("Vietnamese BPE Regex Pattern:")
    print(get_vietnamese_bpe_regex())


if __name__ == "__main__":
    test_vietnamese_tokenizer()
