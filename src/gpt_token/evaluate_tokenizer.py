#!/usr/bin/env python3
"""
Evaluate Vietnamese BPE Tokenizer

This script evaluates tokenizers using metrics from the paper:
- Fertility: Average number of tokens per word (lower is better)
- Proportion of Continued Words (PCW): Percentage of words split into 2+ tokens (lower is better)

Usage:
    # Evaluate your trained tokenizer
    python src/bpe/evaluate_tokenizer.py --tokenizer_path ./output/vietnamese_bpe_hf_tokenizer

    # Compare with other tokenizers
    python src/bpe/evaluate_tokenizer.py --tokenizer_path ./output/vietnamese_bpe_hf_tokenizer --compare

    # Use custom sample count
    python src/bpe/evaluate_tokenizer.py --tokenizer_path ./output/vietnamese_bpe_hf_tokenizer --num_samples 5000
"""

import argparse
from pathlib import Path
import os
os.environ["PYTHONUTF8"] = "1"

import numpy as np
import pandas as pd
from tqdm import tqdm
import random

random.seed(42)
np.random.seed(42)


REFERENCE_TOKENIZERS = [
    ("PhoBERT", "vinai/phobert-base"),
    ("ViSoBERT", "uitnlp/visobert"),
    ("ViDeBERTa", "Fsoft-AIC/videberta-xsmall"),
    ("ViBERT", "FPTAI/vibert-base-cased"),
    ("XLM-RoBERTa", "xlm-roberta-base"),
    # ("VietMistral", "hiieu/Vistral-7B-Chat-function-calling")
]

LANGUAGES = [
    ("Vietnamese", "vie_Latn", "vi"),
    # ("English", "eng_Latn", "en"),
]


def load_evaluation_data(languages: list, num_samples: int = 10000) -> dict:
    """
    Load Wikipedia and FineWeb2 data for evaluation.
    
    Args:
        languages: List of (language_name, lang_code, short_code) tuples
        num_samples: Number of articles/documents to sample per language per dataset
        
    Returns:
        Dictionary with keys 'wikipedia' and 'fineweb2', each mapping lang_code to text
    """
    from datasets import load_dataset
    
    data = {
        'wikipedia': {},
        'fineweb2': {}
    }
    
    for lang_name, lang_code, short_lang_code in languages:
        print(f"Loading Wikipedia for {lang_name} ({short_lang_code})...")
        try:
            wiki_ds = load_dataset(
                "wikimedia/wikipedia", 
                f"20231101.{short_lang_code}", 
                streaming=True, 
                split="train"
            )
            wiki_ds = wiki_ds.shuffle(seed=42, buffer_size=10_000)
            
            ds_iter = iter(wiki_ds)
            wiki_texts = []
            for _ in tqdm(range(num_samples), desc=f"Sampling Wikipedia {lang_name}"):
                try:
                    wiki_texts.append(next(ds_iter)["text"])
                except StopIteration:
                    break
            
            data['wikipedia'][lang_code] = "\n".join(wiki_texts)
            print(f"  Loaded {len(wiki_texts)} Wikipedia articles, {sum(len(t) for t in wiki_texts):,} characters")
            
        except Exception as e:
            print(f"  Error loading Wikipedia {lang_name}: {e}")
        
        if lang_code == "vie_Latn":
            print(f"Loading FineWeb2 for {lang_name}...")
            try:
                fw = load_dataset(
                    "HuggingFaceFW/fineweb-2", 
                    name="vie_Latn", 
                    split="train", 
                    streaming=True
                )
                fw = fw.shuffle(seed=42, buffer_size=10_000)
                
                fw_iter = iter(fw)
                fw_texts = []
                for _ in tqdm(range(num_samples), desc=f"Sampling FineWeb2 {lang_name}"):
                    try:
                        fw_texts.append(next(fw_iter)["text"])
                    except StopIteration:
                        break
                
                data['fineweb2'][lang_code] = "\n".join(fw_texts)
                print(f"  Loaded {len(fw_texts)} FineWeb2 documents, {sum(len(t) for t in fw_texts):,} characters")
                
            except Exception as e:
                print(f"  Error loading FineWeb2 {lang_name}: {e}")
    
    return data


class SimpleSyllableTokenizer:
    """Simple whitespace-based syllable tokenizer for Vietnamese.
    
    Vietnamese words are typically written with spaces between syllables,
    so splitting on whitespace gives us syllables, not compound words.
    This is more appropriate for measuring tokenizer efficiency.
    """
    
    def word_tokenize(self, text: str) -> list:
        """Tokenize text into syllables/words by splitting on whitespace."""
        import re
        # Split on whitespace, filter out empty strings and pure punctuation
        words = re.split(r'\s+', text)
        # Keep only words that contain at least one alphanumeric character
        words = [w for w in words if w and re.search(r'\w', w)]
        return words


def compute_tokenizer_metrics(tokenizer, text: str) -> tuple:
    """
    Computes fertility and proportion of continued words.
    
    Uses simple whitespace tokenization to split text into syllables/words.
    For Vietnamese, this gives syllables (not compound words from segmenters).
    For English, this gives words.
    
    Args:
        tokenizer: HuggingFace tokenizer
        text: Text to evaluate
        
    Returns:
        tuple: (fertility, proportion_continued_words)
            - fertility: average tokens per syllable/word (lower is better)
            - proportion_continued_words: percentage of syllables/words split into 2+ tokens (lower is better)
    """
    syllable_tokenizer = SimpleSyllableTokenizer()
    words = syllable_tokenizer.word_tokenize(text)
    
    if len(words) == 0:
        return 0.0, 0.0
    
    tokens = tokenizer.batch_encode_plus(words, add_special_tokens=False)
    tokens_per_word = np.array(list(map(len, tokens["input_ids"])))
    
    fertility = np.mean(tokens_per_word).item()
    proportion_continued_words = (tokens_per_word >= 2).sum() / len(tokens_per_word)
    
    return fertility, proportion_continued_words


def evaluate_tokenizer(
    tokenizer_path: str,
    languages: list = None,
    num_samples: int = 10000,
    compare_with_others: bool = False,
    reference_tokenizers: list = None,
    compare_paths: list = None
) -> pd.DataFrame:
    """
    Evaluate a tokenizer on multiple languages and datasets.
    
    Args:
        tokenizer_path: Path to HuggingFace tokenizer directory or tokenizer ID
        languages: List of (language_name, lang_code, short_code) tuples
        num_samples: Number of Wikipedia articles to sample per language
        compare_with_others: Whether to also evaluate reference tokenizers
        reference_tokenizers: List of (name, HF_path) tuples for comparison
        
    Returns:
        DataFrame with evaluation results
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    
    if languages is None:
        languages = LANGUAGES
    
    if reference_tokenizers is None:
        reference_tokenizers = REFERENCE_TOKENIZERS
    
    print("\n" + "=" * 70)
    print("Loading Evaluation Data (Wikipedia + FineWeb2)")
    print("=" * 70)
    data = load_evaluation_data(languages, num_samples)
    
    tokenizers_to_evaluate = []
    
    print("\n" + "=" * 70)
    print("Loading Tokenizers")
    print("=" * 70)
    
    tokenizer_path = Path(tokenizer_path).resolve()
    if tokenizer_path.exists() and tokenizer_path.is_dir():
        print(f"Loading tokenizer from local path: {tokenizer_path}")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))
        tokenizer_name = tokenizer_path.name
    else:
        print(f"Loading tokenizer from HuggingFace: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
        tokenizer_name = str(tokenizer_path)
    
    tokenizers_to_evaluate.append((tokenizer_name, tokenizer))
    print(f"  {tokenizer_name}: vocab_size = {tokenizer.vocab_size:,}")
    
    if compare_with_others:
        print("\nLoading reference tokenizers for comparison...")
        for name, path in reference_tokenizers:
            try:
                ref_tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
                tokenizers_to_evaluate.append((name, ref_tokenizer))
                print(f"  {name}: vocab_size = {ref_tokenizer.vocab_size:,}")
            except Exception as e:
                print(f"  Failed to load {name}: {e}")
    
    if compare_paths:
        print("\nLoading additional tokenizers for comparison...")
        for path_str in compare_paths:
            try:
                compare_path = Path(path_str).resolve()
                if compare_path.exists() and compare_path.is_dir():
                    print(f"Loading tokenizer from local path: {compare_path}")
                    compare_tokenizer = PreTrainedTokenizerFast.from_pretrained(str(compare_path))
                    compare_name = compare_path.name
                else:
                    print(f"Loading tokenizer from HuggingFace: {path_str}")
                    compare_tokenizer = AutoTokenizer.from_pretrained(path_str, trust_remote_code=True)
                    compare_name = path_str
                
                tokenizers_to_evaluate.append((compare_name, compare_tokenizer))
                print(f"  {compare_name}: vocab_size = {compare_tokenizer.vocab_size:,}")
            except Exception as e:
                print(f"  Failed to load {path_str}: {e}")

    
    print("\n" + "=" * 70)
    print("Evaluating Tokenizers")
    print("=" * 70)
    
    results = []
    
    for tokenizer_name, tokenizer in tokenizers_to_evaluate:
        print(f"\nEvaluating {tokenizer_name}...")
        
        for lang_name, lang_code, short_lang_code in languages:
            # Evaluate on Wikipedia
            if lang_code in data['wikipedia']:
                print(f"  Computing metrics for {lang_name} (Wikipedia)...")
                
                try:
                    fertility, pcw = compute_tokenizer_metrics(
                        tokenizer, 
                        data['wikipedia'][lang_code]
                    )
                    
                    results.append({
                        "tokenizer": tokenizer_name,
                        "language": lang_name,
                        "dataset": "Wikipedia",
                        "fertility": fertility,
                        "pcw": pcw,
                        "vocab_size": tokenizer.vocab_size
                    })
                    
                    print(f"    Fertility: {fertility:.4f}")
                    print(f"    PCW: {pcw:.4f}")
                    
                except Exception as e:
                    print(f"    Error: {e}")
            
            if lang_code in data['fineweb2']:
                print(f"  Computing metrics for {lang_name} (FineWeb2)...")
                
                try:
                    fertility, pcw = compute_tokenizer_metrics(
                        tokenizer, 
                        data['fineweb2'][lang_code]
                    )
                    
                    results.append({
                        "tokenizer": tokenizer_name,
                        "language": lang_name,
                        "dataset": "FineWeb2",
                        "fertility": fertility,
                        "pcw": pcw,
                        "vocab_size": tokenizer.vocab_size
                    })
                    
                    print(f"    Fertility: {fertility:.4f}")
                    print(f"    PCW: {pcw:.4f}")
                    
                except Exception as e:
                    print(f"    Error: {e}")
    
    return pd.DataFrame(results)


def print_results(df: pd.DataFrame):
    """Print formatted evaluation results."""
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    
    print("\nDetailed Results:")
    print("-" * 70)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))
    
    # Summary by dataset
    for dataset_name in df['dataset'].unique():
        print("\n" + "-" * 70)
        print(f"Summary for {dataset_name} (averaged across languages):")
        print("-" * 70)
        
        dataset_df = df[df['dataset'] == dataset_name]
        summary = dataset_df.groupby('tokenizer').agg({
            'fertility': 'mean',
            'pcw': 'mean',
            'vocab_size': 'first'
        }).round(4)
        
        summary = summary.sort_values('fertility')
        print(summary.to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Vietnamese BPE tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Evaluate your trained tokenizer
  python src/bpe/evaluate_tokenizer.py --tokenizer_path ./output/vietnamese_bpe_hf_tokenizer

  # Compare with other tokenizers
  python src/bpe/evaluate_tokenizer.py --tokenizer_path ./output/vietnamese_bpe_hf_tokenizer --compare

  # Use fewer samples for faster evaluation
  python src/bpe/evaluate_tokenizer.py --tokenizer_path ./output/vietnamese_bpe_hf_tokenizer --num_samples 1000
        """
    )
    
    parser.add_argument(
        '--tokenizer_path',
        type=str,
        required=True,
        help='Path to HuggingFace tokenizer directory or HuggingFace model ID'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=10000,
        help='Number of Wikipedia articles to sample per language (default: 10000)'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare with reference tokenizers (Llama3, Gemma3, Qwen3, etc.)'
    )
    parser.add_argument(
        '--compare_paths',
        type=str,
        nargs='+',
        default=[],
        help='Additional tokenizer paths to compare with (local paths or HuggingFace IDs)'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default=None,
        help='Path to save results as CSV (optional)'
    )
    parser.add_argument(
        '--languages',
        type=str,
        nargs='+',
        default=['vi', 'en'],
        help='Languages to evaluate (default: vi en)'
    )
    
    args = parser.parse_args()
    
    lang_map = {
        'vi': ("Vietnamese", "vie_Latn", "vi"),
        # 'en': ("English", "eng_Latn", "en"),
    }
    
    languages = [lang_map.get(lang, lang_map['vi']) for lang in args.languages if lang in lang_map]
    
    if not languages:
        print("Error: No valid languages specified")
        return 1
    
    print("=" * 70)
    print("Vietnamese BPE Tokenizer Evaluation")
    print("=" * 70)
    print(f"Tokenizer: {args.tokenizer_path}")
    print(f"Languages: {[l[0] for l in languages]}")
    print(f"Samples per language: {args.num_samples:,}")
    print(f"Compare with others: {args.compare}")
    if args.compare_paths:
        print(f"Compare paths: {args.compare_paths}")
    
    try:
        results_df = evaluate_tokenizer(
            tokenizer_path=args.tokenizer_path,
            languages=languages,
            num_samples=args.num_samples,
            compare_with_others=args.compare,
            compare_paths=args.compare_paths
        )
        
        print_results(results_df)
        
        if args.output_csv:
            results_df.to_csv(args.output_csv, index=False)
            print(f"\nResults saved to: {args.output_csv}")
        
        return 0
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
