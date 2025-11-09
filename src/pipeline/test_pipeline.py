#!/usr/bin/env python3
"""
Test script for the MDS data processing pipeline.

This script validates the pipeline components without requiring large downloads.
It creates a small synthetic MDS dataset, runs it through tokenization and chunking,
and verifies the outputs.

Usage:
    python src/pipeline/test_pipeline.py
"""

import json
import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.mds_helpers import (
    save_mds_subset,
    verify_mds_structure,
    discover_subset_folders,
    count_total_samples
)


def create_test_mds_dataset(output_path: Path, num_subsets: int = 2, samples_per_subset: int = 100):
    """
    Create a small synthetic MDS dataset for testing.
    
    Args:
        output_path: Path to create the test dataset
        num_subsets: Number of subset folders to create
        samples_per_subset: Number of samples per subset
    """
    print(f"Creating test MDS dataset at {output_path}")
    print(f"  Subsets: {num_subsets}")
    print(f"  Samples per subset: {samples_per_subset}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    sample_texts = [
        "Đây là một ví dụ văn bản tiếng Việt để kiểm tra tokenizer.",
        "Trí tuệ nhân tạo đang phát triển rất nhanh trong những năm gần đây.",
        "Việt Nam là một quốc gia có nền văn hóa lâu đời và phong phú.",
        "Học máy và deep learning là những lĩnh vực quan trọng của AI.",
        "Ngôn ngữ tự nhiên là một trong những thách thức lớn nhất của AI.",
    ]
    
    for subset_idx in range(num_subsets):
        subset_name = f"{subset_idx:03d}_{subset_idx*10:05d}"
        subset_path = output_path / subset_name
        
        def sample_generator():
            for i in range(samples_per_subset):
                text = " ".join(sample_texts * (i % 5 + 1))
                yield {
                    'id': f'{subset_name}-{i:05d}',
                    'text': text
                }
        
        columns = {'text': 'str', 'id': 'str'}
        save_mds_subset(
            output_path=subset_path,
            columns=columns,
            samples_generator=sample_generator(),
            compression='zstd'
        )
        
        print(f"Created subset {subset_name}")
    
    print(f"Test dataset created successfully")
    return output_path


def test_tokenization(input_path: Path, tokenizer_path: str, output_path: Path):
    """
    Test the tokenization script.
    
    Args:
        input_path: Path to test MDS dataset
        tokenizer_path: Path to tokenizer
        output_path: Path for tokenized output
    """
    print("\n" + "="*60)
    print("Testing Tokenization")
    print("="*60)
    
    from src.tokenization.tokenize_mds_subsets import tokenize_dataset
    
    try:
        stats = tokenize_dataset(
            input_path=input_path,
            output_path=output_path,
            tokenizer_path=tokenizer_path,
            batch_size=50,
            compression='zstd',
            resume=False
        )
        
        print(f"Tokenization successful")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Total tokens: {stats['total_tokens']}")
        
        results = verify_mds_structure(output_path)
        if results['valid']:
            print(f"Output structure is valid")
            print(f"  Subsets: {results['total_subsets']}")
            print(f"  Shards: {results['total_shards']}")
        else:
            print(f"Output structure validation failed:")
            for issue in results['issues']:
                print(f"  - {issue}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Tokenization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chunking(input_path: Path, tokenizer_path: str, output_path: Path, chunk_size: int):
    """
    Test the chunking script.
    
    Args:
        input_path: Path to tokenized MDS dataset
        tokenizer_path: Path to tokenizer
        output_path: Path for chunked output
        chunk_size: Chunk size to use
    """
    print("\n" + "="*60)
    print(f"Testing Chunking (chunk_size={chunk_size})")
    print("="*60)
    
    from src.sampling.chunk_tokenized_mds import chunk_dataset
    
    try:
        if chunk_size >= 8192:
            min_chunk_size = 512
            always_skip_size = 32
        else:
            min_chunk_size = 512
            always_skip_size = 128
        
        stats = chunk_dataset(
            input_path=input_path,
            output_path=output_path,
            tokenizer_path=tokenizer_path,
            chunk_size=chunk_size,
            min_chunk_size=min_chunk_size,
            always_skip_size=always_skip_size,
            backfill=True,
            backfill_no_duplicates=True,
            add_eos_token=False,
            batch_size=50,
            compression='zstd',
            resume=False,
            seed=42
        )
        
        print(f"Chunking successful")
        print(f"  Input samples: {stats['total_input_samples']}")
        print(f"  Output chunks: {stats['total_output_chunks']}")
        print(f"  Tokens written: {stats['total_tokens_written']}")
        print(f"  Tokens skipped: {stats['total_tokens_skipped']}")
        print(f"  Tokens duplicated: {stats['total_duplicated_tokens']}")
        
        results = verify_mds_structure(output_path)
        if results['valid']:
            print(f"Output structure is valid")
            print(f"  Subsets: {results['total_subsets']}")
            print(f"  Shards: {results['total_shards']}")
        else:
            print(f"Output structure validation failed:")
            for issue in results['issues']:
                print(f"  - {issue}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Chunking failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("MDS Pipeline Test Suite")
    print("="*60)
    
    tokenizer_path = "/teamspace/studios/this_studio/data-processing/tokenizer/HUIT-BERT"
    if not Path(tokenizer_path).exists():
        print(f"Tokenizer not found at {tokenizer_path}")
        print("  Please update the tokenizer_path in this test script")
        return 1
    
    temp_dir = Path(tempfile.mkdtemp(prefix="mds_pipeline_test_"))
    print(f"\nUsing temporary directory: {temp_dir}")
    
    try:
        test_raw_path = temp_dir / "test_raw"
        create_test_mds_dataset(test_raw_path, num_subsets=2, samples_per_subset=50)
        
        test_tokenized_path = temp_dir / "test_tokenized"
        tokenize_success = test_tokenization(test_raw_path, tokenizer_path, test_tokenized_path)
        
        if not tokenize_success:
            print("\nPipeline test failed at tokenization stage")
            return 1
        
        test_chunked_1k_path = temp_dir / "test_chunked_1k"
        chunk_1k_success = test_chunking(test_tokenized_path, tokenizer_path, test_chunked_1k_path, 1024)
        
        if not chunk_1k_success:
            print("\nPipeline test failed at 1K chunking stage")
            return 1
        
        test_chunked_8k_path = temp_dir / "test_chunked_8k"
        chunk_8k_success = test_chunking(test_tokenized_path, tokenizer_path, test_chunked_8k_path, 8192)
        
        if not chunk_8k_success:
            print("\nPipeline test failed at 8K chunking stage")
            return 1
        
        print("\n" + "="*60)
        print("All Pipeline Tests Passed!")
        print("="*60)
        print(f"Test artifacts in: {temp_dir}")
        print("\nTo clean up test files:")
        print(f"  rm -rf {temp_dir}")
        print()
        
        return 0
        
    except Exception as e:
        print(f"\nTest suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        print("\nNote: Test files have been left for inspection.")
        print(f"To remove them: rm -rf {temp_dir}")


if __name__ == "__main__":
    exit(main())

