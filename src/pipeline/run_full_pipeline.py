#!/usr/bin/env python3
"""
Full pipeline orchestration for MDS data processing.

This script orchestrates the complete pipeline:
1. Download raw MDS data from HuggingFace
2. Tokenize using HUIT-BERT tokenizer
3. Upload tokenized data to HuggingFace
4. Chunk into 1K sequences and upload
5. Chunk into 8K sequences and upload

Usage:
    # Run full pipeline
    python src/pipeline/run_full_pipeline.py \
        --input_repo QuangDuy/FineWiki-mds \
        --tokenizer_path /teamspace/studios/this_studio/data-processing/tokenizer/HUIT-BERT \
        --output_prefix QuangDuy/FineWiki-mds

    # Run specific stages
    python src/pipeline/run_full_pipeline.py \
        --input_repo QuangDuy/FineWiki-mds \
        --tokenizer_path /path/to/tokenizer \
        --output_prefix QuangDuy/FineWiki-mds \
        --stages tokenize chunk_1k
"""

import argparse
import sys
import subprocess
from pathlib import Path
from typing import List, Optional
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.mds_helpers import download_from_hf


def run_command(cmd: List[str], stage_name: str) -> bool:
    """
    Run a command and return success status.
    
    Args:
        cmd: Command and arguments as list
        stage_name: Name of the stage for logging
        
    Returns:
        True if successful, False otherwise
    """
    print()
    print("="*60)
    print(f"Stage: {stage_name}")
    print("="*60)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        
        print()
        print(f"{stage_name} completed successfully in {elapsed:.1f}s")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print()
        print(f"{stage_name} failed after {elapsed:.1f}s: {e}")
        return False


def run_pipeline(
    input_repo: str,
    tokenizer_path: str,
    output_prefix: str,
    stages: List[str],
    hf_token: Optional[str] = None,
    private: bool = False,
    batch_size_tokenize: int = 5000,
    batch_size_chunk: int = 1000,
    temp_dir: str = "./temp_pipeline",
    resume: bool = False,
    compression: str = 'zstd'
):
    """
    Run the full data processing pipeline.
    
    Args:
        input_repo: HuggingFace input repository (e.g., 'QuangDuy/FineWiki-mds')
        tokenizer_path: Path to tokenizer
        output_prefix: Prefix for output repositories (e.g., 'QuangDuy/FineWiki-mds')
        stages: List of stages to run
        hf_token: HuggingFace API token
        private: Make output repositories private
        batch_size_tokenize: Batch size for tokenization
        batch_size_chunk: Batch size for chunking
        temp_dir: Temporary directory for processing
        resume: Resume from existing outputs
        compression: Compression type
    """
    tokenized_repo = f"{output_prefix}-tokenized"
    chunk_1k_repo = f"{output_prefix}-tokenized-1024"
    chunk_8k_repo = f"{output_prefix}-tokenized-8192"
    
    print("="*60)
    print("MDS Data Processing Pipeline")
    print("="*60)
    print(f"Input repository: {input_repo}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Output tokenized: {tokenized_repo}")
    print(f"Output 1K chunks: {chunk_1k_repo}")
    print(f"Output 8K chunks: {chunk_8k_repo}")
    print(f"Stages to run: {', '.join(stages)}")
    print()
    
    results = {}
    
    if 'tokenize' in stages:
        cmd = [
            'python', 'src/tokenization/tokenize_mds_subsets.py',
            '--hf_repo', input_repo,
            '--output_repo', tokenized_repo,
            '--tokenizer_path', tokenizer_path,
            '--batch_size', str(batch_size_tokenize),
            '--compression', compression,
            '--temp_dir', f"{temp_dir}/tokenization"
        ]
        
        if hf_token:
            cmd.extend(['--hf_token', hf_token])
        if private:
            cmd.append('--private')
        if resume:
            cmd.append('--resume')
        
        success = run_command(cmd, "Tokenization")
        results['tokenize'] = success
        
        if not success:
            print("\nPipeline stopped due to tokenization failure")
            return results
    
    if 'chunk_1k' in stages:
        cmd = [
            'python', 'src/sampling/chunk_tokenized_mds.py',
            '--hf_repo', tokenized_repo,
            '--output_repo', chunk_1k_repo,
            '--tokenizer_path', tokenizer_path,
            '--chunk_size', '1024',
            '--min_chunk_size', '512',
            '--always_skip_size', '128',
            '--batch_size', str(batch_size_chunk),
            '--compression', compression,
            '--temp_dir', f"{temp_dir}/chunk_1k"
        ]
        
        if hf_token:
            cmd.extend(['--hf_token', hf_token])
        if private:
            cmd.append('--private')
        if resume:
            cmd.append('--resume')
        
        success = run_command(cmd, "Chunking (1K)")
        results['chunk_1k'] = success
        
        if not success:
            print("\nPipeline stopped due to 1K chunking failure")
            return results
    
    if 'chunk_8k' in stages:
        cmd = [
            'python', 'src/sampling/chunk_tokenized_mds.py',
            '--hf_repo', tokenized_repo,
            '--output_repo', chunk_8k_repo,
            '--tokenizer_path', tokenizer_path,
            '--chunk_size', '8192',
            '--min_chunk_size', '512',
            '--always_skip_size', '32',
            '--batch_size', str(batch_size_chunk),
            '--compression', compression,
            '--temp_dir', f"{temp_dir}/chunk_8k"
        ]
        
        if hf_token:
            cmd.extend(['--hf_token', hf_token])
        if private:
            cmd.append('--private')
        if resume:
            cmd.append('--resume')
        
        success = run_command(cmd, "Chunking (8K)")
        results['chunk_8k'] = success
        
        if not success:
            print("\nPipeline stopped due to 8K chunking failure")
            return results
    
    print()
    print("="*60)
    print("Pipeline Complete!")
    print("="*60)
    print("Stage results:")
    for stage, success in results.items():
        status = "Success" if success else "Failed"
        print(f"  {status} {stage}")
    
    print()
    print("Output repositories:")
    if 'tokenize' in results:
        print(f"  Tokenized: https://huggingface.co/datasets/{tokenized_repo}")
    if 'chunk_1k' in results:
        print(f"  1K chunks: https://huggingface.co/datasets/{chunk_1k_repo}")
    if 'chunk_8k' in results:
        print(f"  8K chunks: https://huggingface.co/datasets/{chunk_8k_repo}")
    print()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run the full MDS data processing pipeline"
    )
    parser.add_argument(
        '--input_repo',
        type=str,
        required=True,
        help='HuggingFace input repository (e.g., QuangDuy/FineWiki-mds)'
    )
    parser.add_argument(
        '--tokenizer_path',
        type=str,
        required=True,
        help='Path to tokenizer (e.g., /path/to/HUIT-BERT)'
    )
    parser.add_argument(
        '--output_prefix',
        type=str,
        required=True,
        help='Prefix for output repositories (e.g., QuangDuy/FineWiki-mds)'
    )
    parser.add_argument(
        '--stages',
        type=str,
        nargs='+',
        default=['tokenize', 'chunk_1k', 'chunk_8k'],
        choices=['tokenize', 'chunk_1k', 'chunk_8k'],
        help='Stages to run (default: all)'
    )
    
    parser.add_argument(
        '--batch_size_tokenize',
        type=int,
        default=5000,
        help='Batch size for tokenization (default: 5000)'
    )
    parser.add_argument(
        '--batch_size_chunk',
        type=int,
        default=1000,
        help='Batch size for chunking (default: 1000)'
    )
    parser.add_argument(
        '--compression',
        type=str,
        default='zstd',
        choices=['zstd', 'gz', 'none'],
        help='Compression type (default: zstd)'
    )
    parser.add_argument(
        '--temp_dir',
        type=str,
        default='./temp_pipeline',
        help='Temporary directory for processing (default: ./temp_pipeline)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing outputs'
    )
    parser.add_argument(
        '--hf_token',
        type=str,
        default=None,
        help='HuggingFace API token'
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Make output repositories private'
    )
    
    args = parser.parse_args()
    
    results = run_pipeline(
        input_repo=args.input_repo,
        tokenizer_path=args.tokenizer_path,
        output_prefix=args.output_prefix,
        stages=args.stages,
        hf_token=args.hf_token,
        private=args.private,
        batch_size_tokenize=args.batch_size_tokenize,
        batch_size_chunk=args.batch_size_chunk,
        temp_dir=args.temp_dir,
        resume=args.resume,
        compression=args.compression
    )
    
    if all(results.values()):
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())

