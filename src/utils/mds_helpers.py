#!/usr/bin/env python3
"""
Helper utilities for working with MDS datasets.
Provides functions for loading, saving, and uploading MDS datasets.
"""

import json
import re
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from streaming import MDSWriter, StreamingDataset
import numpy as np
from tqdm import tqdm


def discover_subset_folders(dataset_path: Path) -> List[Path]:
    """
    Discover all subset folders matching the pattern XXX_XXXXX.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        List of paths to subset folders, sorted by name
    """
    pattern = re.compile(r'^\d{3}_\d{5}$')
    subset_folders = []
    
    if not dataset_path.exists():
        print(f"Warning: Dataset path {dataset_path} does not exist")
        return []
    
    for item in dataset_path.iterdir():
        if item.is_dir() and pattern.match(item.name):
            index_file = item / "index.json"
            if index_file.exists():
                subset_folders.append(item)
            else:
                print(f"Warning: Found subset folder {item} but no index.json, skipping")
    
    return sorted(subset_folders)


def load_index_file(index_path: Path) -> Dict:
    """Load and parse an MDS index.json file."""
    with open(index_path, 'r') as f:
        return json.load(f)


def save_index_file(index_path: Path, index_data: Dict):
    """Save an MDS index.json file."""
    with open(index_path, 'w') as f:
        json.dump(index_data, f, indent=2)


def load_mds_subset(subset_path: Path, batch_size: int = 1000):
    """
    Load samples from an MDS subset folder.
    
    Args:
        subset_path: Path to subset folder containing index.json and shards
        batch_size: Number of samples to yield at once
        
    Yields:
        Batches of samples as lists of dictionaries
    """
    if not subset_path.exists():
        raise FileNotFoundError(f"Subset path {subset_path} does not exist")
    
    index_file = subset_path / "index.json"
    if not index_file.exists():
        raise FileNotFoundError(f"No index.json found in {subset_path}")
    
    dataset = StreamingDataset(local=str(subset_path), split=None, shuffle=False, batch_size=batch_size)
    
    batch = []
    for sample in dataset:
        batch.append(sample)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    
    if batch:
        yield batch


def save_mds_subset(
    output_path: Path,
    columns: Dict[str, str],
    samples_generator,
    compression: str = 'zstd',
    hashes: List[str] = None,
    size_limit: int = 67108864  # 64MB default
) -> Tuple[int, int]:
    """
    Save samples to an MDS subset folder.
    
    Args:
        output_path: Path to output subset folder
        columns: Dictionary mapping column names to types (e.g., {'id': 'str', 'text': 'str'})
        samples_generator: Generator or iterable yielding sample dictionaries
        compression: Compression type ('zstd', 'gz', or None)
        hashes: List of hash algorithms to use
        size_limit: Maximum shard size in bytes
        
    Returns:
        Tuple of (total_samples, total_size_bytes)
    """
    # Clean existing directory if it's not empty (MDSWriter requires empty directories)
    if output_path.exists() and any(output_path.iterdir()):
        shutil.rmtree(output_path)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    if hashes is None:
        hashes = ['sha1', 'xxh64']
    
    total_samples = 0
    
    with MDSWriter(
        out=str(output_path),
        columns=columns,
        compression=compression,
        hashes=hashes,
        size_limit=size_limit
    ) as writer:
        for sample in samples_generator:
            writer.write(sample)
            total_samples += 1
    
    total_size = sum(f.stat().st_size for f in output_path.glob('*') if f.is_file())
    
    return total_samples, total_size


def verify_mds_structure(dataset_path: Path) -> Dict:
    """
    Verify MDS dataset structure and return statistics.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        Dictionary with verification results and statistics
    """
    results = {
        'valid': True,
        'total_subsets': 0,
        'total_shards': 0,
        'total_samples': 0,
        'issues': []
    }
    
    subset_folders = discover_subset_folders(dataset_path)
    results['total_subsets'] = len(subset_folders)
    
    if not subset_folders:
        results['valid'] = False
        results['issues'].append(f"No subset folders found in {dataset_path}")
        return results
    
    for subset_folder in subset_folders:
        index_file = subset_folder / "index.json"
        
        try:
            index_data = load_index_file(index_file)
            shards = index_data.get('shards', [])
            results['total_shards'] += len(shards)
            
            for shard in shards:
                results['total_samples'] += shard.get('samples', 0)
                
                raw_data = shard.get('raw_data')
                zip_data = shard.get('zip_data')
                
                if raw_data and raw_data.get('basename'):
                    shard_file = subset_folder / raw_data['basename']
                    if not shard_file.exists():
                        results['issues'].append(f"Missing shard file: {shard_file}")
                        results['valid'] = False
                
                if zip_data and zip_data.get('basename'):
                    shard_file = subset_folder / zip_data['basename']
                    if not shard_file.exists():
                        results['issues'].append(f"Missing shard file: {shard_file}")
                        results['valid'] = False
                        
        except Exception as e:
            results['valid'] = False
            results['issues'].append(f"Error processing {subset_folder}: {str(e)}")
    
    return results


def upload_to_hf(
    local_path: Path,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
    commit_message: Optional[str] = None
):
    """
    Upload dataset to HuggingFace Hub.
    
    Args:
        local_path: Path to local dataset directory
        repo_id: HuggingFace repository ID (e.g., 'QuangDuy/FineWiki-mds-tokenized')
        token: HuggingFace API token
        private: Whether to make the repository private
        commit_message: Optional commit message
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to upload to HuggingFace. "
            "Install it with: pip install huggingface_hub"
        )
    
    api = HfApi(token=token)
    
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            token=token,
            exist_ok=True
        )
        print(f"Repository {repo_id} is ready")
    except Exception as e:
        print(f"Note: {e}")
    
    if commit_message is None:
        commit_message = f"Upload dataset from {local_path.name}"
    
    print(f"Uploading {local_path} to {repo_id}...")
    
    api.upload_folder(
        folder_path=str(local_path),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
        token=token
    )
    
    print(f"Successfully uploaded to https://huggingface.co/datasets/{repo_id}")


def download_from_hf(
    repo_id: str,
    cache_dir: Optional[Path] = None,
    token: Optional[str] = None
) -> Path:
    """
    Download MDS dataset from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repository ID
        cache_dir: Optional cache directory
        token: Optional HuggingFace API token
        
    Returns:
        Path to downloaded dataset
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download from HuggingFace. "
            "Install it with: pip install huggingface_hub"
        )
    
    print(f"Downloading {repo_id} from HuggingFace...")
    
    local_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        cache_dir=str(cache_dir) if cache_dir else None,
        token=token,
        resume_download=True,
    )
    
    print(f"Downloaded to: {local_path}")
    return Path(local_path)


def get_tokenizer_special_tokens(tokenizer) -> Dict[str, int]:
    """
    Extract special token IDs from tokenizer.
    
    Args:
        tokenizer: HuggingFace tokenizer
        
    Returns:
        Dictionary with special token IDs
    """
    special_tokens = {}
    
    if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
        special_tokens['BOS'] = tokenizer.bos_token_id
    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
        special_tokens['EOS'] = tokenizer.eos_token_id
    if hasattr(tokenizer, 'cls_token_id') and tokenizer.cls_token_id is not None:
        special_tokens['CLS'] = tokenizer.cls_token_id
    if hasattr(tokenizer, 'sep_token_id') and tokenizer.sep_token_id is not None:
        special_tokens['SEP'] = tokenizer.sep_token_id
    if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
        special_tokens['PAD'] = tokenizer.pad_token_id
    
    return special_tokens


def count_total_samples(dataset_path: Path) -> int:
    """
    Count total samples across all subset folders.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        Total number of samples
    """
    total = 0
    subset_folders = discover_subset_folders(dataset_path)
    
    for subset_folder in subset_folders:
        index_file = subset_folder / "index.json"
        index_data = load_index_file(index_file)
        
        for shard in index_data.get('shards', []):
            total += shard.get('samples', 0)
    
    return total

