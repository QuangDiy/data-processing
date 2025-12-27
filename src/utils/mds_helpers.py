#!/usr/bin/env python3
"""
Helper utilities for working with MDS datasets.
Provides functions for loading, saving, and uploading MDS datasets.
"""

import json
import os
import re
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Iterator
from streaming import MDSWriter, StreamingDataset
from streaming.base.format import reader_from_json
from streaming.base.spanner import Spanner
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


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


def has_compressed_files(subset_path: Path) -> bool:
    """
    Check if subset contains compressed (.zstd) files.
    
    Args:
        subset_path: Path to subset folder
        
    Returns:
        True if compressed files exist, False otherwise
    """
    return any(subset_path.glob("*.mds.zstd"))


def load_mds_subset(
    subset_path: Path, 
    batch_size: int = 1000, 
    delete_zstd_after: bool = False,
    use_streaming: Optional[bool] = None
):
    """
    Load samples from an MDS subset folder.
    
    Automatically chooses optimal loading method:
    - NoStreamingMDSDataset: For uncompressed local files (faster, more efficient)
    - StreamingDataset: For compressed files or when explicitly requested
    
    Args:
        subset_path: Path to subset folder containing index.json and shards
        batch_size: Number of samples to yield at once
        delete_zstd_after: If True, delete .zstd files after loading completes.
                          Only deletes .zstd if corresponding .mds exists (safe).
        use_streaming: Force use of StreamingDataset (True) or NoStreamingMDSDataset (False).
                      If None, automatically detect based on file compression.
        
    Yields:
        Batches of samples as lists of dictionaries
    """
    if not subset_path.exists():
        raise FileNotFoundError(f"Subset path {subset_path} does not exist")
    
    index_file = subset_path / "index.json"
    if not index_file.exists():
        raise FileNotFoundError(f"No index.json found in {subset_path}")
    
    # Auto-detect: use streaming if compressed files exist
    if use_streaming is None:
        use_streaming = has_compressed_files(subset_path)
    
    if use_streaming:
        # Use StreamingDataset for compressed files or when explicitly requested
        print(f"Using StreamingDataset (compressed files detected)")
        dataset = StreamingDataset(
            local=str(subset_path), 
            split=None, 
            shuffle=False, 
            batch_size=batch_size
        )
    else:
        # Use NoStreamingMDSDataset for uncompressed local files (more efficient)
        print(f"Using NoStreamingMDSDataset (no compression, direct access)")
        dataset = NoStreamingMDSDataset(
            local=subset_path,
            split=None,
            shuffle=False
        )
    
    # Yield samples in batches
    batch = []
    for sample in dataset:
        batch.append(sample)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    
    if batch:
        yield batch
    
    # Clean up .zstd files after iteration completes (only if .mds exists)
    if delete_zstd_after:
        zstd_files = list(subset_path.glob("*.mds.zstd"))
        for zstd_file in zstd_files:
            # Safety check: only delete if uncompressed .mds file exists
            mds_file = zstd_file.with_suffix('')  # Remove .zstd -> get .mds path
            if not mds_file.exists():
                continue  # Skip - don't delete .zstd if .mds doesn't exist
            try:
                zstd_file.unlink()
            except Exception as e:
                print(f"Warning: Could not delete {zstd_file}: {e}")


class NoStreamingMDSDataset(Dataset):
    """
    Efficient dataset class for reading local MDS format data without streaming overhead.
    
    This is a slimmer, more efficient alternative to StreamingDataset when:
    - Data is stored locally (no remote streaming needed)
    - Files are uncompressed (.mds without .zstd)
    - You want direct file access without streaming infrastructure
    
    Based on the NoStreamingDataset pattern from MosaicML/OLMo.
    
    Args:
        local: Path to local dataset directory
        split: Optional split subdirectory name
        shuffle: Whether to shuffle samples (default: False)
        
    Example:
        >>> dataset = NoStreamingMDSDataset(
        ...     local='data/FineWiki-mds/datasets--QuangDuy--FineWiki-mds/snapshots/.../000_00000',
        ...     split=None
        ... )
        >>> sample = dataset[0]  # Get first sample
        >>> print(sample['text'][:100])
    """
    
    def __init__(
        self,
        local: Union[str, Path],
        split: Optional[str] = None,
        shuffle: bool = False,
    ) -> None:
        super().__init__()
        
        local = Path(local) if isinstance(local, str) else local
        
        if split is not None:
            split_path = local / split
        else:
            split_path = local
            
        if not split_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {split_path}")
            
        index_file_path = split_path / "index.json"
        if not index_file_path.exists():
            raise FileNotFoundError(f"No index.json found in {split_path}")
            
        with open(index_file_path, 'r') as f:
            obj = json.load(f)
        
        self.shards = []
        for info in obj["shards"]:
            if 'zip_data' not in info and info.get('compression'):
                info = dict(info)
                info['zip_data'] = None
            
            shard = reader_from_json(str(local), split, info)
            
            raw_filename = os.path.join(shard.dirname, shard.split or '', shard.raw_data.basename)
            if not os.path.isfile(raw_filename):
                raise FileNotFoundError(f"Raw shard file not found: {raw_filename}")
            
            shard.validate(True)
            self.shards.append(shard)
        
        samples_per_shard = np.array([shard.samples for shard in self.shards], dtype=np.int64)
        self.total_samples = int(samples_per_shard.sum())
        
        self.spanner = Spanner(samples_per_shard)
        
        self.shuffle = shuffle
        if shuffle:
            self.shuffle_indices = np.random.permutation(self.total_samples)
        else:
            self.shuffle_indices = None
    
    def __getitem__(self, index: int) -> Dict:
        """Get sample by index."""
        if self.shuffle_indices is not None:
            index = self.shuffle_indices[index]
        
        shard_id, shard_sample_id = self.spanner[index]
        
        shard = self.shards[shard_id]
        sample = shard[shard_sample_id]
        
        return sample
    
    def __len__(self) -> int:
        """Total number of samples in dataset."""
        return self.total_samples
    
    def __iter__(self) -> Iterator[Dict]:
        """Iterate over all samples in the dataset."""
        for i in range(len(self)):
            yield self[i]
    
    def __repr__(self) -> str:
        return (
            f"NoStreamingMDSDataset("
            f"shards={len(self.shards)}, "
            f"samples={self.total_samples}, "
            f"shuffle={self.shuffle})"
        )


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
    commit_message: Optional[str] = None,
    num_workers: int = 16,
    print_report_every: int = 30,
    ignore_patterns: Optional[List[str]] = None
):
    """
    Upload large dataset folder to HuggingFace Hub using parallel workers.
    
    Args:
        local_path: Path to local dataset directory
        repo_id: HuggingFace repository ID (e.g., 'QuangDuy/FineWiki-mds-tokenized')
        token: HuggingFace API token
        private: Whether to make the repository private
        commit_message: Optional commit message
        num_workers: Number of parallel upload workers (default: 16)
        print_report_every: Print progress report every N seconds (default: 30)
        ignore_patterns: Optional list of patterns to ignore (e.g., ['*.zstd'])
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
    

    print(f"Uploading large folder to HuggingFace Hub")

    print(f"Local path: {local_path}")
    print(f"Target repo: {repo_id}")
    print(f"Workers: {num_workers}")
    print(f"Report interval: {print_report_every}s")
    if ignore_patterns:
        print(f"Ignoring patterns: {ignore_patterns}")

    
    try:
        api.upload_large_folder(
            folder_path=str(local_path),
            repo_id=repo_id,
            repo_type="dataset",
            num_workers=num_workers,
            print_report=True,
            print_report_every=print_report_every,
            ignore_patterns=ignore_patterns
        )
        
        print(f"Successfully uploaded to https://huggingface.co/datasets/{repo_id}")
        
    except Exception as e:
        print(f"Upload failed: {e}")
        raise


def download_from_hf(
    repo_id: str,
    cache_dir: Optional[Path] = None,
    token: Optional[str] = None,
    subset_folders: Optional[List[str]] = None
) -> Path:
    """
    Download MDS dataset from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repository ID
        cache_dir: Optional cache directory
        token: Optional HuggingFace API token
        subset_folders: Optional list of subset folder names to download (e.g., ['003_00000', '004_00004'])
                        If None, downloads entire dataset
        
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
    
    if subset_folders:
        print(f"Downloading specific subsets from {repo_id}: {', '.join(subset_folders)}")
        allow_patterns = []
        for subset in subset_folders:
            allow_patterns.append(f"{subset}/**")
            allow_patterns.append(f"{subset}/*")
        allow_patterns.append("*.json")
        allow_patterns.append("*.md")
        allow_patterns.append("*.txt")
        
        local_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            cache_dir=str(cache_dir) if cache_dir else None,
            token=token,
            resume_download=True,
            allow_patterns=allow_patterns,
        )
    else:
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


def get_resume_state_path(output_path: Path) -> Path:
    """
    Get the path to the resume state file for a subset.
    
    Args:
        output_path: Path to output subset folder
        
    Returns:
        Path to .resume_state.json file
    """
    return output_path / ".resume_state.json"


def save_resume_state(output_path: Path, samples_processed: int, total_samples: int):
    """
    Save resume state for a subset being processed.
    
    Args:
        output_path: Path to output subset folder
        samples_processed: Number of samples processed so far
        total_samples: Total samples expected in this subset
    """
    output_path.mkdir(parents=True, exist_ok=True)
    state_file = get_resume_state_path(output_path)
    
    state = {
        'samples_processed': samples_processed,
        'total_samples': total_samples,
        'timestamp': str(Path.cwd())  # Just a placeholder for state validity
    }
    
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)


def load_resume_state(output_path: Path) -> Optional[Dict]:
    """
    Load resume state for a subset.
    
    Args:
        output_path: Path to output subset folder
        
    Returns:
        Resume state dictionary or None if no valid state exists
    """
    state_file = get_resume_state_path(output_path)
    
    if not state_file.exists():
        return None
    
    try:
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        # Validate state has required fields
        if 'samples_processed' in state and 'total_samples' in state:
            return state
        else:
            return None
            
    except Exception as e:
        print(f"Warning: Could not load resume state: {e}")
        return None


def clear_resume_state(output_path: Path):
    """
    Clear resume state after successful completion.
    
    Args:
        output_path: Path to output subset folder
    """
    state_file = get_resume_state_path(output_path)
    
    if state_file.exists():
        try:
            state_file.unlink()
        except Exception as e:
            print(f"Warning: Could not delete resume state file: {e}")


def check_resume_status(output_path: Path, input_total_samples: int) -> Tuple[bool, int, str]:
    """
    Check if a subset can be resumed and get status information.
    
    Args:
        output_path: Path to output subset folder
        input_total_samples: Total samples in input
        
    Returns:
        Tuple of (can_resume, samples_to_skip, status_message)
    """
    if output_path.exists():
        stats_files = [
            output_path / "tokenization_stats.json",
            output_path / "chunking_stats.json"
        ]
        
        for stats_file in stats_files:
            if stats_file.exists():
                return False, 0, "complete"
        
        state = load_resume_state(output_path)
        if state and state['samples_processed'] > 0:
            samples_processed = state['samples_processed']
            if samples_processed < input_total_samples:
                return True, samples_processed, f"resumable (processed {samples_processed}/{input_total_samples})"
            else:
                return False, 0, "possibly_corrupt"
    
    return False, 0, "new"

