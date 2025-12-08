#!/usr/bin/env python3
"""
Download Vietnamese datasets from HuggingFace for BPE training.

This script downloads the specified Vietnamese datasets (FineWiki-mds and FineWeb2-vie-mds)
from HuggingFace and saves them to the local data directory.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.mds_helpers import (
    download_from_hf,
    verify_mds_structure,
    count_total_samples,
    discover_subset_folders
)


DEFAULT_DATASETS = [
    "QuangDuy/FineWiki-mds",
    "QuangDuy/FineWeb2-vie-mds",
    "QuangDuy/FineWiki-eng-mds"
]


def download_datasets(
    datasets: list[str],
    output_dir: Path,
    hf_token: str = None
) -> dict:
    """
    Download multiple datasets from HuggingFace.
    
    Args:
        datasets: List of HuggingFace dataset repository IDs
        output_dir: Base directory to save datasets
        hf_token: Optional HuggingFace API token
        
    Returns:
        Dictionary with download statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {
        'datasets': {},
        'total_samples': 0,
        'total_subsets': 0
    }
    
    print("=" * 70)
    print("Downloading Vietnamese Datasets for BPE Training")
    print("=" * 70)
    print()
    
    for repo_id in datasets:
        print(f"Downloading: {repo_id}")
        print("-" * 70)
        
        dataset_name = repo_id.split('/')[-1]
        dataset_path = output_dir / dataset_name
        
        try:
            local_path = download_from_hf(
                repo_id=repo_id,
                cache_dir=dataset_path,
                token=hf_token
            )
            
            print(f"Downloaded to: {local_path}")
            
            print("Verifying MDS structure...")
            is_valid = verify_mds_structure(local_path)
            
            if not is_valid:
                print(f"WARNING: MDS structure verification failed for {repo_id}")
                stats['datasets'][repo_id] = {'error': 'Invalid MDS structure'}
                continue
            
            subset_folders = discover_subset_folders(local_path)
            total_samples = count_total_samples(local_path)
            
            stats['datasets'][repo_id] = {
                'path': str(local_path),
                'num_subsets': len(subset_folders),
                'total_samples': total_samples,
                'status': 'success'
            }
            
            stats['total_samples'] += total_samples
            stats['total_subsets'] += len(subset_folders)
            
            print(f"Successfully downloaded {repo_id}")
            print(f"  - Subsets: {len(subset_folders)}")
            print(f"  - Samples: {total_samples:,}")
            print()
            
        except Exception as e:
            print(f"Failed to download {repo_id}: {e}")
            stats['datasets'][repo_id] = {'error': str(e)}
            print()
            continue
    
    print("=" * 70)
    print("Download Summary")
    print("=" * 70)
    print(f"Total datasets: {len(datasets)}")
    print(f"Successful: {sum(1 for d in stats['datasets'].values() if d.get('status') == 'success')}")
    print(f"Failed: {sum(1 for d in stats['datasets'].values() if 'error' in d)}")
    print(f"Total subsets: {stats['total_subsets']}")
    print(f"Total samples: {stats['total_samples']:,}")
    print()
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download Vietnamese datasets from HuggingFace for BPE training"
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=DEFAULT_DATASETS,
        help=f'HuggingFace dataset repository IDs (default: {" ".join(DEFAULT_DATASETS)})'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data',
        help='Output directory for downloaded datasets (default: ./data)'
    )
    parser.add_argument(
        '--hf_token',
        type=str,
        default=None,
        help='HuggingFace API token (optional)'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    try:
        stats = download_datasets(
            datasets=args.datasets,
            output_dir=output_dir,
            hf_token=args.hf_token
        )
        
        failed = [repo for repo, data in stats['datasets'].items() if 'error' in data]
        if failed:
            print(f"WARNING: {len(failed)} dataset(s) failed to download:")
            for repo in failed:
                print(f"  - {repo}: {stats['datasets'][repo]['error']}")
            return 1
        
        print("All datasets downloaded successfully!")
        return 0
        
    except Exception as e:
        print(f"\nError during download: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
