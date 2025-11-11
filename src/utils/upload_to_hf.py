"""
Upload large folders to HuggingFace Hub
"""
import argparse
import os
from huggingface_hub import HfApi


def upload_folder_to_hf(
    repo_id: str,
    folder_path: str,
    hf_token: str = None,
    repo_type: str = "dataset",
    num_workers: int = 16,
    print_report: bool = True,
    print_report_every: int = 30,
):
    """
    Upload a large folder to HuggingFace Hub
    
    Args:
        repo_id: HuggingFace repository ID (e.g., "username/repo-name")
        folder_path: Local path to the folder to upload
        hf_token: HuggingFace token (if None, will use HF_TOKEN env variable)
        repo_type: Type of repo ("dataset", "model", or "space")
        num_workers: Number of parallel upload workers
        print_report: Whether to print upload progress
        print_report_every: Print report every N seconds
    """
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token is None:
            raise ValueError(
                "HuggingFace token not provided. "
                "Either pass --hf_token or set HF_TOKEN environment variable"
            )
    
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder path does not exist: {folder_path}")
    
    if not os.path.isdir(folder_path):
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    print(f"Uploading folder: {folder_path}")
    print(f"Target repo: {repo_id} ({repo_type})")
    print(f"Workers: {num_workers}")
    
    api = HfApi(token=hf_token)
    
    try:
        api.upload_large_folder(
            repo_id=repo_id,
            folder_path=folder_path,
            repo_type=repo_type,
            num_workers=num_workers,
            print_report=print_report,
            print_report_every=print_report_every,
        )
        print(f"Upload completed successfully!")
        print(f"View at: https://huggingface.co/{repo_type}s/{repo_id}")
    except Exception as e:
        print(f"Upload failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Upload large folders to HuggingFace Hub"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HuggingFace repository ID (e.g., 'username/repo-name')"
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        required=True,
        help="Local path to the folder to upload"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token (default: use HF_TOKEN env variable)"
    )
    parser.add_argument(
        "--repo_type",
        type=str,
        default="dataset",
        choices=["dataset", "model", "space"],
        help="Type of repository (default: dataset)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of parallel upload workers (default: 16)"
    )
    parser.add_argument(
        "--print_report_every",
        type=int,
        default=30,
        help="Print progress report every N seconds (default: 30)"
    )
    
    args = parser.parse_args()
    
    upload_folder_to_hf(
        repo_id=args.repo_id,
        folder_path=args.folder_path,
        hf_token=args.hf_token,
        repo_type=args.repo_type,
        num_workers=args.num_workers,
        print_report=True,
        print_report_every=args.print_report_every,
    )


if __name__ == "__main__":
    main()

