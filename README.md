# MDS Tokenization & Chunking

Pipeline processing MDS dataset from HuggingFace: **Tokenization → Chunking → Upload**

## Quick Start

### 1. Tokenize MDS dataset

```bash
export HF_TOKEN="your_token_here"

python src/tokenization/tokenize_mds_subsets.py \
    --hf_repo QuangDuy/FineWiki-mds \
    --output_repo QuangDuy/FineWiki-mds-tokenized \
    --tokenizer_path /teamspace/studios/this_studio/data-processing/tokenizer/HUIT-BERT \
    --batch_size 10000 \
    --hf_token $HF_TOKEN
```

### 2. Chunk tokenized dataset (1K and/or 8K)

```bash
# 1K chunks
python src/sampling/chunk_tokenized_mds.py \
    --hf_repo QuangDuy/FineWiki-mds-tokenized \
    --output_repo QuangDuy/FineWiki-mds-tokenized-1024 \
    --tokenizer_path /teamspace/studios/this_studio/data-processing/tokenizer/HUIT-BERT \
    --chunk_size 1024 \
    --batch_size 5000 \
    --hf_token $HF_TOKEN

# 8K chunks
python src/sampling/chunk_tokenized_mds.py \
    --hf_repo QuangDuy/FineWiki-mds-tokenized \
    --output_repo QuangDuy/FineWiki-mds-tokenized-8192 \
    --tokenizer_path /teamspace/studios/this_studio/data-processing/tokenizer/HUIT-BERT \
    --chunk_size 8192 \
    --batch_size 5000 \
    --hf_token $HF_TOKEN
```

**Output:**
- `QuangDuy/FineWiki-mds-tokenized` (tokenized data)
- `QuangDuy/FineWiki-mds-tokenized-1024` (1K chunks)
- `QuangDuy/FineWiki-mds-tokenized-8192` (8K chunks)

## Individual Stages

### Tokenization Only
```bash
python src/tokenization/tokenize_mds_subsets.py \
    --hf_repo QuangDuy/FineWiki-mds \
    --output_repo QuangDuy/FineWiki-mds-tokenized \
    --tokenizer_path /path/to/tokenizer \
    --batch_size 80000 \
    --resume \
    --hf_token $HF_TOKEN \
    --upload_workers 16 \
    --upload_report_every 30
```

### Chunking Only (1K)
```bash
python src/sampling/chunk_tokenized_mds.py \
    --hf_repo QuangDuy/FineWiki-mds-tokenized \
    --output_repo QuangDuy/FineWiki-mds-tokenized-1024 \
    --tokenizer_path /path/to/tokenizer \
    --chunk_size 1024 \
    --batch_size 40000 \
    --resume \
    --hf_token $HF_TOKEN \
    --upload_workers 16 \
    --upload_report_every 30
```

### Chunking Only (4K)
```bash
python src/sampling/chunk_tokenized_mds.py \
    --input_dir ./data/FineWiki-mds-tokenized \
    --output_dir ./data/FineWiki-mds-tokenized-4096 \
    --tokenizer_path tokenizer/HUIT-BERT \
    --chunk_size 4096 \
    --min_chunk_size 1024 \
    --always_skip_size 128 \
    --batch_size 1000 \
    --resume \
    --hf_token $HF_TOKEN
  ```
### Chunking Only (8K)
```bash
python src/sampling/chunk_tokenized_mds.py \
    --hf_repo QuangDuy/FineWiki-mds-tokenized \
    --output_repo QuangDuy/FineWiki-mds-tokenized-8192 \
    --tokenizer_path /path/to/tokenizer \
    --chunk_size 8192 \
    --always_skip_size 32 \
    --batch_size 20000 \
    --resume \
    --hf_token $HF_TOKEN \
    --upload_workers 16 \
    --upload_report_every 30
```

## Merging and Splitting Datasets

Merge multiple MDS datasets and split them into train/val/train_small splits.

### Merge from Local Directories

```bash
python src/splitting/merge_mds_subsets.py \
    --source_dir ./source_data \
    --output_dir ./data \
    --train_ratio 0.9 \
    --val_ratio 0.1 \
    --train_small_ratio 0.05 \
    --datasets FineWeb2-vie-mds FineWiki-mds
```

### Merge from HuggingFace and Upload

```bash
python src/splitting/merge_mds_subsets.py \
    --hf_repos QuangDuy/FineWiki-mds QuangDuy/FineWeb2-vie-mds \
    --output_dir ./data \
    --output_repo QuangDuy/merged-dataset \
    --train_ratio 0.9 \
    --val_ratio 0.1 \
    --train_small_ratio 0.05 \
    --hf_token $HF_TOKEN \
    --upload_workers 16 \
    --upload_report_every 30
```

### Create Only train_small Split

```bash
python src/splitting/merge_mds_subsets.py \
    --hf_repos QuangDuy/FineWiki-mds \
    --output_dir ./data \
    --train_small_ratio 0.05 \
    --only-train-small
```

### Options

- `--source_dir`: Source directory containing dataset folders (when using local datasets)
- `--hf_repos`: HuggingFace repository IDs to download (space-separated)
- `--output_dir`: Output directory for merged splits (required)
- `--output_repo`: HuggingFace repository ID to upload splits to (optional)
- `--train_ratio`: Ratio of data for training (default: 0.9)
- `--val_ratio`: Ratio of data for validation (default: 0.1)
- `--train_small_ratio`: Ratio of train data for train_small split (default: 0.05)
- `--seed`: Random seed for reproducible splits (default: 42)
- `--no-symlinks`: Copy files instead of creating symlinks
- `--decompress`: Decompress shards to create raw_data
- `--only-train-small`: Only create train_small split without train/val

**Output Structure:**
```
output_dir/
├── train/
│   ├── index.json
│   └── shard.*.mds (or symlinks)
├── val/
│   ├── index.json
│   └── shard.*.mds
└── train_small/
    ├── index.json
    └── shard.*.mds
```

When `--output_repo` is specified, each split is uploaded as a separate HuggingFace dataset:
- `{output_repo}/train`
- `{output_repo}/val`
- `{output_repo}/train_small`

## Key Features

- Download & process MDS datasets from HuggingFace
- Tokenize with HUIT-BERT tokenizer (output: uint16)
- Chunk with backfill algorithm (no duplicates)
- Support 1K & 8K chunk sizes
- **Fast parallel upload to HuggingFace** (using `upload_large_folder` with configurable workers)
- **Parallel processing** (subset-level parallelization for faster processing)
- **Robust resume capability** (sample-level tracking with automatic recovery)
- Statistics tracking
- Preserve subset structure (XXX_XXXXX folders)

## Parallel Processing

Both tokenization and chunking now support parallel processing of multiple subsets simultaneously, significantly improving performance on multi-core systems.

### Usage

Add the `--num_workers` flag to control parallelization:

```bash
# Auto-detect optimal worker count (CPU cores - 1)
python src/tokenization/tokenize_mds_subsets.py \
    --hf_repo QuangDuy/FineWiki-mds \
    --output_repo QuangDuy/FineWiki-mds-tokenized \
    --tokenizer_path /path/to/tokenizer \
    --batch_size 80000 \
    --num_workers 2 \
    --hf_token $HF_TOKEN

# Explicitly set worker count
python src/sampling/chunk_tokenized_mds.py \
    --hf_repo QuangDuy/FineWiki-mds-tokenized \
    --output_repo QuangDuy/FineWiki-mds-tokenized-1024 \
    --tokenizer_path /path/to/tokenizer \
    --chunk_size 1024 \
    --num_workers 4 \
    --hf_token $HF_TOKEN

# Sequential processing (1 worker)
python src/tokenization/tokenize_mds_subsets.py \
    --input_dir ./data/FineWiki-mds \
    --output_dir ./data/FineWiki-mds-tokenized \
    --tokenizer_path /path/to/tokenizer \
    --num_workers 1
```

### Performance Notes

- **Default behavior**: Auto-detects `CPU cores - 1` workers
- **Best for**: Datasets with multiple subset folders
- **Memory**: Each worker processes one subset at a time
- **Compatible**: Works seamlessly with resume capability

## Configuration Presets

| Config | Chunk Size | Min Size | Skip Size | Use Case |
|--------|-----------|----------|-----------|----------|
| 1K     | 1024      | 512      | 128       | Stage 1 |
| 8K     | 4096      | 1024      | 128        | Stage 2 |

## Resume Capability

The pipeline now includes **robust resume capability** that tracks processing at the sample level, allowing interrupted jobs to continue seamlessly without restarting entire subsets.

### How It Works

- **Sample-level tracking**: Saves progress every 500-1000 samples
- **Automatic detection**: Detects complete, partial, or corrupted subsets
- **Clear feedback**: Shows exactly what's being resumed vs. processed fresh
- **Smart recovery**: Skips already-processed samples efficiently

### Using Resume Mode

Add the `--resume` flag to any tokenization or chunking command:

#### Resume Tokenization
```bash
python src/tokenization/tokenize_mds_subsets.py \
    --hf_repo QuangDuy/FineWiki-mds \
    --output_repo QuangDuy/FineWiki-mds-tokenized \
    --tokenizer_path tokenizer/HUIT-BERT \
    --batch_size 40000 \
    --resume \
    --hf_token $HF_TOKEN
```

#### Resume Chunking
```bash
python src/sampling/chunk_tokenized_mds.py \
    --hf_repo QuangDuy/FineWiki-mds-tokenized \
    --output_repo QuangDuy/FineWiki-mds-tokenized-1024 \
    --tokenizer_path tokenizer/HUIT-BERT \
    --chunk_size 1024 \
    --batch_size 20000 \
    --resume \
    --hf_token $HF_TOKEN
```

### When to Use Resume

- **Long-running jobs**: Multi-hour processing that might be interrupted
- **Unstable connections**: When downloading/uploading to HuggingFace
- **Resource constraints**: If job might be killed due to memory/time limits
- **Iterative development**: Testing pipeline changes without full reprocessing

### Technical Details

- Resume state stored in `.resume_state.json` files per subset
- Automatically cleaned up on successful completion
- Compatible with both local and HuggingFace workflows
- No performance penalty when not using resume mode

## Installation

```bash
pip install -r requirements.txt
```

## Reading Chunk Samples

Inspect and read samples from chunked datasets:

### Read a few samples
```bash
python src/utils/read_chunk_samples.py \
    --chunk_dir temp_chunking/chunked-1024 \
    --num_samples 5
```

### Read with tokenizer to decode text
```bash
python src/utils/read_chunk_samples.py \
    --chunk_dir temp_chunking/chunked-1024 \
    --tokenizer_path tokenizer/HUIT-BERT \
    --num_samples 10
```

### Read from specific subset
```bash
python src/utils/read_chunk_samples.py \
    --chunk_dir temp_chunking/chunked-1024 \
    --subset 000_00000 \
    --num_samples 5
```

### Read all samples in a subset
```bash
python src/utils/read_chunk_samples.py \
    --chunk_dir temp_chunking/chunked-1024 \
    --subset 000_00000 \
    --all
```

### Read all samples with tokenizer
```bash
python src/utils/read_chunk_samples.py \
    --chunk_dir temp_chunking/chunked-1024 \
    --tokenizer_path tokenizer/HUIT-BERT \
    --all
```

## File Structure

```
src/
├── utils/
│   ├── mds_helpers.py              # MDS utilities
│   └── read_chunk_samples.py       # Read and inspect chunk samples
├── tokenization/tokenize_mds_subsets.py   # Tokenization
├── sampling/chunk_tokenized_mds.py   # Chunking
├── splitting/merge_mds_subsets.py   # Merge and split datasets
```
