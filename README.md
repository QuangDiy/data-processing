# MDS Tokenization & Chunking Pipeline

Pipeline processing MDS dataset from HuggingFace: **Tokenization → Chunking → Upload**

## Quick Start

### 1. Test Pipeline (Recommended First)

```bash
python src/pipeline/test_pipeline.py
```

### 2. Run Full Pipeline

```bash
export HF_TOKEN="your_token_here"

python src/pipeline/run_full_pipeline.py \
    --input_repo QuangDuy/FineWiki-mds \
    --tokenizer_path /teamspace/studios/this_studio/data-processing/tokenizer/HUIT-BERT \
    --output_prefix QuangDuy/FineWiki-mds \
    --batch_size_tokenize 10000 \
    --batch_size_chunk 5000 \
    --hf_token $HF_TOKEN
```

**Output:**
- `QuangDuy/FineWiki-mds-tokenized` (tokenized data)
- `QuangDuy/FineWiki-mds-tokenized-1024` (1K chunks)
- `QuangDuy/FineWiki-mds-tokenized-8192` (8K chunks)

## Pipeline Flow

```
Raw MDS (HF) → Tokenization (HUIT-BERT) → Tokenized MDS
                                              ├→ Chunking 1K → Chunked 1024
                                              └→ Chunking 8K → Chunked 8192
```

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
| 8K     | 8192      | 512      | 32        | Stage 2 |

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
└── pipeline/
    ├── run_full_pipeline.py          # Full pipeline
    └── test_pipeline.py              # Test suite
```
