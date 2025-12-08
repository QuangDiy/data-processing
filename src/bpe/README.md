# Vietnamese BPE Tokenizer Training

Vietnamese-optimized BPE tokenizer training using the `bpeasy` library with enhanced regex patterns for Vietnamese language.

## Features

- Vietnamese-specific regex patterns (abbreviations, diacritics, special characters)
- Automatic dataset downloading from HuggingFace
- Multiple output formats (BPEasy JSON, HuggingFace, Tiktoken)
- Comprehensive test suite

## Quick Start

```bash
# 1. Download datasets
python src/bpe/download_datasets.py

# 2. Train tokenizer (32k vocab, max length 16)
python src/bpe/train_vietnamese_bpe.py

# 3. Verify tokenizer
python src/bpe/train_vietnamese_bpe.py --verify_only --tokenizer_path ./output/vietnamese_bpe.json

# 4. Evaluate tokenizer (compare fertility and PCW metrics)
python src/bpe/evaluate_tokenizer.py --tokenizer_path ./output/vietnamese_bpe_hf_tokenizer
```

## Evaluation

Evaluate your tokenizer using standard metrics:

- **Fertility**: Average number of tokens per word (lower = better)
- **PCW (Proportion of Continued Words)**: Percentage of words split into 2+ tokens (lower = better)

```bash
# Evaluate your tokenizer on Vietnamese and English Wikipedia
python src/bpe/evaluate_tokenizer.py --tokenizer_path ./output/vietnamese_bpe_hf_tokenizer

# Compare with reference tokenizers (Llama3, Gemma3, Qwen3, PhoGPT, etc.)
python src/bpe/evaluate_tokenizer.py --tokenizer_path ./output/vietnamese_bpe_hf_tokenizer --compare

# Use fewer samples for faster evaluation
python src/bpe/evaluate_tokenizer.py --tokenizer_path ./output/vietnamese_bpe_hf_tokenizer --num_samples 1000

# Compare with reference tokenizers
python src/bpe/evaluate_tokenizer.py --tokenizer_path "D:\workspace\data-processing\output_eng\vietnamese_bpe_bert_hf_tokenizer" --compare_paths "D:\workspace\data-processing\output\vietnamese_bpe_bert_hf_tokenizer" --num_samples 1000
```

**Required dependencies for evaluation:**
```bash
pip install datasets sentencepiece 'datatrove[multilingual]'
```

## Usage

### Training Options

```bash
# Custom vocab size and max length
python src/bpe/train_vietnamese_bpe.py --vocab_size 50000 --max_token_length 32

# Limited samples (for testing)
python src/bpe/train_vietnamese_bpe.py --max_samples 100000

# Custom output directory
python src/bpe/train_vietnamese_bpe.py --output_dir /path/to/output
```

### Using the Tokenizer

```python
from src.bpe.bpeasy.tokenizer import BPEasyTokenizer

# Load tokenizer
tokenizer = BPEasyTokenizer.from_file("./output/vietnamese_bpe.json")

# Encode/decode
text = "Tp. Hồ Chí Minh là thành phố lớn nhất Việt Nam."
tokens = tokenizer.encode(text)
decoded = tokenizer.decode(tokens)
```

### HuggingFace Format

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./output/vietnamese_bpe_hf")
tokens = tokenizer.encode("Tiếng Việt")
```

## Files

- `train_vietnamese_bpe.py` - Main training script
- `vietnamese_word_tokenizer.py` - Vietnamese regex patterns
- `download_datasets.py` - Dataset downloading utility
- `evaluate_tokenizer.py` - Tokenizer evaluation (fertility, PCW metrics)
- `test_tokenizer_quick.py` - Quick tokenizer test
- `bpeasy/` - Tokenizer wrapper module

## Configuration

**Defaults:**
- Vocabulary size: 32,000
- Max token length: 16
- Datasets: QuangDuy/FineWiki-mds, QuangDuy/FineWeb2-vie-mds
- Output: `./output/`

**Command-line options:** See `--help` for each script

## Troubleshooting

**Dataset not found:** Run `download_datasets.py` first

**Memory issues:** Reduce `--batch_size` or use `--max_samples`

**Verification failures:** Check Unicode normalization and regex patterns

## Credits

This project uses code based on the [bpeasy](https://github.com/gautierdag/bpeasy) library by [Gautier Dag](https://github.com/gautierdag).

- **BPEasy**: Fast BPE tokenizer training in Rust with Python bindings
- **License**: MIT
- **Repository**: https://github.com/gautierdag/bpeasy

