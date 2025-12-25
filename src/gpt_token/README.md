# GPT-NeoX-Style Vietnamese BPE Tokenizer

Train a GPT-NeoX-inspired BPE tokenizer with consistent space splitting and repeated space tokens for efficient encoding of Vietnamese text, code, and LaTeX.

## Features

### 🎯 Key Improvements over GPT-2

1. **Consistent Space Splitting**: Uses `add_prefix_space=True` for uniform tokenization - first word treated same as subsequent words
2. **Byte Fallback**: Enabled `byte_fallback=True` to handle unknown tokens by converting them to byte representations
3. **Repeated Space Tokens**: Special tokens for 1-24 consecutive spaces, making code and LaTeX encoding more efficient
4. **Vietnamese-Optimized**: Trained on diverse Vietnamese corpus (FineWiki + FineWeb2)
5. **BERT-Compatible**: Includes BERT-style special tokens (`[CLS]`, `[MASK]`, `[PAD]`, `[SEP]`, `[UNK]`)

### 📊 Default Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| vocab_size | 32,000 | Standard for medium-sized models |
| max_token_length | 64 | Balance between coverage and efficiency |
| max_consecutive_spaces | 24 | GPT-NeoX standard for code/LaTeX |
| special_tokens | 29 total | 5 BERT tokens + 24 space variants |

## Installation

Ensure you have the required dependencies:

```bash
uv add tokenizers transformers streaming huggingface-hub
```

## Usage

### Training from MDS Datasets

Train on Vietnamese MDS datasets from HuggingFace:

```bash
uv run python src/gpt_token/train_tokenizer.py \
    --mds_datasets QuangDuy/FineWiki-mds QuangDuy/FineWeb2-vie-mds \
    --tokenizer_output_path output/vietnamese_neox_32k.json \
    --hf_output_dir output/vietnamese_neox_32k_hf \
    --vocab_size 32000 \
    --max_token_length 64 \
    --max_consecutive_spaces 24 \
    --cache_dir data
```

### Training from JSONL Files

Train on local JSONL files:

```bash
uv run python src/gpt_token/train_tokenizer.py \
    --json_input_dir data/vietnamese_text \
    --tokenizer_output_path output/tokenizer.json \
    --vocab_size 32000
```

### Testing the Tokenizer

Test the trained tokenizer on various patterns:

```bash
uv run python src/gpt_token/test_neox_tokenizer.py \
    --tokenizer_path output_neox/vietnamese_neox_32k.json \
    --hf_tokenizer_dir output_neox/vietnamese_neox_32k_hf
```

## Command-Line Arguments

### Data Source (choose one)

- `--mds_datasets`: List of HuggingFace MDS dataset repo IDs
- `--json_input_dir`: Directory containing JSONL files

### Output

- `--tokenizer_output_path` **(required)**: Path for tokenizer JSON file
- `--hf_output_dir` *(optional)*: Directory to save HuggingFace format tokenizer

### Configuration

- `--vocab_size`: Vocabulary size (default: 32000)
- `--max_token_length`: Maximum token length for BPE (default: 64)
- `--max_consecutive_spaces`: Max consecutive spaces as special tokens (default: 24)
- `--cache_dir`: Cache directory for MDS datasets (default: "data")
- `--text_key`: Key to extract text from samples (default: "text")

## Python API

### Direct Usage

```python
from train_tokenizer import train_tokenizer

train_tokenizer(
    mds_datasets=["QuangDuy/FineWiki-mds", "QuangDuy/FineWeb2-vie-mds"],
    save_path="output/tokenizer.json",
    vocab_size=32000,
    max_token_length=64,
    max_consecutive_spaces=24,
    cache_dir="data",
    hf_output_dir="output/tokenizer_hf"
)
```

### Loading and Using

```python
from tokenizers import Tokenizer

# Load trained tokenizer
tokenizer = Tokenizer.from_file("output/tokenizer.json")

# Encode text
text = "Xin chào, đây là một ví dụ về tokenization."
encoded = tokenizer.encode(text)
print(f"Tokens: {encoded.ids}")

# Decode tokens
decoded = tokenizer.decode(encoded.ids)
print(f"Decoded: {decoded}")
```

### HuggingFace Format

```python
from transformers import PreTrainedTokenizerFast

# Load HuggingFace format tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained("output/tokenizer_hf")

# Use with special tokens
text = "[CLS] Vietnamese text [SEP]"
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)

# Access special tokens
print(f"CLS token ID: {tokenizer.cls_token_id}")
print(f"PAD token ID: {tokenizer.pad_token_id}")
```

## Architecture

### GPT-NeoX Approach

```
Input Text
    ↓
Custom Pre-tokenizer (ByteLevel, add_prefix_space=True)
    ↓
BPE Model (vocab_size=32k, max_token_length=64)
    ↓
Special Tokens:
  - BERT tokens: [CLS], [MASK], [PAD], [SEP], [UNK]
  - Space tokens: " ", "  ", "   ", ..., (24 spaces)
    ↓
Token IDs
```

### Consistent Space Splitting

**HuggingFace Best Practice**:
```python
# ByteLevel with add_prefix_space=True
tokenize("hello") == first_token_of(tokenize("world hello"))
# Consistent: first word treated same as subsequent words
# Space added to beginning if not present
```

**Why add_prefix_space=True?**:
- Ensures the first word is tokenized consistently with words in the middle of text
- The space preceding a word is part of the word representation (shown as 'Ġ')
- Standardizes tokenization regardless of word position in the sentence

### Efficient Whitespace Encoding

**Without repeated space tokens**:
```python
"    code" → [" ", " ", " ", " ", "c", "o", "d", "e"]  # 8 tokens
```

**With repeated space tokens**:
```python
"    code" → ["    ", "c", "o", "d", "e"]  # 5 tokens (37.5% reduction)
```

## Benefits

### ✅ Consistent Tokenization
- No ambiguity with prefix spaces
- Same text → same tokens, regardless of position
- Easier to reason about model behavior

### ✅ Efficient Whitespace Handling
- Fewer tokens for indented code
- Better compression for LaTeX documents
- Reduced sequence length for structured text

### ✅ Vietnamese Optimization
- Trained on diverse Vietnamese corpus
- Better subword segmentation for Vietnamese
- Improved handling of diacritics

### ✅ BERT Compatibility
- Can be used with BERT-style models
- Standard special token interface
- Compatible with HuggingFace ecosystem

## Examples

### Vietnamese Text
```python
tokenizer.encode("Việt Nam là một quốc gia Đông Nam Á.")
# Efficient encoding with proper diacritic handling
```

### Code with Indentation
```python
code = """
def function():
    if condition:
        return True
"""
tokenizer.encode(code)
# Fewer tokens due to repeated space tokens
```

### LaTeX
```python
latex = "\\begin{equation}\n    E = mc^2\n\\end{equation}"
tokenizer.encode(latex)
# Efficient whitespace encoding
```

## Comparison

| Feature | GPT-2 | GPT-NeoX | This Implementation |
|---------|-------|----------|---------------------|
| Space handling | Inconsistent | Consistent | ✅ Consistent |
| Repeated spaces | No special tokens | 1-24 space tokens | ✅ 1-24 space tokens |
| Special tokens | `<\|endoftext\|>` | Custom | ✅ BERT-style |
| Vietnamese | Not optimized | Not optimized | ✅ Optimized |
| MDS support | No | No | ✅ Yes |

## References

- [GPT-NeoX-20B Paper](https://arxiv.org/abs/2204.06745) - Original GPT-NeoX approach
- [HuggingFace Tokenizers](https://huggingface.co/docs/tokenizers/) - Tokenizers library documentation
- [Byte-Pair Encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding) - BPE algorithm explanation

## License

Apache License 2.0 (inherited from EleutherAI codebase)

