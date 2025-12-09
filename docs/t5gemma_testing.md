# T5Gemma Model Support - Testing Guide

This document provides instructions for testing the T5Gemma architecture support in CTranslate2.

## Overview

T5Gemma is a family of encoder-decoder large language models developed by Google, combining the T5 encoder-decoder architecture with Gemma 2 improvements including:
- Grouped Query Attention (GQA)
- Rotary Position Embeddings (RoPE)
- GeGLU activation
- RMSNorm
- Interleaved local/global attention

## Quick Test with Google Colab

You can test the T5Gemma support using the following Google Colab notebook template:

```python
# Install CTranslate2 from the development branch
# Note: Once merged, use: pip install ctranslate2
!pip install git+https://github.com/jncraton/CTranslate2.git@copilot/support-t5gemma-architecture

# Install transformers with T5Gemma support (requires recent version)
!pip install transformers>=4.50.0 torch

# Convert a T5Gemma model
import ctranslate2

model_name = "harshaljanjani/tiny-t5gemma-test"

converter = ctranslate2.converters.TransformersConverter(
    model_name,
    trust_remote_code=True
)

output_dir = converter.convert("ct2_t5gemma_model")
print(f"Model converted successfully to: {output_dir}")

# Test translation/generation
translator = ctranslate2.Translator(output_dir)

# Example: Translate English to German
source = ["translate English to German: The house is wonderful."]
results = translator.translate_batch(
    [s.split() for s in source]
)

print("\nTranslation results:")
for result in results:
    print(" ".join(result.hypotheses[0]))
```

## Testing Locally

### Prerequisites

1. Build CTranslate2 from source with the T5Gemma support:
```bash
git clone https://github.com/jncraton/CTranslate2.git
cd CTranslate2
git checkout copilot/support-t5gemma-architecture

# Build the Python package
pip install -e python/
```

2. Install dependencies:
```bash
pip install transformers>=4.50.0 torch
```

### Convert a Model

```python
import ctranslate2

# Convert the test model
converter = ctranslate2.converters.TransformersConverter(
    "harshaljanjani/tiny-t5gemma-test",
    trust_remote_code=True
)

output_dir = converter.convert("ct2_t5gemma_model")
```

### Run Inference

```python
import ctranslate2

# Load the converted model
translator = ctranslate2.Translator("ct2_t5gemma_model")

# Prepare input
source_tokens = [["translate", "English", "to", "German", ":", "Hello", "world", "."]]

# Translate
results = translator.translate_batch(source_tokens)

# Display results
for result in results:
    print(" ".join(result.hypotheses[0]))
```

## Testing with Python Wheels

If you want to test with pre-built wheels (for distribution):

1. Build the wheel:
```bash
cd CTranslate2/python
python setup.py bdist_wheel
```

2. The wheel will be in `python/dist/`. Install it:
```bash
pip install dist/ctranslate2-*.whl
```

3. Test the installation:
```python
import ctranslate2
from ctranslate2.converters.transformers import _MODEL_LOADERS

assert "T5GemmaConfig" in _MODEL_LOADERS
print("T5Gemma support is available!")
```

## Expected Behavior

- The converter should successfully load and convert T5Gemma models without errors
- The converted model should preserve the encoder-decoder architecture
- Inference should work with both translation and generation tasks
- The model should support:
  - Variable length inputs
  - Beam search
  - Sampling strategies
  - All standard CTranslate2 inference features

## Supported Models

The following T5Gemma models should be compatible:
- `google/t5gemma-2b-2b-prefixlm-it`
- `google/t5gemma-8b-8b-prefixlm-it`
- `harshaljanjani/tiny-t5gemma-test` (recommended for testing)

## Troubleshooting

### Import Error
If you get `ModuleNotFoundError: No module named 'transformers'`:
```bash
pip install transformers>=4.50.0
```

### Config Not Found
If you get `KeyError: 'T5GemmaConfig'`:
- Ensure you're using the correct branch/version of CTranslate2
- Verify the installation with: `python -c "from ctranslate2.converters.transformers import _MODEL_LOADERS; print('T5GemmaConfig' in _MODEL_LOADERS)"`

### Model Download Issues
If you have network issues downloading models:

**Option 1: Manual download with git-lfs**
```bash
# Install git-lfs if not already installed
git lfs install

# Clone the repository
git clone https://huggingface.co/harshaljanjani/tiny-t5gemma-test

# If clone fails, try pulling LFS files separately
cd tiny-t5gemma-test
git lfs pull

# Use the local path
python -c "
import ctranslate2
converter = ctranslate2.converters.TransformersConverter('./tiny-t5gemma-test')
converter.convert('ct2_model')
"
```

**Option 2: Use huggingface-cli with caching**
```bash
# Download with retry logic
huggingface-cli download harshaljanjani/tiny-t5gemma-test --local-dir ./tiny-t5gemma-test

# Convert from local directory
python -c "
import ctranslate2
converter = ctranslate2.converters.TransformersConverter('./tiny-t5gemma-test')
converter.convert('ct2_model')
"
```

**Option 3: Download during stable network hours**
- HuggingFace may have temporary connectivity issues
- Retry during off-peak hours (typically late night UTC)
- Use a stable, high-bandwidth connection

**Option 4: Use smaller test models first**
- Test with smaller T5 models to verify setup
- Then graduate to T5Gemma models once connectivity is stable

## Performance Expectations

T5Gemma models benefit from CTranslate2's optimizations:
- Reduced memory footprint compared to transformers
- Faster inference through layer fusion and optimized kernels
- Support for quantization (INT8, INT16, FP16) for further speedups
- Efficient batching and parallel execution

## Contributing

If you find issues with T5Gemma support:
1. Check the [GitHub Issues](https://github.com/jncraton/CTranslate2/issues)
2. Provide a minimal reproduction example
3. Include model details and error messages
