#!/usr/bin/env python3
"""
Example script demonstrating T5Gemma model conversion and inference with CTranslate2.

This script shows how to:
1. Convert a T5Gemma model from Hugging Face transformers
2. Run inference with the converted model
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Convert and test T5Gemma models with CTranslate2"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="harshaljanjani/tiny-t5gemma-test",
        help="Hugging Face model name or path to local model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ct2_t5gemma_model",
        help="Output directory for converted model",
    )
    parser.add_argument(
        "--source-text",
        type=str,
        default="translate English to German: The house is wonderful.",
        help="Source text for testing",
    )
    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        help="Skip conversion and only run inference",
    )

    args = parser.parse_args()

    try:
        import ctranslate2
    except ImportError:
        print("Error: ctranslate2 is not installed.", file=sys.stderr)
        print("Install it with: pip install ctranslate2", file=sys.stderr)
        sys.exit(1)

    # Verify T5Gemma support
    try:
        from ctranslate2.converters.transformers import _MODEL_LOADERS

        if "T5GemmaConfig" not in _MODEL_LOADERS:
            print(
                "Error: T5Gemma support is not available in this CTranslate2 version.",
                file=sys.stderr,
            )
            sys.exit(1)
        print("✓ T5Gemma support is available")
    except Exception as e:
        print(f"Error checking T5Gemma support: {e}", file=sys.stderr)
        sys.exit(1)

    # Convert model
    if not args.skip_conversion:
        print(f"\nConverting model: {args.model}")
        print(f"Output directory: {args.output_dir}")

        try:
            converter = ctranslate2.converters.TransformersConverter(
                args.model, trust_remote_code=True
            )
            output_dir = converter.convert(args.output_dir)
            print(f"✓ Model converted successfully to: {output_dir}")
        except Exception as e:
            print(f"Error during conversion: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            sys.exit(1)
    else:
        output_dir = args.output_dir
        print(f"\nSkipping conversion, using existing model at: {output_dir}")

    # Run inference
    print("\nRunning inference...")
    print(f"Source text: {args.source_text}")

    try:
        translator = ctranslate2.Translator(output_dir)

        # Tokenize source text (simple whitespace split for demo)
        source_tokens = [args.source_text.split()]

        results = translator.translate_batch(source_tokens)

        print("\nTranslation results:")
        for i, result in enumerate(results):
            print(f"  Result {i + 1}: {' '.join(result.hypotheses[0])}")

        print("\n✓ Inference completed successfully")

    except Exception as e:
        print(f"Error during inference: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
