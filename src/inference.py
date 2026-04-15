"""
inference.py — Single-sentence and batch inference for the mT5 translation pipeline.

Importable by streamlit_app.py; also runnable as a CLI script.

Usage:
    # Single sentence
    python src/inference.py \
        --model_dir results/mt5-hinglish \
        --task hinglish \
        --text "kal milte hain"

    # Batch (CSV in → CSV out)
    python src/inference.py \
        --model_dir results/mt5-hinglish \
        --task nyishi \
        --input_csv data/nyishi_dataset.csv \
        --output_csv results/nyishi_predictions.csv
"""

import argparse
import os

import torch
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from preprocess import TASK_PREFIXES, normalize_text, add_prefix

# ── Module-level model cache ──────────────────────────────────────────────────
# Keyed by model_dir so multiple model paths can coexist in the same process.
_MODEL_CACHE: dict = {}


def load_model(model_dir: str):
    """Load (and cache) tokenizer + model from model_dir."""
    if model_dir not in _MODEL_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        model.eval()
        _MODEL_CACHE[model_dir] = (tokenizer, model)
    return _MODEL_CACHE[model_dir]


# ── Core translation function ─────────────────────────────────────────────────

def translate(
    text: str,
    task: str,
    model,
    tokenizer,
    max_new_tokens: int = 128,
    device: str = "cpu",
) -> str:
    """
    Translate a single string.

    Args:
        text:   Source text (without prefix).
        task:   One of 'hinglish', 'nyishi', 'english_to_hinglish'.
        model:  Loaded AutoModelForSeq2SeqLM.
        tokenizer: Matching tokenizer.
    Returns:
        Translated string.
    """
    if task not in TASK_PREFIXES:
        raise ValueError(f"Unknown task '{task}'. Choose from: {list(TASK_PREFIXES.keys())}")

    prefix = TASK_PREFIXES[task]
    source = add_prefix(normalize_text(text), prefix)

    inputs = tokenizer(source, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def translate_batch(
    texts: list,
    task: str,
    model,
    tokenizer,
    batch_size: int = 8,
    max_new_tokens: int = 128,
    device: str = "cpu",
) -> list:
    """Translate a list of strings in batches."""
    if task not in TASK_PREFIXES:
        raise ValueError(f"Unknown task '{task}'. Choose from: {list(TASK_PREFIXES.keys())}")

    prefix = TASK_PREFIXES[task]
    sources = [add_prefix(normalize_text(t), prefix) for t in texts]

    model.to(device)
    model.eval()
    results = []

    for i in range(0, len(sources), batch_size):
        batch = sources[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        results.extend(decoded)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run translation inference with mT5")
    parser.add_argument("--model_dir", required=True, help="Path to model checkpoint directory")
    parser.add_argument(
        "--task",
        required=True,
        choices=list(TASK_PREFIXES.keys()),
        help="Translation direction",
    )
    parser.add_argument("--text", help="Single sentence to translate")
    parser.add_argument("--input_csv", help="CSV with 'input' column for batch translation")
    parser.add_argument("--output_csv", help="Path to write predictions CSV")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    if not args.text and not args.input_csv:
        parser.error("Provide --text for single inference or --input_csv for batch mode.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_model(args.model_dir)

    if args.text:
        result = translate(args.text, args.task, model, tokenizer, args.max_new_tokens, device)
        print(f"Input : {args.text}")
        print(f"Output: {result}")

    elif args.input_csv:
        df = pd.read_csv(args.input_csv)
        if "input" not in df.columns:
            raise ValueError("Input CSV must have an 'input' column.")

        predictions = translate_batch(
            df["input"].tolist(),
            args.task,
            model,
            tokenizer,
            args.batch_size,
            args.max_new_tokens,
            device,
        )
        df["prediction"] = predictions

        out_path = args.output_csv or args.input_csv.replace(".csv", "_predictions.csv")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved {len(predictions)} predictions to {out_path}")
