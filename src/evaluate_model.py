"""
evaluate.py — Compute BLEU and ROUGE scores for fine-tuned mT5-small.

Runs evaluation on:
  1. Hinglish held-out test split (fine-tuned model)
  2. Nyishi zero-shot evaluation (fine-tuned model, no Nyishi training)
  3. Optionally: Hinglish test split with raw baseline mT5-small

Writes a summary CSV to --output_csv (default: results/bleu_scores.csv).

Usage:
    python src/evaluate.py \
        --model_dir results/mt5-hinglish \
        --hinglish_csv data/hinglish_dataset.csv \
        --nyishi_csv data/nyishi_dataset.csv \
        --output_csv results/bleu_scores.csv \
        --run_baseline
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import evaluate as hf_evaluate
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from preprocess import (
    load_dataset_from_csv,
    prepare_hinglish_splits,
    prepare_nyishi_eval,
)


# ── Generation ────────────────────────────────────────────────────────────────

def generate_translations(
    model,
    tokenizer,
    dataset,
    batch_size: int = 8,
    max_new_tokens: int = 128,
    device: str = "cpu",
) -> list:
    """
    Run model.generate() over a tokenized HuggingFace Dataset in batches.
    Returns a list of decoded translation strings.
    """
    model.eval()
    model.to(device)

    dataset = dataset.with_format("torch")
    loader = DataLoader(dataset, batch_size=batch_size)

    all_predictions = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
            )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            all_predictions.extend(decoded)

    return all_predictions


def get_references_from_dataset(dataset, tokenizer) -> list:
    """Decode label token IDs (replacing -100) back into reference strings."""
    label_ids = np.array(dataset["labels"])
    label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
    references = tokenizer.batch_decode(label_ids.tolist(), skip_special_tokens=True)
    return references


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_bleu(predictions: list, references: list) -> float:
    metric = hf_evaluate.load("sacrebleu")
    result = metric.compute(
        predictions=predictions,
        references=[[ref] for ref in references],
    )
    return round(result["score"], 4)


def compute_rouge(predictions: list, references: list) -> dict:
    metric = hf_evaluate.load("rouge")
    result = metric.compute(predictions=predictions, references=references)
    return {k: round(v, 4) for k, v in result.items()}


# ── Main evaluation runner ────────────────────────────────────────────────────

def run_evaluation(
    model_dir: str,
    hinglish_csv: str,
    nyishi_csv: str,
    output_csv: str = "results/bleu_scores.csv",
    batch_size: int = 8,
    max_new_tokens: int = 128,
    run_baseline: bool = True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Load fine-tuned model ─────────────────────────────────────────────────
    print(f"\nLoading fine-tuned model from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    finetuned_model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    # ── Prepare datasets ──────────────────────────────────────────────────────
    print("Preparing Hinglish test split…")
    _, hinglish_test_ds = prepare_hinglish_splits(hinglish_csv, tokenizer, test_size=0.1)

    print("Preparing Nyishi evaluation set…")
    nyishi_ds = prepare_nyishi_eval(nyishi_csv, tokenizer)

    records = []

    # ── Fine-tuned: Hinglish test ─────────────────────────────────────────────
    print("\n[1/3] Evaluating fine-tuned model on Hinglish test split…")
    hinglish_refs = get_references_from_dataset(hinglish_test_ds, tokenizer)
    hinglish_preds = generate_translations(
        finetuned_model, tokenizer, hinglish_test_ds, batch_size, max_new_tokens, device
    )
    bleu = compute_bleu(hinglish_preds, hinglish_refs)
    rouge = compute_rouge(hinglish_preds, hinglish_refs)
    print(f"  BLEU: {bleu}  |  ROUGE-L: {rouge.get('rougeL')}")
    records.append({"model": "finetuned_mt5", "dataset": "hinglish_test", "bleu": bleu, **rouge})

    # ── Fine-tuned: Nyishi zero-shot ──────────────────────────────────────────
    print("\n[2/3] Zero-shot evaluation on Nyishi…")
    nyishi_refs = get_references_from_dataset(nyishi_ds, tokenizer)
    nyishi_preds = generate_translations(
        finetuned_model, tokenizer, nyishi_ds, batch_size, max_new_tokens, device
    )
    bleu = compute_bleu(nyishi_preds, nyishi_refs)
    rouge = compute_rouge(nyishi_preds, nyishi_refs)
    print(f"  BLEU: {bleu}  |  ROUGE-L: {rouge.get('rougeL')}")
    print("  Note: Low Nyishi BLEU is expected — the goal is non-zero transfer vs baseline.")
    records.append({"model": "finetuned_mt5", "dataset": "nyishi_zeroshot", "bleu": bleu, **rouge})

    # ── Baseline: raw mT5-small on Hinglish test ──────────────────────────────
    if run_baseline:
        print("\n[3/3] Evaluating baseline mT5-small (no fine-tuning) on Hinglish test…")
        baseline_tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
        baseline_model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
        _, hinglish_test_base = prepare_hinglish_splits(hinglish_csv, baseline_tokenizer, test_size=0.1)
        base_refs = get_references_from_dataset(hinglish_test_base, baseline_tokenizer)
        base_preds = generate_translations(
            baseline_model, baseline_tokenizer, hinglish_test_base, batch_size, max_new_tokens, device
        )
        bleu = compute_bleu(base_preds, base_refs)
        rouge = compute_rouge(base_preds, base_refs)
        print(f"  BLEU: {bleu}  |  ROUGE-L: {rouge.get('rougeL')}")
        records.append({"model": "baseline_mt5", "dataset": "hinglish_test", "bleu": bleu, **rouge})

    # ── Write results CSV ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    results_df = pd.DataFrame(records)
    # Reorder columns
    col_order = ["model", "dataset", "bleu", "rouge1", "rouge2", "rougeL", "rougeLsum"]
    results_df = results_df.reindex(columns=[c for c in col_order if c in results_df.columns])
    results_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate mT5-small on Hinglish and Nyishi")
    parser.add_argument("--model_dir", required=True, help="Path to fine-tuned model checkpoint")
    parser.add_argument("--hinglish_csv", required=True)
    parser.add_argument("--nyishi_csv", required=True)
    parser.add_argument("--output_csv", default="results/bleu_scores.csv")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--run_baseline",
        action="store_true",
        help="Also evaluate raw google/mt5-small as a baseline",
    )
    args = parser.parse_args()

    run_evaluation(
        model_dir=args.model_dir,
        hinglish_csv=args.hinglish_csv,
        nyishi_csv=args.nyishi_csv,
        output_csv=args.output_csv,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        run_baseline=args.run_baseline,
    )
