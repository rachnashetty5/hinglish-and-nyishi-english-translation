"""
train.py — Fine-tune mT5-small on Hinglish-English translation.

Uses HuggingFace Seq2SeqTrainer with generation-based BLEU evaluation.
The saved checkpoint can later be used for zero-shot Nyishi evaluation.

Usage:
    # TSV from archive (cs_query/en_query auto-mapped, no extra flags needed)
    python src/train.py \
        --hinglish_csv "archive (1)/Human Annotated Data/train.tsv" \
        --output_dir results/mt5-hinglish

    # Explicit column names
    python src/train.py \
        --hinglish_csv "archive (1)/Human Annotated Data/train.tsv" \
        --input_col cs_query --output_col en_query

    # Original CSV format (unchanged)
    python src/train.py \
        --hinglish_csv data/hinglish_dataset.csv \
        --output_dir results/mt5-hinglish \
        --epochs 5 --batch_size 8 --lr 3e-5
"""

import argparse
import os

import numpy as np
import sacrebleu as sb
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from preprocess import prepare_hinglish_splits


def build_compute_metrics(tokenizer):
    """Return a compute_metrics function bound to the given tokenizer."""

    def compute_metrics(eval_pred):
        predictions, label_ids = eval_pred

        # Decode predictions
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 (ignored padding) with pad_token_id before decoding
        label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # sacrebleu.corpus_bleu expects references as a list-of-lists
        result = sb.corpus_bleu(decoded_preds, [decoded_labels])
        return {"bleu": round(result.score, 4)}

    return compute_metrics


def train(
    hinglish_csv: str,
    output_dir: str,
    input_col: str = "input",
    output_col: str = "output",
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 3e-5,
    max_seq_len: int = 128,
    seed: int = 42,
    model_name: str = "google/mt5-small",
):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    print(f"Preparing dataset from: {hinglish_csv}")
    train_ds, val_ds = prepare_hinglish_splits(
        hinglish_csv, tokenizer,
        input_col=input_col, output_col=output_col,
        max_seq_len=max_seq_len, seed=seed,
    )
    print(f"  Train samples : {len(train_ds)}")
    print(f"  Val   samples : {len(val_ds)}")

    # Dynamic padding — paired with padding=False in preprocess.py
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        optim="adamw_torch",
        # Generation-based evaluation: decode output tokens and compute BLEU
        # (not teacher-forced logits — gives meaningful translation quality signal)
        predict_with_generate=True,
        generation_max_length=max_seq_len,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        logging_steps=50,
        # fp16 disabled: mT5 is known to produce NaN gradients with fp16
        # due to overflow in the embedding layer on some hardware.
        # Enable manually with --fp16 if you have a confirmed stable GPU setup.
        fp16=False,
        seed=seed,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,   # replaces deprecated tokenizer= (transformers ≥4.46)
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
    )

    print("Starting training…")
    trainer.train()

    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune mT5-small on Hinglish→English")
    parser.add_argument("--hinglish_csv", required=True,
        help="Path to Hinglish data file (.csv or .tsv). "
             "TSV files with cs_query/en_query columns are auto-mapped.")
    parser.add_argument("--output_dir", default="results/mt5-hinglish")
    parser.add_argument("--input_col", default="input",
        help="Source column name (default: 'input'; TSV auto-detects cs_query)")
    parser.add_argument("--output_col", default="output",
        help="Target column name (default: 'output'; TSV auto-detects en_query)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", default="google/mt5-small")
    args = parser.parse_args()

    train(
        hinglish_csv=args.hinglish_csv,
        output_dir=args.output_dir,
        input_col=args.input_col,
        output_col=args.output_col,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        model_name=args.model_name,
    )
