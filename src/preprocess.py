"""
preprocess.py — Data loading, normalization, and tokenization for the
Hinglish-English / Nyishi-English translation pipeline.

Supported file formats:
  - CSV  (comma-separated) with 'input' / 'output' columns
  - TSV  (tab-separated)   with 'input' / 'output' OR 'cs_query' / 'en_query' columns
    (the archive TSV files use cs_query=Hinglish, en_query=English)

Delimiter is detected automatically from the file extension.
Column names are auto-mapped: cs_query→input, en_query→output when present.

Usage (library):
    from src.preprocess import prepare_hinglish_splits, prepare_nyishi_eval
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

    # TSV from archive — columns auto-mapped, no extra args needed
    train_ds, val_ds = prepare_hinglish_splits(
        "archive (1)/Human Annotated Data/train.tsv", tokenizer
    )
    # Old CSV format still works unchanged
    train_ds, val_ds = prepare_hinglish_splits("data/hinglish_dataset.csv", tokenizer)
"""

import re
import unicodedata

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

# ── Task prefixes ────────────────────────────────────────────────────────────
HINGLISH_PREFIX = "translate Hinglish to English: "
NYISHI_PREFIX = "translate Nyishi to English: "
ENGLISH_TO_HINGLISH_PREFIX = "translate English to Hinglish: "

TASK_PREFIXES = {
    "hinglish": HINGLISH_PREFIX,
    "nyishi": NYISHI_PREFIX,
    "english_to_hinglish": ENGLISH_TO_HINGLISH_PREFIX,
}

# ── Text normalization ────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Lowercase, strip control characters, collapse whitespace."""
    if not isinstance(text, str):
        text = str(text)
    # Normalize unicode (NFC form)
    text = unicodedata.normalize("NFC", text)
    # Remove control characters (except regular whitespace)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cc" or ch in " \t\n")
    # Lowercase
    text = text.lower()
    # Collapse multiple whitespace into single space and strip ends
    text = re.sub(r"\s+", " ", text).strip()
    return text


def add_prefix(text: str, prefix: str) -> str:
    return prefix + text


# ── File loading ─────────────────────────────────────────────────────────────

def load_dataset_from_file(
    file_path: str,
    input_col: str = "input",
    output_col: str = "output",
) -> pd.DataFrame:
    """
    Load a CSV or TSV translation file and return a DataFrame with
    normalized 'input' (source) and 'output' (target) columns.

    Delimiter detection:
        .tsv extension  → tab-separated
        anything else   → comma-separated
        If the detected delimiter yields only 1 column, the other is tried.

    Column mapping:
        Uses input_col / output_col as-is when present.
        If those columns are missing but cs_query + en_query exist,
        they are automatically mapped (cs_query→input, en_query→output).

    Extra columns (en_parse, cs_parse, domain, …) are silently dropped.
    """
    # ── Delimiter detection ───────────────────────────────────────────────────
    sep = "\t" if file_path.lower().endswith(".tsv") else ","
    df = pd.read_csv(file_path, sep=sep)

    # Fallback: if only one column was parsed, the delimiter was wrong
    if len(df.columns) == 1:
        alt_sep = "," if sep == "\t" else "\t"
        df = pd.read_csv(file_path, sep=alt_sep)

    # ── Column mapping ────────────────────────────────────────────────────────
    if input_col not in df.columns or output_col not in df.columns:
        # Auto-map from LinCE / archive TSV column names
        if "cs_query" in df.columns and "en_query" in df.columns:
            print(f"  Auto-mapping cs_query→input, en_query→output in '{file_path}'")
            df = df.rename(columns={"cs_query": "input", "en_query": "output"})
            input_col, output_col = "input", "output"
        else:
            raise ValueError(
                f"'{file_path}' must have '{input_col}' and '{output_col}' columns "
                f"(or 'cs_query'/'en_query'). Found: {list(df.columns)}"
            )

    df = df[[input_col, output_col]].copy()
    df = df.rename(columns={input_col: "input", output_col: "output"})
    df = df.dropna().reset_index(drop=True)
    df["input"] = df["input"].apply(normalize_text)
    df["output"] = df["output"].apply(normalize_text)
    return df


def load_dataset_from_csv(csv_path: str) -> pd.DataFrame:
    """Backward-compatible alias for load_dataset_from_file."""
    return load_dataset_from_file(csv_path)


# ── Tokenization ──────────────────────────────────────────────────────────────

def tokenize_dataset(
    df: pd.DataFrame,
    tokenizer,
    prefix: str,
    max_seq_len: int = 128,
) -> Dataset:
    """
    Tokenize a DataFrame into a HuggingFace Dataset ready for Seq2SeqTrainer.

    - Encoder inputs: prefix + source text, truncated to max_seq_len
    - Decoder labels: target text, truncated to max_seq_len
    - Padding token IDs in labels are replaced with -100 so loss ignores them
    - padding=False here; DataCollatorForSeq2Seq handles dynamic batch padding
    """
    dataset = Dataset.from_pandas(df)

    def preprocess_fn(examples):
        # Prepend task prefix to each source sentence
        inputs = [add_prefix(src, prefix) for src in examples["input"]]
        targets = examples["output"]

        model_inputs = tokenizer(
            inputs,
            max_length=max_seq_len,
            truncation=True,
        )

        labels = tokenizer(
            text_target=list(targets),
            max_length=max_seq_len,
            truncation=True,
        )

        # Replace padding token id with -100 so CrossEntropyLoss ignores it
        label_ids = [
            [(token_id if token_id != tokenizer.pad_token_id else -100) for token_id in ids]
            for ids in labels["input_ids"]
        ]
        model_inputs["labels"] = label_ids
        return model_inputs

    tokenized = dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )
    return tokenized


# ── High-level helpers ────────────────────────────────────────────────────────

def prepare_hinglish_splits(
    file_path: str,
    tokenizer,
    input_col: str = "input",
    output_col: str = "output",
    test_size: float = 0.1,
    max_seq_len: int = 128,
    seed: int = 42,
) -> tuple:
    """
    Load a Hinglish-English file (CSV or TSV), split into train/val, tokenize.
    Returns (train_dataset, val_dataset).

    For archive TSV files the cs_query/en_query columns are auto-mapped;
    no need to pass input_col/output_col explicitly in that case.
    """
    df = load_dataset_from_file(file_path, input_col=input_col, output_col=output_col)
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=seed)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_ds = tokenize_dataset(train_df, tokenizer, HINGLISH_PREFIX, max_seq_len)
    val_ds = tokenize_dataset(val_df, tokenizer, HINGLISH_PREFIX, max_seq_len)
    return train_ds, val_ds


def prepare_nyishi_eval(
    file_path: str,
    tokenizer,
    input_col: str = "input",
    output_col: str = "output",
    max_seq_len: int = 128,
) -> Dataset:
    """
    Load a Nyishi-English file (CSV or TSV) and tokenize with Nyishi prefix.
    Used only for zero-shot evaluation — never for training.
    """
    df = load_dataset_from_file(file_path, input_col=input_col, output_col=output_col)
    return tokenize_dataset(df, tokenizer, NYISHI_PREFIX, max_seq_len)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description="Preprocess and inspect a translation file (CSV or TSV)")
    parser.add_argument("--csv", required=True, help="Path to input file (.csv or .tsv)")
    parser.add_argument("--task", choices=list(TASK_PREFIXES.keys()), default="hinglish")
    parser.add_argument("--input_col", default="input", help="Source column name (default: 'input'; TSV auto-detects cs_query)")
    parser.add_argument("--output_col", default="output", help="Target column name (default: 'output'; TSV auto-detects en_query)")
    parser.add_argument("--max_seq_len", type=int, default=128)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    df = load_dataset_from_file(args.csv, input_col=args.input_col, output_col=args.output_col)
    print(f"Loaded {len(df)} rows from {args.csv}")

    ds = tokenize_dataset(df, tokenizer, TASK_PREFIXES[args.task], args.max_seq_len)
    print(f"Tokenized dataset: {ds}")
    print("Sample features:", {k: ds[0][k][:8] for k in ds.column_names})
