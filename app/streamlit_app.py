"""
streamlit_app.py — Web interface for the mT5 multilingual translation pipeline.

Run with:
    streamlit run app/streamlit_app.py

The model is loaded once and cached via @st.cache_resource to avoid
reloading on every interaction.
"""

import sys
import os

# Allow imports from src/ when running from the project root or app/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
from inference import load_model, translate, TASK_PREFIXES

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Multilingual Translator — mT5",
    page_icon="🌐",
    layout="centered",
)

# ── Model loading (cached) ────────────────────────────────────────────────────

DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(__file__), "..", "results", "mt5-hinglish"
)

@st.cache_resource(show_spinner="Loading translation model…")
def get_model(model_dir: str):
    return load_model(model_dir)


# ── Sidebar: settings ─────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")
    model_dir = st.text_input(
        "Model directory",
        value=DEFAULT_MODEL_DIR,
        help="Path to the fine-tuned mT5 checkpoint",
    )
    st.markdown("---")
    st.markdown(
        "**Tasks:**\n"
        "- `hinglish` — Hinglish → English\n"
        "- `nyishi` — Nyishi → English *(zero-shot)*\n"
        "- `english_to_hinglish` — English → Hinglish"
    )

# ── Main UI ───────────────────────────────────────────────────────────────────

st.title("Multilingual Translator")
st.caption("Powered by fine-tuned mT5-small · Zero-shot Nyishi support")

TASK_LABELS = {
    "hinglish": "Hinglish → English",
    "nyishi": "Nyishi → English (zero-shot)",
    "english_to_hinglish": "English → Hinglish",
}

task = st.selectbox(
    "Translation direction",
    options=list(TASK_LABELS.keys()),
    format_func=lambda k: TASK_LABELS[k],
)

input_text = st.text_area(
    "Source text",
    placeholder="Enter text to translate…",
    height=120,
)

translate_btn = st.button("Translate", type="primary", use_container_width=True)

if translate_btn:
    if not input_text.strip():
        st.warning("Please enter some text to translate.")
    else:
        if not os.path.isdir(model_dir):
            st.error(
                f"Model directory not found: `{model_dir}`\n\n"
                "Run training first:\n"
                "```\npython src/train.py --hinglish_csv data/hinglish_dataset.csv\n```"
            )
        else:
            try:
                tokenizer, model = get_model(model_dir)
                with st.spinner("Translating…"):
                    result = translate(input_text, task, model, tokenizer)
                st.markdown("### Translation")
                st.success(result)
            except Exception as e:
                st.error(f"Translation failed: {e}")
