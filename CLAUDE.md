CLAUDE.md

Project Title

Zero-Shot Cross-Dialect Transfer: Leveraging T5 for Code-Switched Hinglish and Nyishi-English Translation

⸻

Project Overview

This project develops a multilingual neural machine translation system using the mT5 transformer model to perform translation between:
	•	Hinglish → English
	•	English → Hinglish
	•	Nyishi → English
	•	English → Nyishi (optional extension)

The system focuses on enabling zero-shot transfer learning, where knowledge gained from Hinglish-English translation helps improve performance on Nyishi-English translation despite limited Nyishi training data.

The goal is to demonstrate how transformer-based models can support low-resource languages and mixed-language (code-switched) inputs.

⸻

Objectives
	1.	Build a translation pipeline using a pretrained multilingual transformer (mT5).
	2.	Fine-tune the model on Hinglish-English parallel datasets.
	3.	Evaluate zero-shot performance on Nyishi-English translation.
	4.	Compare baseline vs fine-tuned model outputs.
	5.	Measure translation quality using BLEU and ROUGE scores.
	6.	Deploy a simple inference interface (optional: Streamlit app).

⸻

Motivation

Code-switched languages such as Hinglish are widely used in informal communication but remain underrepresented in NLP systems. Nyishi is a low-resource indigenous language with limited digital corpora.

Traditional neural translation systems require large parallel datasets. This project explores whether multilingual transfer learning enables translation even when datasets are scarce.

This contributes toward inclusive AI systems supporting regional and underrepresented languages.

⸻

Dataset Description

Hinglish-English Dataset

Possible sources:
	•	LinCE Code-Switching Dataset
	•	Kaggle Hinglish-English datasets
	•	Custom curated conversational Hinglish corpus

Sample format:

Input: mujhe coffee chahiye
Output: I want coffee

Nyishi-English Dataset

Since Nyishi is low-resource:
	•	small manually collected sentence pairs
	•	dictionary-based mapping
	•	translated sample corpus

Target size:

200–500 sentence pairs (sufficient for demonstration purposes)

⸻

Data Preprocessing Pipeline

Steps:
	1.	Text normalization
	2.	Lowercasing
	3.	Removal of special characters
	4.	Sentence alignment
	5.	Tokenization using T5 tokenizer
	6.	Prefix-based task formatting

Example formatting:

translate Hinglish to English: kal milte hain

translate Nyishi to English: 

⸻

Model Architecture

Model used:

mT5-small (Multilingual Text-to-Text Transfer Transformer)

Architecture pipeline:

Input Sentence
↓
Task Prefix Injection
↓
Tokenizer
↓
mT5 Encoder
↓
mT5 Decoder
↓
Translated Output

The model treats all NLP tasks as text-to-text transformation problems.

⸻

Baseline Model

Baseline approaches used for comparison:
	1.	Pretrained mT5 (without fine-tuning)
	2.	Google Translate outputs (optional benchmark)

These establish reference translation performance before domain adaptation.

⸻

Proposed Methodology

Step 1: Load pretrained mT5-small

Step 2: Format dataset using task-specific prompts

Example:

translate Hinglish to English: tum kaha ho

Step 3: Fine-tune model on Hinglish-English dataset

Step 4: Perform zero-shot inference on Nyishi-English dataset

Step 5: Evaluate translation quality

⸻

Training Configuration

Recommended hyperparameters:

Model: mT5-small
Batch Size: 8
Epochs: 5
Learning Rate: 3e-5
Optimizer: AdamW
Max Sequence Length: 128
Tokenizer: T5Tokenizer
Framework: HuggingFace Transformers

Environment:

Python 3.10+
PyTorch
Transformers
Datasets
Evaluate

Optional:

Google Colab GPU

⸻

Evaluation Metrics

Translation performance measured using:

BLEU Score
Measures n-gram overlap between predicted and reference translations

ROUGE Score
Measures recall overlap of generated sequences

Optional:

METEOR Score
Sentence-level semantic similarity

Comparison table example:

Model	BLEU Score
Baseline	XX
Fine-tuned mT5	XX


⸻

Zero-Shot Transfer Strategy

The model is trained primarily on Hinglish-English translation pairs.

During inference, the trained model is evaluated on Nyishi-English translation without explicit Nyishi training exposure.

This demonstrates cross-lingual transfer capability of transformer architectures.

⸻

Implementation Workflow

Step 1: Dataset collection

Step 2: Preprocessing and formatting

Step 3: Tokenization

Step 4: Model fine-tuning

Step 5: Zero-shot evaluation

Step 6: Metric computation

Step 7: Result comparison

Step 8: Visualization (optional graphs)

⸻

Expected Outcomes
	1.	Improved translation accuracy after fine-tuning
	2.	Demonstration of multilingual transfer learning
	3.	Functional translation pipeline
	4.	Performance comparison charts
	5.	Support for low-resource dialect translation

⸻

Optional Extensions

Possible enhancements:

Language detection module before translation

Streamlit-based web interface

Attention visualization

Comparison with mBART model

Larger multilingual dataset integration

⸻

Project Directory Structure

Suggested layout:

project/
│
├── data/
│   ├── hinglish_dataset.csv
│   ├── nyishi_dataset.csv
│
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── training.ipynb
│
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│
├── results/
│   ├── bleu_scores.csv
│
├── app/
│   ├── streamlit_app.py
│
└── README.md

⸻

Reproducibility Details

Include:

Exact dataset sources
Hyperparameters
Training steps
Evaluation scripts
Environment configuration

This ensures experimental consistency and transparency.

⸻

Future Scope

Extend translation support to additional Indian code-switched dialects

Increase Nyishi dataset size

Experiment with larger transformer variants

Deploy as multilingual chatbot assistant

Integrate speech-to-text pipeline for spoken dialect translation

⸻

Conclusion

This project demonstrates how multilingual transformer models can enable translation for mixed-language and low-resource linguistic settings using zero-shot transfer learning.

The approach highlights the effectiveness of transfer learning in improving accessibility for underrepresented languages while maintaining scalable architecture for future extensions.