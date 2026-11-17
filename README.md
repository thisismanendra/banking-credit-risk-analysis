# Banking and Credit Risk Analysis Assistant

**Team InsightX**

## Project Overview

A domain-focused banking and credit-risk research prototype
combining a custom Transformer experiment, historical
LendingClub data analysis and a local AI assistant.

## Technology

- Python
- PyTorch
- Transformers
- Ollama
- Qwen3 4B
- DuckDB
- Pandas
- Streamlit
- Hugging Face

## Main Features

- Banking and credit-risk explanations.
- Debt-to-income calculations.
- LendingClub historical default-rate analysis.
- Grade-wise Charged Off comparison.
- Exact historical dataset statistics.
- Evidence display and response caching.
- Local application through Ollama and Streamlit.

## Dataset

LendingClub accepted loans:

- Labelled records: 1,190,174.
- Fully Paid: 952,713.
- Charged Off: 237,461.

The full dataset is not included in this GitHub repository.
See `HUGGING_FACE.md` for the model and training dataset
repository references.

## Custom Transformer Research

An experimental 80.8M-parameter custom Transformer was built
and evaluated.

The V9 experiment achieved validation loss 1.1883, but
exploratory held-out outcome accuracy was 54% and profile-level
ROC-AUC was 0.5148.

This model is not a validated credit-risk predictor and is
not the deployed conversational language model.

## Deployed Assistant

The working local assistant uses a pretrained Qwen3 4B model
through Ollama, combined with the DuckDB data engine and
Python arithmetic.

## Run

1. Install Python dependencies:

   `python -m pip install -r requirements.txt`

2. Install Ollama for Windows.
3. Ensure the Ollama model `banking-assistant:final` exists.
4. Ensure the local LendingClub Parquet store is available.
5. Double-click `START_BANKING_ASSISTANT.bat`.

The application runs locally at:

`http://localhost:8503`

## Main Files

- `banking_app.py`: user interface.
- `banking_app_backend.py`: query routing.
- `banking_assistant_runtime.py`: core runtime.
- `banking_data_engine.py`: data analysis engine.
- `custom_slm_architecture.py`: custom architecture source.
- `notebooks/`: sanitized research notebook.
- `evaluation/`: experimental test reports.
- `ARCHITECTURE.md`: system architecture.
- `HUGGING_FACE.md`: model and dataset hosting references.

## Research Limitations

The historical outcome data describes resolved accepted
loans. It does not guarantee future performance.

This prototype is not intended for credit approvals,
individual credit scores or regulated financial decisions.

Qwen3 4B and the custom V9 Transformer are distinct models.
