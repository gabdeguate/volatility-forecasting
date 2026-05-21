# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python -m venv quant && source quant/bin/activate
pip install -r requirements.txt
```

Venv is `quant/`, already committed. Always activate before running notebooks or scripts.

## Running Notebooks

Run in order — each notebook saves artifacts consumed by the next:

```
notebooks/01_eda.ipynb         # fetches BTC-USD via yfinance, saves data/btc_data.csv
notebooks/02_garch.ipynb       # fits GARCH(1,1), saves data/garch_results.csv
notebooks/03_lstm.ipynb        # trains LSTM, saves models/mse_lstm_model.pth and models/garch_lstm_model.pth
notebooks/04_transformer.ipynb # trains Transformer, saves models/transformer_model.pth
```

Launch Jupyter: `jupyter notebook` from repo root with venv active.

## Architecture

**Data flow**: `yfinance` → log returns + 30-day realized vol → `StandardScaler` normalized → 30-day sliding windows → model input.

**Target**: next-day realized volatility (positive scalar). All models use `Softplus` output to guarantee positivity.

**Model definitions** live in `models/lstm.py` and `models/transformer.py`. Notebooks import these directly.

**LSTM** (`models/lstm.py`): 2-layer LSTM, hidden=64, dropout=0.2. Takes `(batch, seq=30, features=2)`, outputs `(batch, 1)`. Uses last timestep hidden state.

**Transformer** (`models/transformer.py`): 2 encoder layers, d_model=64, 4 heads. Prepends learnable CLS token; learned positional embeddings (seq_len+1 positions). CLS output → linear → Softplus.

**GARCH baseline**: fit via `arch` library in `02_garch.ipynb`. Rolling window — refit daily on all history, forecast one step ahead. No saved model file; `data/garch_results.csv` stores predictions.

## Train/Val/Test Split

- Train: 2018–2022
- Val: 2023
- Test: 2024

Data fetched from `yfinance` at notebook runtime; `data/btc_data.csv` caches it.

## Key Results

| Model | MSE | MAE |
|---|---|---|
| GARCH(1,1) | 4.15e-05 | 0.00540 |
| Transformer | 7.59e-06 | 0.00225 |
| LSTM | 2.56e-06 | 0.00114 |

LSTM wins on MSE/MAE. Directional accuracy is not a meaningful metric here — target is smooth rolling vol, not raw returns.

## Planned (Not Yet Built)

`api/` and `dashboard/` dirs exist but are empty. Intended: FastAPI serving saved `.pth` models + Streamlit dashboard. Stack is already in `requirements.txt` (fastapi, uvicorn, streamlit).
