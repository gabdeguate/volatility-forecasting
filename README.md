# Crypto Volatility Forecasting

Compares GARCH, LSTM, and Transformer models for forecasting BTC daily volatility. Built as an ML engineering portfolio project.

## Results

| Model | MSE | MAE |
|---|---|---|
| GARCH(1,1) | 4.15e-05 | 0.00540 |
| Transformer | 7.59e-06 | 0.00225 |
| LSTM | **2.56e-06** | **0.00114** |

LSTM outperforms Transformer on this task. Short, autocorrelated financial time series favor the sequential inductive bias of LSTMs over attention-based models. Both deep learning models beat GARCH by a wide margin on magnitude accuracy.

## Data

Daily BTC-USD prices from 2018-01-31 to 2024-12-30 via `yfinance`. Features:

- **Log returns**: `log(P_t / P_{t-1})`
- **30-day realized volatility**: rolling standard deviation of log returns

Train: 2018-2022 | Val: 2023 | Test: 2024

## Models

**GARCH(1,1)** — statistical baseline using the `arch` library. Rolling window evaluation: refit daily on all available history, forecast one step ahead. Parameters: α=0.087, β=0.838 (persistence = 0.924).

**LSTM** — two-layer LSTM, hidden size 64, dropout 0.2, 30-day input window. Input features are StandardScaler normalized. Softplus output activation to guarantee positive volatility predictions.

**Transformer** — two encoder layers, `d_model=64`, 4 attention heads, CLS token aggregation, learned positional embeddings, Softplus output. Same input normalization as LSTM.

## Setup

```bash
git clone https://github.com/gabdeguate/volatility-forecasting
cd volatility-forecasting
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Run notebooks in order:

```
notebooks/01_eda.ipynb         # data pipeline and EDA
notebooks/02_garch.ipynb       # GARCH baseline
notebooks/03_lstm.ipynb        # LSTM model
notebooks/04_transformer.ipynb # Transformer model
```

## Project Structure

```
volatility-forecasting/
├── data/
│   └── btc_data.csv
├── models/
│   ├── lstm.py
│   └── transformer.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_garch.ipynb
│   ├── 03_lstm.ipynb
│   └── 04_transformer.ipynb
├── api/
├── dashboard/
├── requirements.txt
└── README.md
```

## Key Findings

- Volatility clustering is strong in BTC data (α+β=0.924), consistent with financial literature
- BTC log returns show excess kurtosis — fat tails mean extreme events occur more often than a normal distribution predicts
- LSTM's persistence bias (predicting values close to yesterday) is partially corrected by input normalization
- Directional accuracy is a poor metric for smooth rolling volatility targets; MSE and MAE are the appropriate evaluation criteria
