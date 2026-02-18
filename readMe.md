# Multi-Stock Transformer for Time-Series Forecasting

## Overview
This project trains a Transformer-based deep learning model to forecast next-day stock returns using historical OHLCV data and technical indicators.

The implementation uses one shared model across multiple tickers. This allows the network to learn:
- temporal market structure through self-attention
- stock-specific behavior through learned stock embeddings

The codebase includes data collection, feature engineering, sequence dataset creation, model training with early stopping, test-set evaluation, and deployment artifact export.

## How the Project Works
### 1. Data Acquisition and Caching
`data_loader.py` downloads historical data from Yahoo Finance and caches each ticker in `data/<ticker>.csv`.

Default date range in the current implementation:
- `start="2020-01-01"`
- `end="2024-01-01"`

### 2. Feature Engineering
`features.py` adds technical signals using the `ta` library:
- `SMA_10`
- `EMA_10`
- `RSI` (14)
- `MACD`
- `Returns` (percent change of close price)

Rows with missing indicator values are removed.

### 3. Multi-Stock Sequence Dataset
`dataset.py` builds sliding windows per stock:
- input: `LOOKBACK_WINDOW` consecutive feature rows
- target: next-day scaled return (`Returns_Scaled`)
- metadata: stock id (`Stock_ID`) per sequence

This keeps each sequence internally consistent by ticker while still enabling joint training.

### 4. Transformer Model
`model.py` defines `MultiStockTransformer` with:
- linear feature embedding
- learned stock embedding
- learned positional encoding
- Transformer encoder stack
- temporal mean pooling
- final linear head with `tanh` output

The output is a bounded return prediction, which helps stabilize extreme predictions.

### 5. Training and Validation
`train.py` orchestrates the pipeline:
- concatenates all ticker data into one training frame
- chronologically splits train/test
- fits `StandardScaler` on train features only
- creates train/validation subsets from training data
- trains with Adam + MSE loss
- applies early stopping using:
  - `EARLY_STOPPING_PATIENCE`
  - `EARLY_STOPPING_MIN_DELTA`
### 6. Evaluation
The test workflow computes:
- MAE
- MSE
- RMSE
- RSE
- Direction Accuracy (sign match between actual and predicted returns)

Metrics are printed overall and approximately per stock. A run summary is appended to `summary.txt`.

### 7. Artifacts and Visualization
After training:
- model is saved to `deployment_artifacts/multi_stock_transformer.pkl`
- scaler is saved to `deployment_artifacts/scaler.pkl`
- stock mapping is saved to `deployment_artifacts/stock_mapping.json`

The script also plots predicted vs actual prices for recent points of randomly selected tickers.

## Project Structure
```text
stock_prediction_transformers/
|-- data/                         # Cached ticker CSV files
|-- deployment_artifacts/         # Exported model/scaler/mapping
|-- data_loader.py                # Download and local caching
|-- features.py                   # Technical feature engineering
|-- dataset.py                    # Sliding-window dataset for multi-stock sequences
|-- model.py                      # Transformer architecture
|-- train.py                      # End-to-end training/evaluation pipeline
|-- utils.py                      # Metric functions
|-- requirements.txt              # Python dependencies
|-- readMe.md                     # Project documentation
```

## Installation
1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
pip install ta
```

Note: `ta` is used in `features.py` and should be present in the environment.

## Usage
Run training from the project root:

```bash
python train.py
```

You can adjust experiment settings in `train.py` via the `config` dictionary, including:
- tickers
- lookback window
- learning rate
- batch size
- train split ratio
- early stopping parameters
- artifact output directory

## Applications
This project can support:
- research prototypes for multi-asset return forecasting
- signal generation for quantitative strategy experimentation
- comparative studies between Transformers and recurrent models (LSTM/GRU)
- feature ablation and model sensitivity analysis on technical indicators

It is best suited for educational and research workflows, and not as a standalone production trading system without additional risk, execution, and monitoring layers.

## Future Improvements
1. Implement full walk-forward validation inside the fold loop (currently scaffolded).
2. Add transaction costs, slippage, and position sizing in a proper backtest.
3. Evaluate finance-specific metrics (Sharpe, Sortino, max drawdown, hit ratio).
4. Expand target designs (multi-horizon regression, volatility-aware targets, classification heads).
5. Add hyperparameter optimization and reproducible experiment tracking.
6. Include richer features (market regime proxies, volume profile, macro variables).
7. Introduce uncertainty estimation for prediction confidence.
8. Improve per-stock evaluation indexing for exact sequence-level attribution.
9. Add model versioning and inference scripts for deployment workflows.
10. Add unit/integration tests for data preparation, dataset generation, and metrics.

## Notes
- This repository currently includes trained model files (`.pth`/`.pkl`) and summary logs from previous runs.
- Ensure dependencies and dataset periods align with your experiment goals before retraining.
