import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_loader import download_data
from features import add_features
from dataset import MultiStockDataset
from model import MultiStockTransformer
from utils import calculate_metrics


def main():
    tickers = [
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "SBIN.NS",
        "0P0000XVUQ.BO",  # Parag Parikh (try)
        "0P0001B6QH.BO"  # WhiteOak (try)
    ]

    all_data = []
    stock_mapping = {}

    print("Downloading and processing data...")

    valid_idx = 0

    for ticker in tickers:
        try:
            df = download_data(ticker)

            if df is None or df.empty:
                print(f"Skipping {ticker} (no data)")
                continue

            df = add_features(df)

            if df.empty:
                print(f"Skipping {ticker} (no features)")
                continue

            df['Stock_ID'] = valid_idx
            stock_mapping[valid_idx] = ticker

            all_data.append(df)
            valid_idx += 1

        except Exception as e:
            print(f"Skipping {ticker} due to error: {e}")
            continue

    df = pd.concat(all_data)
    df.sort_index(inplace=True)

    feature_columns = df.columns.drop(['Returns', 'Stock_ID'])
    features = df[feature_columns].values
    stock_ids = df['Stock_ID'].values
    targets = df['Returns'].values * 10

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    split = int(len(features) * 0.8)

    train_dataset = MultiStockDataset(
        features[:split],
        stock_ids[:split]
    )

    test_dataset = MultiStockDataset(
        features[split:],
        stock_ids[split:]
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = MultiStockTransformer(
        input_dim=features.shape[1],
        num_stocks=len(tickers)
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 50

    print("Training started...")

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for X, stock_id, y in train_loader:
            optimizer.zero_grad()
            output = model(X, stock_id)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.6f}")

    print("Training complete.")

    torch.save(model.state_dict(), "multi_stock_transformer.pth")

    # =============================
    # 🔥 TEST SET METRICS
    # =============================

    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for X, stock_id, y in test_loader:
            output = model(X, stock_id)
            predictions.extend(output.numpy())
            actuals.extend(y.numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    mae, mse, rmse, rse = calculate_metrics(actuals, predictions)

    direction_acc = np.mean(
        np.sign(actuals) == np.sign(predictions)
    )

    print("\n===== Evaluation Metrics (Test Set - Returns) =====")
    print(f"MAE  : {mae:.6f}")
    print(f"MSE  : {mse:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"RSE  : {rse:.6f}")
    print(f"Direction Accuracy: {direction_acc:.4f}")

    # =============================
    # 🔥 MONTHLY PRICE GRAPH
    # =============================

    selected_stock_id = 0
    selected_ticker = stock_mapping[selected_stock_id]

    print(f"\nGenerating monthly prediction for {selected_ticker}")

    df_single = download_data(selected_ticker)
    df_single = add_features(df_single)

    feature_columns = df_single.columns.drop(['Returns'])
    features_single = df_single[feature_columns].values
    returns_single = df_single['Returns'].values
    close_prices = df_single['Close'].values

    features_single = scaler.transform(features_single)

    lookback = 30

    predicted_prices = []
    actual_prices = []
    dates = []

    for i in range(len(features_single) - lookback):

        X_seq = torch.tensor(
            features_single[i:i + lookback],
            dtype=torch.float32
        ).unsqueeze(0)

        stock_tensor = torch.tensor([selected_stock_id], dtype=torch.long)

        with torch.no_grad():
            pred_return = model(X_seq, stock_tensor).item() / 10
            pred_return = np.clip(pred_return, -0.1, 0.1)  # max ±10%

        current_price = close_prices[i + lookback - 1]
        real_price = close_prices[i + lookback]

        predicted_price = current_price * (1 + pred_return)

        predicted_prices.append(predicted_price)
        actual_prices.append(real_price)
        dates.append(df_single.index[i + lookback])

    # last 30 days
    predicted_prices = predicted_prices[-30:]
    actual_prices = actual_prices[-30:]
    dates = dates[-30:]

    plt.figure(figsize=(10, 5))

    plt.plot(dates, actual_prices, label="Actual Price", linewidth=2)
    plt.plot(dates, predicted_prices, label="Predicted Price", linestyle="--")

    plt.ylim(
        min(actual_prices) * 0.95,
        max(actual_prices) * 1.05
    )

    plt.xticks(rotation=45)
    plt.legend()
    plt.title(f"{selected_ticker} - Last Month Prediction")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
