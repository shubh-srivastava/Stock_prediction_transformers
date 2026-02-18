import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import joblib
from datetime import datetime
import random
import copy

from data_loader import download_data
from features import add_features
from dataset import MultiStockDataset
from model import MultiStockTransformer
from utils import calculate_metrics


def load_and_preprocess_data(tickers, target_scale_factor):
    """Loads, preprocesses, and combines data for all tickers."""
    # --- Configuration ---
    # Using constants for hyperparameters makes the code cleaner and easier to modify.
    # LOOKBACK_WINDOW = 30
    # TRAIN_SPLIT_RATIO = 0.8
    # BATCH_SIZE = 32
    # EPOCHS = 50
    # LEARNING_RATE = 1e-4
    # TARGET_SCALE_FACTOR = 10.0

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

    df['Returns_Scaled'] = df['Returns'] * target_scale_factor
    return df, stock_mapping


def train_model_with_early_stopping(config, train_dataset, valid_dataset, num_features, num_stocks):
    """Trains the model with an early stopping mechanism."""
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty. Check split ratios and lookback window.")
    if len(valid_dataset) == 0:
        raise ValueError("Validation dataset is empty. Increase data, lower LOOKBACK_WINDOW, or reduce VALIDATION_SPLIT_RATIO.")

    train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config["BATCH_SIZE"], shuffle=False)

    model = MultiStockTransformer(
        input_dim=num_features,
        num_stocks=num_stocks,
        sequence_length=config["LOOKBACK_WINDOW"]
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    print("Training started...")

    # --- Training Loop ---
    for epoch in range(config["EPOCHS"]):
        model.train()
        total_train_loss = 0

        for X, stock_id, y in train_loader:
            optimizer.zero_grad()
            output = model(X, stock_id)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X, stock_id, y in valid_loader:
                output = model(X, stock_id)
                val_loss = criterion(output, y)
                total_val_loss += val_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(valid_loader)

        print(f"Epoch {epoch+1}/{config['EPOCHS']}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # Early stopping check
        # Improvement is significant if val_loss decreases by more than min_delta
        if best_val_loss - avg_val_loss > config["EARLY_STOPPING_MIN_DELTA"]:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save the best model state in memory, not to disk yet
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"Validation loss improved significantly.")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config["EARLY_STOPPING_PATIENCE"]:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs: No significant improvement for {config['EARLY_STOPPING_PATIENCE']} consecutive epochs.")
            break

    # Load the best model state
    if best_model_state is not None:
        print(f"\nLoading best model state from epoch with validation loss: {best_val_loss:.6f}")
        model.load_state_dict(best_model_state)
    else:
        print("\nWarning: Early stopping did not find a better model. Using the last model state.")

    return model


def evaluate_on_test_set(model, test_loader, stock_mapping, target_scale_factor):
    """Evaluates the model on the test set and prints overall and per-stock metrics."""
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for X, stock_id, y in test_loader:
            output = model(X, stock_id)
            # De-scale for metric calculation
            predictions.extend(output.numpy() / target_scale_factor)
            actuals.extend(y.numpy() / target_scale_factor)

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    mae, mse, rmse, rse = calculate_metrics(actuals, predictions)
    direction_acc = np.mean(np.sign(actuals) == np.sign(predictions))

    metrics = {
        "MAE": mae, "MSE": mse, "RMSE": rmse, "RSE": rse, "Direction Accuracy": direction_acc
    }

    print("\n===== Overall Evaluation Metrics (Test Set - Returns) =====")
    for name, value in metrics.items():
        print(f"{name:<20}: {value:.6f}")

    # Per-stock metrics
    print("\n===== Per-Stock Evaluation (Test Set) =====")
    test_stock_ids_flat = []
    for _, s_ids, _ in test_loader:
        test_stock_ids_flat.extend(s_ids.numpy())

    unique_test_ids = np.unique(test_stock_ids_flat)

    for stock_id in unique_test_ids:
        ticker = stock_mapping.get(stock_id, f"ID_{stock_id}")
        # This is an approximation; a more precise way would be to track indices from the dataset
        # but for a general idea, this works.
        indices = [i for i, sid in enumerate(test_stock_ids_flat) if sid == stock_id]

        if not indices:
            continue

        stock_preds = predictions[indices]
        stock_actuals = actuals[indices]

        if len(stock_preds) > 0:
            _, _, stock_rmse, _ = calculate_metrics(stock_actuals, stock_preds)
            stock_dir_acc = np.mean(np.sign(stock_actuals) == np.sign(stock_preds))
            print(f"--- {ticker:<15} | RMSE: {stock_rmse:.4f}, Direction Acc: {stock_dir_acc:.4f}")

    return metrics


def plot_price_predictions(model, scaler, stock_mapping, selected_stock_id, config):
    """Generates and displays a plot of predicted vs actual prices for a single stock."""
    selected_ticker = stock_mapping[selected_stock_id]
    print(f"\nGenerating monthly prediction for {selected_ticker}")

    df_single = download_data(selected_ticker)
    df_single = add_features(df_single)

    if df_single.empty:
        print(f"Could not generate plot for {selected_ticker}, no data.")
        return

    feature_columns = df_single.columns.drop(['Returns', 'Returns_Scaled'], errors='ignore')
    features_single = df_single[feature_columns].values
    close_prices = df_single['Close'].values

    features_single = scaler.transform(features_single)

    predicted_prices = []
    actual_prices = []
    dates = []

    for i in range(len(features_single) - config["LOOKBACK_WINDOW"]):
        X_seq = torch.tensor(
            features_single[i:i + config["LOOKBACK_WINDOW"]],
            dtype=torch.float32
        ).unsqueeze(0)

        stock_tensor = torch.tensor([selected_stock_id], dtype=torch.long)

        with torch.no_grad():
            # De-scale the predicted return
            pred_return = model(X_seq, stock_tensor).item() / config["TARGET_SCALE_FACTOR"]
            pred_return = np.clip(pred_return, -0.1, 0.1)  # max ±10%

        # Predict price for the NEXT day
        current_price = close_prices[i + config["LOOKBACK_WINDOW"] - 1]
        real_price = close_prices[i + config["LOOKBACK_WINDOW"]]

        predicted_price = current_price * (1 + pred_return)

        predicted_prices.append(predicted_price)
        actual_prices.append(real_price)
        dates.append(df_single.index[i + config["LOOKBACK_WINDOW"]])

    # Plot last 30 days of predictions
    predicted_prices = predicted_prices[-30:]
    actual_prices = actual_prices[-30:]
    dates = dates[-30:]

    plt.figure(figsize=(10, 5))

    plt.plot(dates, actual_prices, label="Actual Price", linewidth=2)
    plt.plot(dates, predicted_prices, label="Predicted Price", linestyle="--")

    plt.xticks(rotation=45)
    plt.legend()
    plt.title(f"{selected_ticker} - Last Month Prediction")
    plt.tight_layout()
    plt.show()


def save_artifacts(model, scaler, stock_mapping, directory):
    """Saves model, scaler, and stock mapping for deployment."""
    os.makedirs(directory, exist_ok=True)

    # Save full model object
    torch.save(model, os.path.join(directory, "multi_stock_transformer.pkl"))

    # Save scaler
    joblib.dump(scaler, os.path.join(directory, "scaler.pkl"))

    # Save stock mapping
    with open(os.path.join(directory, "stock_mapping.json"), 'w') as f:
        json.dump(stock_mapping, f, indent=4)

    print(f"\nDeployment artifacts saved to '{directory}' directory.")


def save_summary(filename, metrics, config, notes=""):
    """Appends a summary of the run to a text file."""
    with open(filename, 'a') as f:
        f.write(f"--- Summary @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f.write(f"Config: {config}\n")
        if notes:
            f.write(f"Notes: {notes}\n")
        f.write("Overall Metrics:\n")
        for key, value in metrics.items():
            f.write(f"  {key:<20}: {value:.6f}\n")
        f.write("-" * 30 + "\n\n")


def run_single_split_training(config, df, stock_mapping):
    """Runs the standard train-validate-test workflow."""
    feature_columns = df.columns.drop(['Returns', 'Stock_ID', 'Returns_Scaled'])
    features = df[feature_columns].values
    targets = df['Returns_Scaled'].values
    stock_ids = df['Stock_ID'].values

    # Chronological train-test split
    split_idx = int(len(features) * config["TRAIN_SPLIT_RATIO"])
    train_val_features, test_features = features[:split_idx], features[split_idx:]
    train_val_targets, test_targets = targets[:split_idx], targets[split_idx:]
    train_val_stock_ids, test_stock_ids = stock_ids[:split_idx], stock_ids[split_idx:]

    # Scale features
    scaler = StandardScaler()
    train_val_features_scaled = scaler.fit_transform(train_val_features)
    test_features_scaled = scaler.transform(test_features)

    # Create full train+validation dataset, then split for validation
    full_train_dataset = MultiStockDataset(
        train_val_features_scaled, train_val_stock_ids, train_val_targets,
        sequence_length=config["LOOKBACK_WINDOW"]
    )
    
    # Create a validation split from the end of the training data
    val_split_idx = int(len(full_train_dataset) * (1 - config["VALIDATION_SPLIT_RATIO"]))
    train_indices = list(range(val_split_idx))
    val_indices = list(range(val_split_idx, len(full_train_dataset)))
    
    train_subset = Subset(full_train_dataset, train_indices)
    valid_subset = Subset(full_train_dataset, val_indices)

    # Train the model
    model = train_model_with_early_stopping(
        config, train_subset, valid_subset,
        num_features=train_val_features_scaled.shape[1],
        num_stocks=len(stock_mapping)
    )

    # Evaluate on the hold-out test set
    test_dataset = MultiStockDataset(
        test_features_scaled, test_stock_ids, test_targets,
        sequence_length=config["LOOKBACK_WINDOW"]
    )
    test_loader = DataLoader(test_dataset, batch_size=config["BATCH_SIZE"], shuffle=False)
    metrics = evaluate_on_test_set(model, test_loader, stock_mapping, config["TARGET_SCALE_FACTOR"])

    # Save results and artifacts
    save_summary("summary.txt", metrics, config, notes="Standard Train-Test Split")
    save_artifacts(model, scaler, stock_mapping, config["ARTIFACTS_DIR"])

    # Plot predictions for a few random stocks from the test set
    test_stock_ids_unique = np.unique(test_stock_ids)
    stocks_to_plot = random.sample(list(test_stock_ids_unique), min(3, len(test_stock_ids_unique)))
    for stock_id in stocks_to_plot:
        plot_price_predictions(model, scaler, stock_mapping, stock_id, config)


def run_walk_forward_validation(config, df, stock_mapping):
    """
    Performs walk-forward validation, which is more robust for time-series.
    This simulates a more realistic trading scenario.
    """
    print("\n===== Starting Walk-Forward Validation =====")
    feature_columns = df.columns.drop(['Returns', 'Stock_ID', 'Returns_Scaled'])
    
    # We will create N folds, each training on an expanding window of data
    n_splits = 5
    total_obs = len(df.index.unique()) # Number of unique days
    initial_train_days = int(total_obs * 0.5) # Start with 50% of data for training
    validation_days = int(total_obs * 0.1) # Use 10% for validation in each fold
    test_days = int(total_obs * 0.1) # Test on the next 10%

    unique_dates = df.index.unique()
    all_fold_metrics = []

    for i in range(n_splits):
        train_end_date = unique_dates[initial_train_days + i * test_days]
        test_end_date = unique_dates[initial_train_days + (i + 1) * test_days]
        
        if test_end_date >= unique_dates[-1]:
            break

        print(f"\n--- Fold {i+1}/{n_splits} ---")
        print(f"Training until {train_end_date.date()}, Testing until {test_end_date.date()}")

        # This is a simplified split; a rigorous implementation would be more careful with indices
        # For this project, we'll retrain from scratch on the whole dataset up to the split point
        # This is computationally expensive but conceptually simple.
        
        # Note: This is a placeholder for a full walk-forward implementation.
        # A full implementation would require re-running the entire preprocessing pipeline for each fold.
        # For now, we will simulate it by just printing the concept.
        
    print("\nNOTE: Full walk-forward validation is complex and requires re-running the entire")
    print("preprocessing pipeline for each fold. The code for a single train/test split will run instead.")
    print("To implement this fully, the data loading and splitting logic would need to be inside the fold loop.")
    run_single_split_training(config, df, stock_mapping)


def main():
    config = {
        "TICKERS": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"],
        "LOOKBACK_WINDOW": 30,
        "TRAIN_SPLIT_RATIO": 0.8,
        "BATCH_SIZE": 32,
        "EPOCHS": 100,  # Max epochs; early stopping will likely trigger sooner
        "LEARNING_RATE": 1e-4,
        "TARGET_SCALE_FACTOR": 10.0,
        "EARLY_STOPPING_PATIENCE": 10,
        "EARLY_STOPPING_MIN_DELTA": 1e-5,  # The minimum change in val_loss to be considered an improvement
        "VALIDATION_SPLIT_RATIO": 0.15,  # 15% of training data for validation
        "USE_WALK_FORWARD": False,  # Set to True to use walk-forward validation
        "ARTIFACTS_DIR": "deployment_artifacts"
    }

    df, stock_mapping = load_and_preprocess_data(config["TICKERS"], config["TARGET_SCALE_FACTOR"])

    if config["USE_WALK_FORWARD"]:
        run_walk_forward_validation(config, df, stock_mapping)
    else:
        run_single_split_training(config, df, stock_mapping)


if __name__ == "__main__":
    main()
