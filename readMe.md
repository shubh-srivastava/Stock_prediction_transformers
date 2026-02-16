Multi-Stock Transformer for Financial Time-Series Forecasting
Project Overview
This project implements a Transformer-based deep learning model for forecasting next-day returns of multiple Indian equities using daily historical market data.
The system trains a single shared Transformer model across multiple stocks, learning both:
Temporal patterns (via self-attention)
Stock-specific behavior (via stock embeddings)
The model is evaluated using regression and financial performance metrics and includes visualization of predicted vs actual stock prices for recent periods.