# TimeSeriesForcast

A Python project for time series forecasting using a fine-tuned DeepSeek LLM to predict cryptocurrency prices in a 5-minute timeframe.

## Project Overview

This project aims to fine-tune a small language model (DeepSeek R1 Distrill 3B parameters) to predict cryptocurrency prices. The model takes historical price data (OHLCV: Open, High, Low, Close, Volume) and technical indicators as input and predicts the price for the next 5-minute candle.

## Features

- **Data Collection**: Fetch historical cryptocurrency price data from exchanges using the CCXT library
- **Technical Analysis**: Calculate technical indicators (RSI, MACD, EMA) using pandas_ta
- **Data Preprocessing**: Normalize and prepare data for model training
- **Model Fine-tuning**: Fine-tune a DeepSeek LLM for time series forecasting
- **Price Prediction**: Generate predictions for future price movements
- **Visualization**: Plot historical and predicted prices

## Project Structure

- `main.py`: Main script to run the project in different modes
- `data_prep.py`: Functions for data collection and preprocessing
- `model_training.py`: Model fine-tuning pipeline
- `predict.py`: Functions for making predictions with the fine-tuned model
- `requirements.txt`: Project dependencies

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the project in one of the following modes:

## Usage

### Data Collection

To collect and preprocess cryptocurrency price data:

```bash
python main.py --mode data --symbol BTC/USDT --timeframe 5m --limit 1000
```

This will fetch 1000 5-minute candles for BTC/USDT, calculate technical indicators, and save the data to `data.csv`.

### Model Training

To fine-tune the model on the collected data:

```bash
python main.py --mode train
```

This will load the data from `data.csv`, prepare it for training, and fine-tune the DeepSeek model. The fine-tuned model will be saved to the `fine_tuned_model` directory.

### Price Prediction

To make predictions using the fine-tuned model:

```bash
python main.py --mode predict
```

This will load the fine-tuned model, fetch the latest data, and predict the price for the next 5-minute candle. The prediction will be displayed in the console and a plot will be saved to `prediction_plot.png`.

## Model Details

The project uses a DeepSeek R1 Distrill model with 3B parameters. The model is fine-tuned to predict cryptocurrency prices based on historical price data and technical indicators. The input to the model is a sequence of 10 5-minute candles, and the output is the predicted price for the next candle.

## Limitations

- The model's predictions are based on historical patterns and may not account for unexpected market events
- Cryptocurrency markets are highly volatile and unpredictable
- The model's performance depends on the quality and quantity of the training data

## Future Improvements

- Add more technical indicators and features
- Experiment with different model architectures and hyperparameters
- Implement backtesting to evaluate the model's performance
- Add support for multiple cryptocurrencies and timeframes
- Implement real-time prediction and trading
