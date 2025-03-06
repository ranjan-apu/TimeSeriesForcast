import os
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import functions from other modules
from data_prep import fetch_data, add_indicators, preprocess_data
from model_training import fine_tune_model
from predict import predict_next_price, prepare_input_sequence, plot_prediction

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Time Series Forecasting with LLM')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict', 'data'],
                        help='Mode to run: train, predict, or data')
    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                        help='Symbol to fetch data for')
    parser.add_argument('--timeframe', type=str, default='5m',
                        help='Timeframe for the data')
    parser.add_argument('--limit', type=int, default=1000,
                        help='Number of candles to fetch')
    return parser.parse_args()

def data_collection_mode(args):
    """Collect and preprocess data"""
    print(f"Fetching {args.limit} {args.timeframe} candles for {args.symbol}...")
    df = fetch_data(symbol=args.symbol, timeframe=args.timeframe, limit=args.limit)
    df = add_indicators(df)
    df = preprocess_data(df)
    df.to_csv('data.csv', index=False)
    print(f"Data saved to data.csv. Shape: {df.shape}")
    return df

def training_mode():
    """Train the model"""
    print("Starting model fine-tuning...")
    model, tokenizer, scaler = fine_tune_model()
    print("Model fine-tuning complete!")
    print(f"Model saved to ./fine_tuned_model")
    return model, tokenizer, scaler

def prediction_mode():
    """Make predictions using the fine-tuned model"""
    # Check if model exists
    if not os.path.exists("./fine_tuned_model"):
        print("Error: Fine-tuned model not found. Please train the model first.")
        return
    
    # Load model and tokenizer
    print("Loading fine-tuned model...")
    model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model")
    tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")
    scaler = np.load("./fine_tuned_model/scaler.npy", allow_pickle=True).item()
    
    # Fetch latest data
    print("Fetching latest data...")
    df = fetch_data(limit=100)  # Get more data than needed
    df = add_indicators(df)
    df = preprocess_data(df)
    
    # Prepare input sequence
    prompt = prepare_input_sequence(df, sequence_length=10, scaler=scaler)
    
    # Predict next price
    print("Predicting next price...")
    predicted_price = predict_next_price(model, tokenizer, prompt, scaler)
    
    # Print prediction
    print(f"Predicted next price: ${predicted_price:.2f}")
    
    # Plot prediction
    plot_prediction(df, predicted_price)
    print("Prediction plot saved to prediction_plot.png")

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Run in the specified mode
    if args.mode == 'data':
        data_collection_mode(args)
    elif args.mode == 'train':
        # First collect data if it doesn't exist
        if not os.path.exists('data.csv'):
            print("Data file not found. Collecting data first...")
            data_collection_mode(args)
        # Then train the model
        training_mode()
    elif args.mode == 'predict':
        prediction_mode()

if __name__ == '__main__':
    main()
