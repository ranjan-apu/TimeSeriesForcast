import pandas as pd
import numpy as np
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_prep import fetch_data, add_indicators, preprocess_data
from datetime import datetime

def load_model_and_tokenizer():
    """Load the fine-tuned model and tokenizer"""
    model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model")
    tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")
    # Load the scaler
    scaler = np.load("./fine_tuned_model/scaler.npy", allow_pickle=True).item()
    return model, tokenizer, scaler

def prepare_input_sequence(df, sequence_length=10, scaler=None):
    """Prepare the input sequence for prediction"""
    # Get the last sequence_length rows
    sequence = df.tail(sequence_length)
    
    # Extract features
    features = ['open', 'high', 'low', 'close', 'volume', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'EMA_20']
    
    # Normalize if scaler is provided
    if scaler:
        sequence[features] = scaler.transform(sequence[features])
    
    # Format the sequence as text
    sequence_text = ""
    for i, (_, row) in enumerate(sequence.iterrows()):
        sequence_text += f"Time {i+1}: Open={row['open']:.2f}, High={row['high']:.2f}, Low={row['low']:.2f}, Close={row['close']:.2f}, Volume={row['volume']:.2f}, RSI={row['RSI_14']:.2f}\n"
    
    # Create prompt
    prompt = f"Given the following price data for a cryptocurrency:\n{sequence_text}\nPredict the next price:"
    
    return prompt

def predict_next_price(model, tokenizer, prompt, scaler=None):
    """Predict the next price using the fine-tuned model"""
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate prediction
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=20,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.95,
        )
    
    # Decode the prediction
    prediction_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Raw model output: {prediction_text}")
    
    # Extract the predicted price from the text using regex
    # Look for numbers after the prompt
    prompt_end = "Predict the next price:"
    if prompt_end in prediction_text:
        prediction_part = prediction_text.split(prompt_end)[1].strip()
        # Try to find a number in the prediction part
        price_match = re.search(r'\d+\.?\d*', prediction_part)
        if price_match:
            predicted_price = float(price_match.group())
        else:
            # Fallback: try to get any number from the output
            price_match = re.search(r'\d+\.?\d*', prediction_text)
            if price_match:
                predicted_price = float(price_match.group())
            else:
                print("Warning: Could not extract a price from the model output. Using default value.")
                predicted_price = 0.0  # Default value if extraction fails
    else:
        # If we can't find the prompt end, try to get any number from the last line
        last_line = prediction_text.split("\n")[-1].strip()
        price_match = re.search(r'\d+\.?\d*', last_line)
        if price_match:
            predicted_price = float(price_match.group())
        else:
            print("Warning: Could not extract a price from the model output. Using default value.")
            predicted_price = 0.0  # Default value if extraction fails
    
    print(f"Extracted predicted price (before scaling): {predicted_price}")
    
    # Inverse transform if scaler is provided
    if scaler:
        # Create a dummy array with the same structure as the original data
        dummy = np.zeros((1, len(scaler.feature_names_in_)))
        # Find the index of 'close' in the feature names
        close_idx = list(scaler.feature_names_in_).index('close')
        # Set the predicted price
        dummy[0, close_idx] = predicted_price
        # Inverse transform
        dummy = scaler.inverse_transform(dummy)
        # Get the actual price
        predicted_price = dummy[0, close_idx]
    
    return predicted_price

def evaluate_predictions(actual_prices, predicted_prices):
    """Calculate evaluation metrics for the predictions"""
    mse = mean_squared_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_prices, predicted_prices)
    r2 = r2_score(actual_prices, predicted_prices)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
    
    # Calculate directional accuracy
    actual_direction = np.sign(np.diff(np.append([actual_prices[0]], actual_prices)))
    predicted_direction = np.sign(np.diff(np.append([actual_prices[0]], predicted_prices)))
    directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'Directional_Accuracy': directional_accuracy
    }

def plot_prediction(historical_data, predicted_price, save_path='prediction_plot.png'):
    """Plot the historical prices and the predicted price"""
    plt.figure(figsize=(12, 6))
    
    # Plot historical prices
    plt.plot(historical_data.index, historical_data['close'], label='Historical Close Prices')
    
    # Plot predicted price
    last_timestamp = historical_data.index[-1]
    next_timestamp = last_timestamp + pd.Timedelta(minutes=5)  # Assuming 5-minute intervals
    plt.scatter(next_timestamp, predicted_price, color='red', s=100, label='Predicted Price')
    
    # Add the last actual price and the predicted price as text
    last_price = historical_data['close'].iloc[-1]
    plt.annotate(f'Last Price: ${last_price:.2f}', 
                 xy=(last_timestamp, last_price),
                 xytext=(last_timestamp - pd.Timedelta(minutes=30), last_price * 1.02),
                 arrowprops=dict(arrowstyle='->', color='blue'))
    
    plt.annotate(f'Predicted: ${predicted_price:.2f}', 
                 xy=(next_timestamp, predicted_price),
                 xytext=(next_timestamp - pd.Timedelta(minutes=20), predicted_price * 1.02),
                 arrowprops=dict(arrowstyle='->', color='red'))
    
    # Calculate percentage change
    pct_change = ((predicted_price - last_price) / last_price) * 100
    direction = 'up' if pct_change > 0 else 'down'
    plt.title(f'Cryptocurrency Price Prediction: {direction.upper()} {abs(pct_change):.2f}%')
    
    plt.xlabel('Time')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return save_path

def plot_multiple_predictions(historical_data, predictions, test_data=None, window_size=10, save_path='multiple_predictions.png'):
    """Plot multiple predictions against historical data"""
    plt.figure(figsize=(14, 8))
    
    # Plot historical prices
    plt.plot(historical_data.index, historical_data['close'], label='Historical Prices', color='blue')
    
    # Plot test data if provided
    if test_data is not None:
        plt.plot(test_data.index, test_data['close'], label='Actual Future Prices', color='green')
    
    # Plot predictions
    prediction_times = [historical_data.index[-1] + pd.Timedelta(minutes=5*i) for i in range(1, len(predictions)+1)]
    plt.plot(prediction_times, predictions, label='Predicted Prices', color='red', linestyle='--', marker='o')
    
    plt.title('Multiple-Step Price Predictions')
    plt.xlabel('Time')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return save_path

def plot_error_analysis(actual_prices, predicted_prices, save_path='error_analysis.png'):
    """Plot error analysis for predictions"""
    plt.figure(figsize=(16, 12))
    
    # Calculate errors
    errors = actual_prices - predicted_prices
    abs_errors = np.abs(errors)
    pct_errors = (abs_errors / actual_prices) * 100
    
    # Plot 1: Actual vs Predicted
    plt.subplot(2, 2, 1)
    plt.scatter(actual_prices, predicted_prices, alpha=0.6)
    plt.plot([min(actual_prices), max(actual_prices)], [min(actual_prices), max(actual_prices)], 'r--')
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.grid(True)
    
    # Plot 2: Error Distribution
    plt.subplot(2, 2, 2)
    sns.histplot(errors, kde=True)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    
    # Plot 3: Absolute Error vs Actual Price
    plt.subplot(2, 2, 3)
    plt.scatter(actual_prices, abs_errors, alpha=0.6)
    plt.title('Absolute Error vs Actual Price')
    plt.xlabel('Actual Price')
    plt.ylabel('Absolute Error')
    plt.grid(True)
    
    # Plot 4: Percentage Error Distribution
    plt.subplot(2, 2, 4)
    sns.histplot(pct_errors, kde=True)
    plt.title('Percentage Error Distribution')
    plt.xlabel('Percentage Error (%)')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return save_path

def make_multiple_predictions(model, tokenizer, initial_data, num_predictions=5, scaler=None):
    """Make multiple sequential predictions"""
    predictions = []
    current_data = initial_data.copy()
    
    for i in range(num_predictions):
        # Prepare input sequence
        prompt = prepare_input_sequence(current_data, sequence_length=10, scaler=scaler)
        
        # Predict next price
        predicted_price = predict_next_price(model, tokenizer, prompt, scaler)
        predictions.append(predicted_price)
        
        # Add the prediction to the data for the next prediction
        last_row = current_data.iloc[-1].copy()
        next_timestamp = current_data.index[-1] + pd.Timedelta(minutes=5)  # Assuming 5-minute intervals
        last_row['close'] = predicted_price
        # Set other price columns to the same value for simplicity
        last_row['open'] = predicted_price
        last_row['high'] = predicted_price
        last_row['low'] = predicted_price
        
        # Create a new row with the prediction
        new_row = pd.DataFrame([last_row], index=[next_timestamp])
        current_data = pd.concat([current_data, new_row])
        
        # Recalculate indicators if needed
        current_data = add_indicators(current_data)
    
    return predictions

def evaluate_model_on_test_data(model, tokenizer, test_data, sequence_length=10, scaler=None):
    """Evaluate the model on test data"""
    actual_prices = []
    predicted_prices = []
    
    # We need at least sequence_length + 1 data points to make a prediction and evaluate it
    for i in range(len(test_data) - sequence_length):
        # Get the sequence and the next actual price
        sequence = test_data.iloc[i:i+sequence_length]
        next_actual = test_data.iloc[i+sequence_length]['close']
        actual_prices.append(next_actual)
        
        # Prepare input sequence
        prompt = prepare_input_sequence(sequence, sequence_length=sequence_length, scaler=scaler)
        
        # Predict next price
        predicted_price = predict_next_price(model, tokenizer, prompt, scaler)
        predicted_prices.append(predicted_price)
    
    # Calculate evaluation metrics
    metrics = evaluate_predictions(np.array(actual_prices), np.array(predicted_prices))
    
    # Plot error analysis
    plot_error_analysis(np.array(actual_prices), np.array(predicted_prices))
    
    return metrics, actual_prices, predicted_prices

def main():
    # Fetch latest data
    print("Fetching and preprocessing data...")
    df = fetch_data(limit=200)  # Get more data than needed to ensure we have enough after preprocessing
    df = add_indicators(df)
    df = preprocess_data(df)
    
    # Split data into training and testing sets
    train_size = int(len(df) * 0.8)
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]
    
    print(f"Data split: {len(train_data)} training samples, {len(test_data)} testing samples")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer, scaler = load_model_and_tokenizer()
    
    # Single prediction
    print("\n1. Making a single prediction...")
    prompt = prepare_input_sequence(df.tail(10), sequence_length=10, scaler=scaler)
    predicted_price = predict_next_price(model, tokenizer, prompt, scaler)
    print(f"Predicted next price: ${predicted_price:.2f}")
    
    # Plot single prediction
    plot_path = plot_prediction(df.tail(20), predicted_price)
    print(f"Single prediction plot saved to {plot_path}")
    
    # Multiple sequential predictions
    print("\n2. Making multiple sequential predictions...")
    num_predictions = 5
    multiple_predictions = make_multiple_predictions(model, tokenizer, df.tail(10), num_predictions, scaler)
    print(f"Multiple predictions: {[f'${price:.2f}' for price in multiple_predictions]}")
    
    # Plot multiple predictions
    multi_plot_path = plot_multiple_predictions(df.tail(20), multiple_predictions)
    print(f"Multiple predictions plot saved to {multi_plot_path}")
    
    # Evaluate on test data
    print("\n3. Evaluating model on test data...")
    metrics, actual_prices, predicted_prices = evaluate_model_on_test_data(model, tokenizer, test_data, sequence_length=10, scaler=scaler)
    
    # Print evaluation metrics
    print("\nModel Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot actual vs predicted for test data
    test_plot_path = plot_multiple_predictions(
        test_data.iloc[:10],  # First 10 points as history
        predicted_prices,
        test_data.iloc[10:10+len(predicted_prices)],  # Actual future prices
        save_path='test_evaluation.png'
    )
    print(f"Test evaluation plot saved to {test_plot_path}")
    
    print("\nAnalysis complete! Check the generated plots for visualizations.")
    
    # Create a timestamp for this prediction run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Prediction timestamp: {timestamp}")
    
    return predicted_price, multiple_predictions, metrics

if __name__ == "__main__":
    main()
