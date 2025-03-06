import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
from data_prep import fetch_data, add_indicators, preprocess_data

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
    
    # Extract the predicted price from the text
    # This assumes the model outputs just the price value
    predicted_price = float(prediction_text.split("\n")[-1].strip())
    
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

def plot_prediction(historical_data, predicted_price):
    """Plot the historical prices and the predicted price"""
    plt.figure(figsize=(12, 6))
    
    # Plot historical prices
    plt.plot(historical_data.index, historical_data['close'], label='Historical Close Prices')
    
    # Plot predicted price
    last_timestamp = historical_data.index[-1]
    next_timestamp = last_timestamp + pd.Timedelta(minutes=5)  # Assuming 5-minute intervals
    plt.scatter(next_timestamp, predicted_price, color='red', s=100, label='Predicted Price')
    
    plt.title('Cryptocurrency Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig('prediction_plot.png')
    plt.close()

def main():
    # Fetch latest data
    df = fetch_data(limit=100)  # Get more data than needed to ensure we have enough after preprocessing
    df = add_indicators(df)
    df = preprocess_data(df)
    
    # Load model and tokenizer
    model, tokenizer, scaler = load_model_and_tokenizer()
    
    # Prepare input sequence
    prompt = prepare_input_sequence(df, sequence_length=10, scaler=scaler)
    
    # Predict next price
    predicted_price = predict_next_price(model, tokenizer, prompt, scaler)
    
    # Print prediction
    print(f"Predicted next price: ${predicted_price:.2f}")
    
    # Plot prediction
    plot_prediction(df, predicted_price)
    
    return predicted_price

if __name__ == "__main__":
    main()
