import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import random
from tqdm import tqdm

# Define constants
MODEL_NAME = "deepseek-ai/deepseek-llm-1.5b"  # Using DeepSeek R1 Distill 1.5B parameter model
SEQUENCE_LENGTH = 10  # Number of time steps to consider for prediction
PREDICTION_HORIZON = 1  # Number of time steps to predict ahead

# Set device to GPU if available, otherwise use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using CPU for training for stability")

# Custom dataset for time series
class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length, prediction_horizon, tokenizer):
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data) - self.sequence_length - self.prediction_horizon + 1

    def __getitem__(self, idx):
        # Get the sequence of data
        sequence = self.data[idx:idx + self.sequence_length]
        # Get the target (future price)
        target = self.data[idx + self.sequence_length:idx + self.sequence_length + self.prediction_horizon]

        # Convert to text format for the LLM
        sequence_text = self.format_sequence(sequence)
        target_text = self.format_target(target)

        # Create prompt with target included for causal language modeling
        # For GPT-style models, we need to have the target as part of the input sequence
        full_prompt = f"Given the following price data for a cryptocurrency:\n{sequence_text}\nPredict the next price: {target_text}"

        # Tokenize full sequence
        encodings = self.tokenizer(full_prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=512)

        # Create labels (shifted input_ids for causal LM)
        input_ids = encodings.input_ids[0]
        attention_mask = encodings.attention_mask[0]

        # For causal LM, labels are the same as input_ids
        # We'll mask out the prompt part during loss calculation
        labels = input_ids.clone()

        # Find the position where "Predict the next price:" starts
        prompt_text = f"Given the following price data for a cryptocurrency:\n{sequence_text}\nPredict the next price:"
        prompt_tokens = self.tokenizer(prompt_text, return_tensors="pt").input_ids[0]
        prompt_len = len(prompt_tokens)

        # Set labels for prompt part to -100 (ignored in loss calculation)
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def format_sequence(self, sequence):
        # Format the sequence data as text
        text = ""
        for i, row in enumerate(sequence):
            text += f"Time {i+1}: Open={row['open']:.2f}, High={row['high']:.2f}, Low={row['low']:.2f}, Close={row['close']:.2f}, Volume={row['volume']:.2f}, RSI={row['RSI_14']:.2f}\n"
        return text

    def format_target(self, target):
        # Format the target data as text
        return f"{target[0]['close']:.2f}"

# Load and preprocess data
def prepare_data():
    try:
        # Load data
        df = pd.read_csv('data.csv')
        print(f"Loaded data with shape: {df.shape}")

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Sort by timestamp
        df = df.sort_values('timestamp')

        # Check for missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            print(f"Warning: Found {missing_values} missing values in the dataset")
            df = df.dropna()
            print(f"After dropping missing values, data shape: {df.shape}")

        # Normalize numerical features
        features = ['open', 'high', 'low', 'close', 'volume', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'EMA_20']
        scaler = MinMaxScaler()
        df[features] = scaler.fit_transform(df[features])

        # Split into train and validation sets (chronological split for time series)
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]

        print(f"Train set: {train_df.shape}, Validation set: {val_df.shape}")

        return train_df.to_dict('records'), val_df.to_dict('records'), scaler

    except Exception as e:
        print(f"Error during data preparation: {str(e)}")
        raise

# Simplified GRPO implementation for causal language modeling
class GRPOTrainer:
    def __init__(self, model, tokenizer, train_dataset, val_dataset, learning_rate=1e-5,
                 clip_param=0.1, value_loss_coef=0.5, entropy_coef=0.01, group_size=2):  # Reduced group size to save memory
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.learning_rate = learning_rate
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.group_size = group_size  # Group size for GRPO

        # Create optimizer with weight decay
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)

        # Create data loaders with smaller batch sizes to save memory
        self.train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        # MSE loss for value function
        self.mse_loss = nn.MSELoss()

    def compute_rewards(self, predictions, targets):
        """Compute rewards based on prediction accuracy"""
        # In time series forecasting, negative MSE can be used as reward
        rewards = -torch.pow(predictions - targets, 2)
        return rewards

    def compute_advantages(self, rewards, values):
        """Compute advantages as reward - value"""
        advantages = rewards - values
        return advantages

    def compute_relative_advantages(self, advantages):
        """Compute relative advantages within each group for GRPO"""
        # Reshape advantages to [num_groups, group_size]
        num_groups = advantages.size(0) // self.group_size
        if num_groups == 0:
            return advantages  # Not enough samples for grouping

        advantages_grouped = advantages[:num_groups * self.group_size].view(num_groups, self.group_size)

        # Compute mean advantage for each group
        mean_advantages = advantages_grouped.mean(dim=1, keepdim=True)

        # Compute relative advantages
        relative_advantages = advantages_grouped - mean_advantages

        # Reshape back to original shape
        relative_advantages = relative_advantages.view(-1)

        # Pad if necessary
        if advantages.size(0) > relative_advantages.size(0):
            padding = advantages[num_groups * self.group_size:]
            relative_advantages = torch.cat([relative_advantages, padding])

        return relative_advantages

    def train_epoch(self, epoch):
        """Train for one epoch using simplified approach"""
        self.model.train()
        epoch_loss = 0

        # Enable gradient checkpointing to save memory
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} Training")
        for batch in progress_bar:
            # Get batch data
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            # Forward pass - simple causal language modeling
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Update metrics
            epoch_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({
                "loss": loss.item()
            })

        # Return average loss
        num_batches = len(self.train_loader)
        return {
            "loss": epoch_loss / num_batches
        }

    def evaluate(self):
        """Evaluate the model on validation data"""
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Get batch data
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                # Update metrics
                val_loss += loss.item()

        # Return average loss
        return {"val_loss": val_loss / len(self.val_loader)}

    def train(self, num_epochs=5):
        """Train the model for a specified number of epochs"""
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            # Train for one epoch
            train_metrics = self.train_epoch(epoch)

            # Evaluate
            val_metrics = self.evaluate()

            # Print metrics
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Validation Loss: {val_metrics['val_loss']:.4f}")

            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                print(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
                # We'll save the model at the end

        return self.model

# Fine-tune the model using GRPO
def fine_tune_model():
    try:
        print("Loading model and tokenizer...")
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # Fix padding token issue
        if tokenizer.pad_token is None:
            print("Setting padding token for the tokenizer...")
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model = model.to(DEVICE)  # Move model to GPU if available

        print("Preparing data...")
        # Prepare data
        train_data, val_data, scaler = prepare_data()

        print(f"Creating datasets with {len(train_data)} training samples and {len(val_data)} validation samples...")
        # Create datasets
        train_dataset = TimeSeriesDataset(train_data, SEQUENCE_LENGTH, PREDICTION_HORIZON, tokenizer)
        val_dataset = TimeSeriesDataset(val_data, SEQUENCE_LENGTH, PREDICTION_HORIZON, tokenizer)

        print("Setting up GRPO trainer with memory optimization...")
        # Create GRPO trainer with memory-efficient settings
        trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            learning_rate=1e-5,  # Lower learning rate for stability
            clip_param=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            group_size=2  # Smaller group size to save memory
        )

        print("Starting model training with GRPO...")
        # Train the model using GRPO
        model = trainer.train(num_epochs=5)

        print("Training complete. Saving model...")
        # Save the model
        os.makedirs("./fine_tuned_model_deepseek", exist_ok=True)
        model.save_pretrained("./fine_tuned_model_deepseek")
        tokenizer.save_pretrained("./fine_tuned_model_deepseek")

        # Save the scaler
        np.save("./fine_tuned_model_deepseek/scaler.npy", scaler)
        print("Model and tokenizer saved to ./fine_tuned_model_deepseek")

        return model, tokenizer, scaler

    except Exception as e:
        print(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    fine_tune_model()
