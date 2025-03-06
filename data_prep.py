import ccxt
import pandas as pd
import pandas_ta as ta

# Fetch historical data
def fetch_data(symbol='BTC/USDT', timeframe='5m', limit=1000):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Add technical indicators
def add_indicators(df):
    df.ta.rsi(append=True)
    df.ta.macd(append=True)
    df.ta.ema(length=20, append=True)
    return df

# Preprocess data for training
def preprocess_data(df):
    df = df.dropna()  # Drop rows with NaN values
    return df

# Main function
if __name__ == '__main__':
    df = fetch_data()
    df = add_indicators(df)
    df = preprocess_data(df)
    df.to_csv('data.csv', index=False)
    print('Data saved to data.csv')
