import yfinance as yf
import numpy as np
import pandas as pd

# Fetch ETH-USD data using yfinance
eth_data = yf.download('ETH-USD', start='2023-01-01', end='2023-12-31')

# Calculate MACD
def calculate_macd(data, short_window=12, long_window=26):
    # Calculate Short and Long EMAs
    short_ema = data['Close'].ewm(span=short_window, min_periods=1, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, min_periods=1, adjust=False).mean()
    
    # Calculate MACD Line
    macd_line = short_ema - long_ema
    
    # Calculate Signal Line
    signal_line = macd_line.ewm(span=9, min_periods=1, adjust=False).mean()
    
    return macd_line, signal_line

# Calculate Moving Averages
def calculate_moving_averages(data, window=20):
    ma = data['Close'].rolling(window=window).mean()
    return ma

def calculate_z_scores(data):
    mean = data.mean()
    std_dev = data.std()
    z_scores = (data - mean) / std_dev
    return z_scores


def generate_signals(z_scores, z_score_threshold):
    signals = []
    for z_score in z_scores:
        if z_score > z_score_threshold:
            signals.append('Buy')
        elif z_score < -z_score_threshold:
            signals.append('Sell')
        else:
            signals.append('Hold')
    return signals


def visualize_signals(signals):
    for signal in signals:
        if signal == 'Buy':
            print("^")  # ASCII representation for Buy
        elif signal == 'Sell':
            print("v")  # ASCII representation for Sell
        else:
            print("-")  # ASCII representation for Hold


# Optimize Z-Score Thresholds
def optimize_thresholds(data):
    # Loop through different z-score thresholds
    # Evaluate performance of each threshold
    # Return the optimum threshold
    pass

# Main Execution
if __name__ == "__main__":
    # Calculate MACD and Moving Averages
    macd, signal = calculate_macd(eth_data)
    moving_avg = calculate_moving_averages(eth_data)
    
    # Calculate Z-Scores
    z_scores = calculate_z_scores(eth_data)
    
    # Generate Signals
    z_score_threshold = optimize_thresholds(z_scores)
    signals = generate_signals(eth_data, z_score_threshold)
    
    # Visualize Signals in Terminal
    visualize_signals(signals)
