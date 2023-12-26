import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Data Collection for Ethereum and S&P 500
eth = yf.download('USDT-USD', start='2015-08-07', end='2023-12-25', interval='1d')
sp500 = yf.download('SPY', start='2015-08-07', end='2023-12-25', interval='1d')

# Step 2: Feature Engineering for ETH
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

eth['RSI'] = calculate_rsi(eth)

eth['EMA_12'] = eth['Close'].ewm(span=12, min_periods=0, adjust=False).mean()
eth['EMA_26'] = eth['Close'].ewm(span=26, min_periods=0, adjust=False).mean()
eth['MACD'] = eth['EMA_12'] - eth['EMA_26']

# Calculate 'Long_Position' based on RSI and MACD for Ethereum
eth['Long_Position'] = np.where((eth['RSI'] < 30) & (eth['MACD'] > 0), 1, 0)
eth['Short_Position'] = np.where((eth['RSI'] > 70) & (eth['MACD'] < 0), 1, 0)

# Aligning data for ETH and S&P 500 returns
eth_returns = eth['Close'].pct_change().dropna()
market_returns = sp500['Close'].pct_change().dropna()

# Align the indices of eth_returns and market_returns
common_index = eth_returns.index.intersection(market_returns.index)
eth_returns_aligned = eth_returns.loc[common_index]
market_returns_aligned = market_returns.loc[common_index]

# Calculate beta
covariance = np.cov(eth_returns_aligned, market_returns_aligned)
eth_variance = np.var(eth_returns_aligned)
beta = covariance[0][1] / eth_variance

# Risk-free rate
risk_free_rate = 0.02  # Replace with the risk-free rate

# Calculate Sharpe Ratio
sharpe_ratio = (eth_returns_aligned.mean() - risk_free_rate) / eth_returns_aligned.std()

# Calculate Max Drawdown
cumulative_returns = eth['Close'].pct_change().dropna().add(1).cumprod()
max_drawdown = (1 - cumulative_returns.div(cumulative_returns.cummax())).max()

# Print performance metrics for Ethereum
print(f"Beta (ETH): {beta}")
print(f"Sharpe Ratio (ETH): {sharpe_ratio}")
print(f"Max Drawdown (ETH): {max_drawdown}")

# Plotting ETH Price with Long/Short Positions (Log Scale)
plt.figure(figsize=(12, 6))

# Plotting Price in Log Scale
plt.plot(eth.index, eth['Close'], label='Price', color='blue')
plt.yscale('log')  # Set y-axis to log scale

# Plotting Long and Short Positions
long_pos_indices = eth[eth['Long_Position'] == 1].index
short_pos_indices = eth[eth['Short_Position'] == 1].index

# Plotting Long and Short Positions
plt.scatter(long_pos_indices, eth.loc[long_pos_indices]['Close'], color='green', marker='^', label='Long', alpha=0.7)
plt.scatter(short_pos_indices, eth.loc[short_pos_indices]['Close'], color='red', marker='v', label='Short', alpha=0.7)

plt.title('ETH Price with Long/Short Positions (Log Scale)')
plt.xlabel('Date')
plt.ylabel('Price (Log Scale)')
plt.legend()
plt.grid()
plt.show()
