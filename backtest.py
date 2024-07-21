# import pandas as pd
import cudf as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import os
from tabulate import tabulate

TRADING_DAYS = 250

# Data Acquisition Params
start = '2016-01-01'
end = '2023-12-31'
bmk_ticker = "SPY"

# Portfolio Composition Params
num_positions = 300
rebalance_frequency = "?"
holding_period = "?"

# Load daily stock data

daily = pd.read_csv('data.csv')

# Transformations

daily['timestamp'] = daily['timestamp'].astype(str).str[:10]
daily['timestamp'] = pd.to_datetime(daily['timestamp'])

daily['caldt'] = daily['timestamp'].dt.strftime("%Y-%m-%d")
daily['mdt'] = daily['timestamp'].dt.strftime("%Y-%m")

daily = daily[(daily['caldt'] >= start) & (daily['caldt'] <= end)]

daily = daily[['symbol','caldt','mdt','close']].copy()

daily = daily.sort_values(by=['symbol','caldt'])

daily = daily.reset_index(drop=True)

daily['ret'] = daily.groupby('symbol')['close'].pct_change()

# Aggregation to monthly data

monthly = daily.groupby(['symbol','mdt'])[['caldt','close']].agg({'caldt': 'last','close': 'last'})

monthly = monthly.reset_index()

# Generate features

monthly['ret'] = monthly.groupby('symbol')['close'].pct_change()

monthly['logret'] = np.log(1+monthly['ret'])

monthly['mom'] = monthly.groupby('symbol')['logret'].rolling(11,11).sum().reset_index(drop=True)

monthly['mom'] = monthly.groupby('symbol')['mom'].shift(1)

monthly['momlag'] = monthly.groupby('symbol')['mom'].shift(1)

# Trading filters

monthly['prclag'] = monthly.groupby('symbol')['close'].shift(1)

monthly = monthly.query('momlag == momlag and prclag >= 5')

# Portfolio generation

monthly['score'] = monthly.groupby('mdt')['momlag'].rank(ascending=False)

port = monthly[monthly['score'] <= num_positions].reset_index(drop=True).copy()

# Merge portfolio with daily data to create backtest

port = port[['symbol','mdt','score']]

test = pd.merge(left=daily,right=port,on=['symbol','mdt'], how='inner')

# Backtest Transformations

test = test.groupby('caldt')['ret'].mean().to_frame().reset_index()

test = test.sort_values(by='caldt').reset_index(drop=True)

test['cumret'] = (1+test['ret']).cumprod() - 1

# Load benchmark daily data

bmk = yf.download(bmk_ticker, start=start, end=end)

bmk.to_csv('bmk.csv')

bmk = pd.read_csv('bmk.csv',index_col=False)

# Transformations

bmk['Date'] = pd.to_datetime(bmk['Date'])

bmk['caldt'] = bmk['Date'].dt.strftime("%Y-%m-%d")
bmk['mdt'] = bmk['Date'].dt.strftime("%Y-%m")

bmk = bmk.rename(columns={'Adj Close': 'close'})

bmk = bmk[['caldt','mdt','close']]

bmk = bmk[bmk['caldt'] >= test['caldt'].min()]

bmk['ret'] = bmk['close'].pct_change()
bmk['cumret'] = (1+bmk['ret']).cumprod() - 1

bmk = bmk.drop(columns=['close', 'mdt'])

bmk = bmk.reset_index(drop = True)

# Chart

test['caldt'] = pd.to_datetime(test['caldt'])
bmk['caldt'] = pd.to_datetime(bmk['caldt'])

test = test.to_pandas()
bmk = bmk.to_pandas()

sns.lineplot(data=test,x='caldt',y='cumret', label='Port')
sns.lineplot(data=bmk,x='caldt',y='cumret', label='Benchmark')

plt.savefig('chart.png')

# Metrics

port_cumret = test['cumret'].iloc[-1] 
bmk_cumret = bmk['cumret'].iloc[-1]

port_ret = port_cumret / test['cumret'].count() * TRADING_DAYS
bmk_ret = bmk_cumret / bmk['cumret'].count() * TRADING_DAYS

port_vol = test['ret'].std()
bmk_vol = bmk['ret'].std()

port_er = test['ret'].mean()
bmk_er = bmk['ret'].mean()

port_sharpe = port_er / port_vol * (TRADING_DAYS/np.sqrt(TRADING_DAYS))
bmk_sharpe = bmk_er / bmk_vol * (TRADING_DAYS/np.sqrt(TRADING_DAYS))

correlation = test['ret'].corr(bmk['ret'])
covariance = test['ret'].cov(bmk['ret'])

beta = covariance / (bmk_vol**2)

# Create a table with the results

table = [
    ["Metric", "Portfolio", "Benchmark"],
    ["Total Return", f"% {round(port_cumret * 100, 2)}", f"% {round(bmk_cumret * 100, 2)}" ],
    ["Annual Return", f"% {round(port_ret * 100, 2)}", f"% {round(bmk_ret * 100, 2)}"],
    ["Expected Return", f"% {round(port_er * 100,2)}", f"% {round(bmk_er * 100,2)}"],
    ["Volatility", f"% {round(port_vol * 100,2)}", f"% {round(bmk_vol * 100,2)}"],
    ["Sharpe", f"{round((port_sharpe),2)}", f"{round((bmk_sharpe),2)}"],
    ["Correlation", round(correlation,2)],
    ["Beta", round(beta,2)]
]

# Print the table

print(tabulate(table, headers="firstrow", tablefmt="grid"))