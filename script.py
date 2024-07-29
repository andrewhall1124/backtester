# import pandas as pd
import cudf as pd
import numpy as np
from backtest import Backtest
from performance import Performance
from models.momentum_model import momentum_model

def _historical_data():
    df = pd.read_csv('data/data.csv')
    # Transformations

    df['timestamp'] = df['timestamp'].astype(str).str[:10]
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['caldt'] = df['timestamp'].dt.strftime("%Y-%m-%d")
    df['mdt'] = df['timestamp'].dt.strftime("%Y-%m")

    # historical_data = historical_data[(historical_data['caldt'] >= start) & (historical_data['caldt'] <= end)]

    df = df[['symbol','caldt','mdt','close']].copy()

    df = df.sort_values(by=['symbol','caldt'])

    df = df.reset_index(drop=True)

    df['ret'] = df.groupby('symbol')['close'].pct_change()

    return df

def _benchmark_data():
    # Load benchmark daily data
    bmk = pd.read_csv('data/bmk.csv',index_col=False)

    # Transformations

    bmk['Date'] = pd.to_datetime(bmk['Date'])

    bmk['caldt'] = bmk['Date'].dt.strftime("%Y-%m-%d")
    bmk['mdt'] = bmk['Date'].dt.strftime("%Y-%m")

    bmk = bmk.rename(columns={'Adj Close': 'close'})

    bmk = bmk[['caldt','mdt','close']]

    bmk['ret'] = bmk['close'].pct_change()

    bmk = bmk.drop(columns=['close', 'mdt'])

    bmk = bmk.reset_index(drop = True)

    return bmk

historical_data = _historical_data()
benchmark_data = _benchmark_data()

start = '2021-01-01'
end = '2023-12-31'

parameters = {
    'num_positions': 20
}

backtest = Backtest(historical_data, momentum_model, parameters)

backtest_data = backtest.test(start,end)

performance = Performance(backtest_data, benchmark_data)

performance.chart()
performance.table()
    