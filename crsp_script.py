# import pandas as pd
import cudf as pd
import numpy as np
from backtest import Backtest
from performance import Performance
from models.momentum_model_crsp import momentum_model

def _daily_data():
    df = pd.read_csv("crsp_daily.csv", index_col=0)

    df = df.rename(columns={'permno':'symbol', 'prc': 'close'})

    df = df.drop(columns=['shrcd','excd','siccd','vol','shr'])

    df = df.sort_values(by=['symbol','caldt'])

    df['mdt'] = pd.to_datetime(df['caldt']).dt.strftime("%Y-%m")

    return df

def _monthly_data():
    df = pd.read_csv('crsp_monthly.csv')

    df = df.rename(columns={'permno':'symbol', 'prc': 'close'})

    df = df.drop(columns=['cusip','ticker','shrcd','excd','siccd','vol','shr','cumfacshr'])

    df = df.sort_values(by=['symbol','caldt'])

    df['mdt'] = pd.to_datetime(df['caldt']).dt.strftime("%Y-%m")

    return df

def _benchmark_data():
    # Load benchmark daily data
    bmk = pd.read_csv('data/bmk.csv',index_col=False)

    bmk['Date'] = pd.to_datetime(bmk['Date'])

    bmk['caldt'] = bmk['Date'].dt.strftime("%Y-%m-%d")
    bmk['mdt'] = bmk['Date'].dt.strftime("%Y-%m")

    bmk = bmk.rename(columns={'Adj Close': 'close'})

    bmk = bmk[['caldt','mdt','close']]

    bmk['ret'] = bmk['close'].pct_change()

    bmk = bmk.drop(columns=['close', 'mdt'])

    bmk = bmk.reset_index(drop = True)

    bmk = bmk.fillna(0)

    return bmk

daily_data = _daily_data()
monthly_data = _monthly_data()
benchmark_data = _benchmark_data()

start = '2005-01-01'
end = '2023-12-31'

parameters = {
    'num_positions': 20,
    'monthly_data': monthly_data
}

backtest = Backtest(daily_data, momentum_model, parameters)

backtest_data = backtest.test(start,end)

performance = Performance(backtest_data, benchmark_data)

performance.chart()
performance.table()
    