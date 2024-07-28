# import pandas as pd
import cudf as pd
import numpy as np
from backtest import Backtest
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

historical_data = _historical_data()
parameters = {
    'num_positions': 10
}

start = '2021-01-01'
end = '2023-12-31'

backtest = Backtest(historical_data, momentum_model, parameters)

result = backtest.test(start,end)

print(result)
