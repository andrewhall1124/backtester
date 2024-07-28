from backtest import Backtest
# import pandas as pd
import cudf as pd
import numpy as np

historical_data = pd.read_csv('data/data.csv')

# Transformations

historical_data['timestamp'] = historical_data['timestamp'].astype(str).str[:10]
historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])

historical_data['caldt'] = historical_data['timestamp'].dt.strftime("%Y-%m-%d")
historical_data['mdt'] = historical_data['timestamp'].dt.strftime("%Y-%m")

# historical_data = historical_data[(historical_data['caldt'] >= start) & (historical_data['caldt'] <= end)]

historical_data = historical_data[['symbol','caldt','mdt','close']].copy()

historical_data = historical_data.sort_values(by=['symbol','caldt'])

historical_data = historical_data.reset_index(drop=True)

historical_data['ret'] = historical_data.groupby('symbol')['close'].pct_change()

parameters = {
    'num_positions': 10
}

def momentum_model(daily: pd.DataFrame, parameters: dict):
    num_positions = parameters['num_positions']

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

    return port

start = '2021-01-01'
end = '2023-12-31'

backtest = Backtest(historical_data, momentum_model, parameters)

result = backtest.test(start,end)

print(result)
