# import pandas as pd
import cudf as pd
import numpy as np

def momentum_model(daily: pd.DataFrame, parameters: dict):
    monthly = parameters['monthly_data']
    num_positions = parameters['num_positions']

    # Generate features

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