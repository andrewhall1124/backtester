import cudf as pd
# import pandas as pd
# import cupy as np
import numpy as np    


def fip_model(daily, parameters):
    num_positions = parameters['num_positions']

    # Create additional columns
    daily['up'] = daily['ret'].apply(lambda x: 1 if x > 0 else 0)
    daily['down'] = daily['ret'].apply(lambda x: 1 if x < 0 else 0)
    daily['total'] = 1


    # Aggregate to monthly level
    monthly = daily.groupby(['symbol','mdt'])[['caldt','close','up','down', 'total']].agg({'caldt': 'last','close': 'last','up':'sum','down':'sum', 'total':'sum'})

    monthly.reset_index("mdt",inplace=True)
    monthly.reset_index("symbol",inplace=True)

    # Create additional monthly columns
    monthly['%neg-%pos'] = (monthly['down']-daily['up']) / monthly['total']

    monthly['ret'] = monthly.groupby('symbol')['close'].pct_change()

    monthly['logret'] = np.log(1 + daily['ret'])


    # Generate pret and id features
    monthly['pret'] = monthly.groupby('symbol')['logret'].rolling(11,11).sum().reset_index(drop=True)
    monthly['pret'] = monthly.groupby('symbol')['pret'].shift(2)

    monthly['id'] = monthly.groupby('symbol')['%neg-%pos'].rolling(11,11).mean().reset_index(drop=True)
    monthly['id'] = monthly.groupby('symbol')['id'].shift(2)

    monthly['id'] = np.sign(monthly['pret']) * monthly['id']

    monthly['momscore'] = monthly['pret'] * abs(monthly['id'])


    # Filter universe by price and availability
    monthly['prclag'] = monthly.groupby('symbol')['close'].shift(1)

    monthly = monthly.query("pret == pret and id == id and prclag >= 5").reset_index(drop=True)


    # Generate portfolio
    monthly['score'] = monthly.groupby('mdt')['momscore'].rank(ascending=False)

    port = monthly[monthly['score'] <= num_positions].reset_index(drop=True).copy()

    return port