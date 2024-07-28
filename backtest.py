# import pandas as pd
import cudf as pd

class Backtest:

    def __init__(self, historical_data: pd.DataFrame, model, model_params):

        self.historical_data = historical_data
        self.model = model
        self.model_params = model_params

    def test(self, start_date: str, end_date: str) -> pd.DataFrame:
        daily = self.historical_data
        params = self.model_params


        # Filter timeframe

        daily = daily[(daily['caldt'] >= start_date) & (daily['caldt'] <= end_date)]

        # Compute model portfolios

        port = self.model(daily,params)

        # Merge portfolio with daily data to create backtest

        port = port[['symbol','mdt','score']]

        test = pd.merge(left=daily,right=port,on=['symbol','mdt'], how='inner')

        # Backtest Transformations

        test = test.groupby('caldt')['ret'].mean().to_frame().reset_index()

        test = test.sort_values(by='caldt').reset_index(drop=True)

        return test