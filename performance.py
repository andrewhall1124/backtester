# import pandas as pd
import cudf as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

TRADING_DAYS = 250

class Performance:
    
    def __init__(self, backtest_data: pd.DataFrame, benchmark_data: pd.DataFrame):
        self.test = backtest_data
        self.bmk = benchmark_data

        start = self.test['caldt'].min()
        end = self.test['caldt'].max()

        self.bmk = self.bmk[(self.bmk['caldt'] >= start) & (self.bmk['caldt'] <= end)].reset_index(drop=True)

        self.test['cumret'] = (1+self.test['ret']).cumprod() - 1
        self.bmk['cumret'] = (1+self.bmk['ret']).cumprod() - 1
    
    def chart(self):
        test = self.test
        bmk = self.bmk

        # Chart
        test['caldt'] = pd.to_datetime(test['caldt'])
        bmk['caldt'] = pd.to_datetime(bmk['caldt'])

        test = test.to_pandas()
        bmk = bmk.to_pandas()

        sns.lineplot(data=test,x='caldt',y='cumret', label='Port')
        sns.lineplot(data=bmk,x='caldt',y='cumret', label='Benchmark')

        plt.savefig('chart.png')
    
    def table(self):
        test = self.test
        bmk = self.bmk

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

    def portfolio_metrics(self):
        test = self.test
        bmk = self.bmk

        port_cumret = test['cumret'].iloc[-1] 

        port_ret = port_cumret / test['cumret'].count() * TRADING_DAYS

        port_vol = test['ret'].std()
        bmk_vol = bmk['ret'].std()

        port_er = test['ret'].mean()

        port_sharpe = port_er / port_vol * (TRADING_DAYS/np.sqrt(TRADING_DAYS))

        correlation = test['ret'].corr(bmk['ret'])
        covariance = test['ret'].cov(bmk['ret'])

        beta = covariance / (bmk_vol**2)

        result = {
            'Total Return': port_cumret,
            'Annual Return': port_ret,
            'Volatility': port_vol,
            'Expected Return': port_er,
            'Sharpe': port_sharpe,
            'Correlation': correlation,
            'Beta': beta
        }

        return result

