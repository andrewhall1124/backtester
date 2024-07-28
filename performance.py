import pandas as pd
import numpy as np

class Performance:
    def __init__(self, returns: pd.Series):
        self.returns = returns

        # Transformations

        bmk['Date'] = pd.to_datetime(bmk['Date'])

        bmk['caldt'] = bmk['Date'].dt.strftime("%Y-%m-%d")
        bmk['mdt'] = bmk['Date'].dt.strftime("%Y-%m")

        bmk = bmk.rename(columns={'Adj Close': 'close'})

        bmk = bmk[['caldt','mdt','close']]

        bmk = bmk[bmk['caldt'] >= returns['caldt'].min()]

        bmk['ret'] = bmk['close'].pct_change()
        bmk['cumret'] = (1+bmk['ret']).cumprod() - 1

        bmk = bmk.drop(columns=['close', 'mdt'])

        bmk = bmk.reset_index(drop = True)

    def result(self):


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