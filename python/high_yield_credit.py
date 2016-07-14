# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 12:39:10 2016

@author: Matt Brady
"""

import pandas as pd
#import bbgREST as bbg
from pandas_datareader import data as pdr
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.dates as mdates
import matplotlib
import numpy as np
from datetime import datetime
import random
import matplotlib.pyplot as mpl
import scipy.stats as stats
from scipy.stats import ttest_ind
from scipy.stats import ttest_1samp
import bt as bt

import plotly.plotly as py
from plotly.tools import FigureFactory as FF


import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

class WeighTarget(bt.Algo):
    """
    Sets target weights based on a target weight DataFrame.

    Args:
        * target_weights (DataFrame): DataFrame containing the target weights

    Sets:
        * weights

    """

    def __init__(self, target_weights):
        self.tw = target_weights

    def __call__(self, target):
        # get target weights on date target.now
        if target.now in self.tw.index:
            w = self.tw.ix[target.now]

            # save in temp - this will be used by the weighing algo
            # also dropping any na's just in case they pop up
            target.temp['weights'] = w.dropna()

        # return True because we want to keep on moving down the stack
        return True   


#data from bloomberg
def get_data(ticker_dict):

    ticker_list = list(ticker_dict.keys())    

    dt_start = pd.datetime(1900,1,1)

    bl_data = bbg.get_histData(ticker_list,['PX_LAST'],dt_start,pd.datetime.today().date())

    bl_data = bl_data.pivot(index='date', columns='Ticker', values='PX_LAST')
    
    return bl_data

def long_only_ew(data, name='Long Only'):
    s = bt.Strategy(name, [bt.algos.RunOnce(),
                           bt.algos.SelectAll(),
                           bt.algos.WeighEqually(),
                           bt.algos.Rebalance()])
    
    return bt.Backtest(s, data)


    
    
def ma_cross_spread_test(data_df,fast,slow):
    
    data_df_copy = data_df.copy()
    
    data_df_copy['slow'] = pd.rolling_mean(data_df_copy['US HY Spread'],slow)

    data_df_copy['fast'] = pd.rolling_mean(data_df_copy['US HY Spread'],fast)

    data_df_copy.dropna(inplace=True)
    
   
    signal_bool = (data_df_copy['fast'] >= data_df_copy['slow']).to_frame()
    
    signal_bool = signal_bool.shift(1)
    
    signal_bool.dropna(inplace=True)

    signal_wghts1 = pd.DataFrame(np.where(signal_bool == True,0, 1), index=signal_bool.index)
    signal_wghts1.columns = ['HY Tot Index']
    
    cash_wght = pd.DataFrame(np.where(signal_bool == True,1, 0), index=signal_bool.index)
    cash_wght.columns = ['Cash Tot Index']

    combine_hy_csh_wght = pd.concat([signal_wghts1,cash_wght], axis=1)

    return_data = data_df_copy[['HY Tot Index','Cash Tot Index']]

    s1 = bt.Strategy('Long/Short Spread Crossover: ' + str(fast) + 'm ' + str(slow) + 'm', [bt.algos.WeighTarget(combine_hy_csh_wght),
                               bt.algos.Rebalance()])   
 
    beta_return = data_df_copy['HY Tot Index'].to_frame()
    
    long_only = long_only_ew(beta_return, name='HY Long Beta')

    results = bt.Backtest(s1, return_data)

    
    return results, long_only


def ma_cross_tot_rt_test(data_df,fast,slow):
    
    data_df_copy = data_df.copy()
    
    data_df_copy['slow'] = pd.rolling_mean(data_df_copy['HY Tot Index'],slow)

    data_df_copy['fast'] = pd.rolling_mean(data_df_copy['HY Tot Index'],fast)

    data_df_copy.dropna(inplace=True)
    
   
    signal_bool = (data_df_copy['fast'] >= data_df_copy['slow']).to_frame()
    
    signal_bool = signal_bool.shift(1)
    
    signal_bool.dropna(inplace=True)

    signal_wghts1 = pd.DataFrame(np.where(signal_bool == True, 1, -1), index=signal_bool.index)
    signal_wghts1.columns = ['HY Tot Index']
    
    cash_wght = pd.DataFrame(np.where(signal_bool == True,-1, 1), index=signal_bool.index)
    cash_wght.columns = ['Cash Tot Index']

    combine_hy_csh_wght = pd.concat([signal_wghts1,cash_wght], axis=1)

    return_data = data_df_copy[['HY Tot Index','Cash Tot Index']]

    s1 = bt.Strategy('Long/Short Total Return Crossover: ' + str(fast) + 'm ' + str(slow) + 'm', [bt.algos.WeighTarget(combine_hy_csh_wght),
                               bt.algos.Rebalance()])   

    results = bt.Backtest(s1, return_data)

    
    return results   
    
    
    
    
    
def main():
    
    
    data = pd.read_excel('us_hy_credit.xlsx', index_col=0)
    data_df = data.copy()
    data_df[['HY Tot Index','US Trs Tot Index','IG Tot Index','Cash Tot Index']] = data_df[['US HY Return','US Int. Trsy Return','US IG Return','Cash Return']].add(1).cumprod()

    spread_dic = {1: [3,6,9,12,24,36], 3: [6,9,12,24,36], 6: [9,12,24,36], 9: [12,24,36], 12: [24,36]}
   
    spread_dic = {6: [12]}
    
    long_short_results = {}
    long_neutral_results = {}
    
    
    data_df['HY Spread Change'] = data_df['US HY Spread'].diff()
    
    period = 3
    data['HY Spread 3m Avg Chng'] = pd.rolling_mean(data_df['HY Spread Change'], window=period)
    
    
    data['HY Breakeven'] = data_df['US HY Spread'] / 12
    
    data['HY Spread 3m Avg Chng'].plot()
    

    plt.show()
    
    data['HY Breakeven'].plot()
    plt.show()
    
    
    signal_bool = pd.DataFrame()
    signal_bool = (data['HY Breakeven'] >= data['HY Spread 3m Avg Chng']).to_frame()
    
    print(data[['HY Breakeven','HY Spread 3m Avg Chng']])
    

'''    
    for key, list in spread_dic.items():
        fast = key
        for value in list:
            slow = value
            #long_short_cash_spread, beta = ma_cross_spread_test(data_df,fast,slow)   
            long_short_cash_tot = ma_cross_tot_rt_test(data_df,fast,slow)   
            #res = bt.run(long_short_cash_spread,long_short_cash_tot, beta)
            res = bt.run(long_short_cash_tot)
            #res.plot(logy=True) 
            res.display()

'''
#res.to_csv(sep=',',path='output.csv')
        
    

'''
    data_df['US HY Spread log'] = np.log(data_df['US HY Spread'] * 100)
    hy_df['HY Spread MA'] = (data_df['US HY Spread log'] - pd.expanding_mean(data_df['US HY Spread log'], min_periods=24))/  pd.expanding_std(data_df['US HY Spread log'], min_periods=24)
    
    
    hy_df['HY Spread Z EMA'] = (data_df['US HY Spread log'] - pd.ewma(data_df['US HY Spread log'], min_periods = 24, halflife=12)) / pd.ewmstd(data_df['US HY Spread log'], halflife=12)

    hy_df['US HY Spread log'] = data_df['US HY Spread log']
    
    ax1 = hy_df['HY Spread Roll Z'].plot()
    hy_df['US HY Spread log'].plot(ax=ax1,secondary_y = True)
    
    
    
    plt.show()
'''
    
main()
    






'''


signal.columns = ['HY Tot Index']

signal = signal.shift(1)

signal['HY Tot Index'] = np.where(signal == False, 0, 1)


s = bt.Strategy('ma crossover', [bt.algos.WeighTarget(signal),
                               bt.algos.Rebalance()])

data = hy_df['HY Tot Index'].to_frame()

data.columns = ['HY Tot Index']

t = bt.Backtest(s, data)


res = bt.run(t)
res.plot()



'''
