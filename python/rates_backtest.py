# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 12:39:10 2016

@author: ubuntu
"""

import pandas as pd
import bbgREST as bbg
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

           



def get_data(ticker_dict):

    ticker_list = list(ticker_dict.keys())    

    dt_start = pd.datetime(1900,1,1)

    bl_data = bbg.get_histData(ticker_list,['PX_LAST'],dt_start,pd.datetime.today().date())

    bl_data = bl_data.pivot(index='date', columns='Ticker', values='PX_LAST')
    
    return bl_data

def my_test(x):
    
    if x['CBT4TNCN Index'] >= 2:
        return False
    elif x['CBT4TNCN Index'] <= -2:
        return True
    else:
        return 0
        
def get_position_score(data_df):
    
    
    temp_df = data_df.to_frame()
    
    temp_df = (temp_df-pd.rolling_mean(temp_df,window=52))/pd.rolling_std(temp_df,window=52)
    
    temp_df.dropna(inplace=True)

    
    temp_df['long_short'] = temp_df.apply(lambda x: my_test(x), axis=1)
    
    print(temp_df)

    return
    
    


ticker_dict = {'USGG10YR Index': 'US 10yr','CBT4TNCN Index': 'CFT Spec Net Longs',
               'TY1 COMB Comdty':'US 10yr Futures'}
               

data_df = get_data(ticker_dict)



positions_score = get_position_score(bl_data['CBT4TNCN Index'])




    
new_idx = bl_data['CBT4TNCN Index'].dropna().index



   
    
    
    
    
test_ticker_df = bl_data_df['CBT4TNCN Index']




'''



def long_only_ew(data, name='Long Only'):
    s = bt.Strategy(name, [bt.algos.RunOnce(),
                           bt.algos.SelectAll(),
                           bt.algos.WeighEqually(),
                           bt.algos.Rebalance()])
    
    return bt.Backtest(s, data)
    
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
    
        
test_ticker = test_ticker.to_frame()      
test_ticker['long_short'] = test_ticker.apply(lambda x: my_test(x), axis=1)


test_ticker = test_ticker.reindex(pd.date_range(test_ticker.index[0],test_ticker.index[-1],freq='B'))

yield_data = bl_data['USGG10YR Index']


sma = pd.rolling_mean(yield_data, 252)



signal = (yield_data < sma).to_frame()
signal.columns = columns = ['TY1 COMB Comdty']



s = bt.Strategy('above252sma', [bt.algos.SelectWhere(signal),
                               bt.algos.WeighEqually(),
                               bt.algos.Rebalance()])

data = bl_data['TY1 COMB Comdty'].to_frame().dropna()
t = bt.Backtest(s, data)

benchmark = long_only_ew(bl_data['TY1 COMB Comdty'].to_frame().dropna())

res = bt.run(t, benchmark)
res.plot()

#plot = bt.merge(yield_data, sma).plot(figsize=(15, 5))



combined_signal_index= test_ticker.join(backtest_ticker['TY1 COMB Comdty'])

look_forward = 30


combined_signal_index['ticker_return'] = combined_signal_index['TY1 COMB Comdty'].pct_change(look_forward)

combined_signal_index[['CBT4TNCN Index','long_short']] = combined_signal_index[['CBT4TNCN Index','long_short']].shift(3)



combined_signal_index[['TY1 COMB Comdty','ticker_return']] = combined_signal_index[['TY1 COMB Comdty','ticker_return']].shift(-look_forward)

combined_signal_index = combined_signal_index[:-look_forward]

combined_signal_index['ticker_return'].fillna(method='ffill',inplace=True)

result_frame = combined_signal_index[['long_short','ticker_return']].dropna(axis=0,how='any')
average_return = result_frame['ticker_return'].mean()



short_df = result_frame[result_frame['long_short'] == False]
short_df = short_df['ticker_return'] * -1
hit_short = len(short_df[short_df > 0]) / len(short_df)

short_mean_tst = ttest_1samp(short_df, -average_return)




long_df= result_frame[result_frame['long_short'] == True]

long_df = long_df['ticker_return']

hit_long= len(long_df[long_df > 0]) / len(long_df)
long_mean_tst = ttest_1samp(long_df, average_return)


fig,ax = plt.subplots(2,2)
ax1 = ax[0,0]    
ax2 = ax[0,1]    
ax3 = ax[1,0]    
ax4 = ax[1,1]

short_df.plot(kind='kde',ax=ax1,color='b', title='Level')
long_df.plot(kind='kde',ax=ax1,color='r')

ax1.legend(['short','long'], fontsize=6)


test = stats.ks_2samp(short_df, long_df)
means_test = ttest_ind(short_df, long_df)

'''

