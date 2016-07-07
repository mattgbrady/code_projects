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


data_df = pd.read_excel('us_hy_credit.xlsx', index_col=0)




hy_df = data_df[['US HY Monthly Return','US HY Spread']].copy()







slow = 2
fast = 12


hy_df['slow'] = pd.rolling_mean(hy_df['US HY Spread'],slow)

hy_df['fast'] = pd.rolling_mean(hy_df['US HY Spread'],fast)

hy_df.dropna(inplace=True)


signal = (hy_df['fast'] >= hy_df['slow']).to_frame()

hy_df['HY Tot Index'] = data_df['US HY Monthly Return'].add(1).cumprod()


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



''''
#testing CFTC net long spec positions based on rolling z-score threshold
positions_score = get_position_score(bl_data['CBT4TNCN Index'])
'''