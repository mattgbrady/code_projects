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


    
    
def ma_cross_spread_test(hy_df,fast,slow):
    
    hy_df_copy = hy_df.copy()
    
    hy_df_copy['slow'] = pd.rolling_mean(hy_df_copy['US HY Spread'],slow)

    hy_df_copy['fast'] = pd.rolling_mean(hy_df_copy['US HY Spread'],fast)

    hy_df_copy.dropna(inplace=True)
    
   
    signal_bool = (hy_df_copy['fast'] >= hy_df_copy['slow']).to_frame()
    
    signal_bool = signal_bool.shift(1)
    
    signal_bool.dropna(inplace=True)

    signal_wghts1 = pd.DataFrame(np.where(signal_bool == True, -1, 1), index=signal_bool.index)
    signal_wghts1.columns = ['HY Tot Index']
    

    s1 = bt.Strategy('Long/Short ma crossover: ' + str(fast) + 'm ' + str(slow) + 'm', [bt.algos.WeighTarget(signal_wghts1),
                               bt.algos.Rebalance()])
                        
    signal_wghts2 = pd.DataFrame(np.where(signal_bool == True, 0, 1), index=signal_bool.index)      
    signal_wghts2.columns = ['HY Tot Index']                  
                     
    s2 = bt.Strategy('Long/Neutral ma crossover: ' + str(fast) + 'm ' + str(slow) + 'm', [bt.algos.WeighTarget(signal_wghts2),
                               bt.algos.Rebalance()])                               
                               
                     
    trsry_wghts = pd.DataFrame(np.where(signal_bool == True, 1, 0), index=signal_bool.index)
    
    trsry_wghts.columns = ['US Trs Tot Index']
    
    combine_hy_trs_wght = pd.concat([signal_wghts2,trsry_wghts], axis=1)
    
    
    ig_credit_wght = pd.DataFrame(np.where(signal_bool == True, 1, 0), index=signal_bool.index)
    
    
    ig_credit_wght.columns = ['IG Tot Index']
    
    combine_hy_ig_wght = pd.concat([signal_wghts2,ig_credit_wght], axis=1)    
    
    
    
    data1 = hy_df['HY Tot Index'].to_frame()

    data1.columns = ['HY Tot Index']
    
    data2 = hy_df[['HY Tot Index','US Trs Tot Index']]
    
    data3 = hy_df[['HY Tot Index','IG Tot Index']]
    
    
    s3 = bt.Strategy('Long/Neutral w/ Treasuries ma crossover: ' + str(fast) + 'm ' + str(slow) + 'm', [bt.algos.WeighTarget(combine_hy_trs_wght),
                               bt.algos.Rebalance()]) 

    s4 = bt.Strategy('Long/Neutral w/ IG ma crossover: ' + str(fast) + 'm ' + str(slow) + 'm', [bt.algos.WeighTarget(combine_hy_ig_wght),
                               bt.algos.Rebalance()]) 
    
    
    hy_equal = pd.DataFrame(np.where(signal_bool == True, 0.5, 0.5), index=signal_bool.index)      
    
    
    hy_equal.columns = ['HY Tot Index']   
    
    
    ig_equal = pd.DataFrame(np.where(signal_bool == True, 0.5, 0.5), index=signal_bool.index)      
    
    
    ig_equal.columns = ['IG Tot Index'] 
    
    combine_hy_ig_equal = pd.concat([hy_equal,ig_equal], axis=1)     
    
    s5= bt.Strategy('50% HY / 50% IG', [bt.algos.WeighTarget(combine_hy_ig_equal),
                               bt.algos.Rebalance()]) 
    
    long_only = long_only_ew(data1, name='HY Long Beta')

    test1 = bt.Backtest(s1, data1)

    test2 = bt.Backtest(s2, data1)
    
    test3 = bt.Backtest(s3, data2)
    
    test4 = bt.Backtest(s4, data3)
    
    hy_ig_port = bt.Backtest(s5, data3)

    
    return test1, test2, test3, test4, hy_ig_port, long_only

    
    
    
    
def main():
    
    
    data_df = pd.read_excel('us_hy_credit.xlsx', index_col=0)
    hy_df = data_df[['US HY Return','US HY Spread','US Int. Trsy Return']].copy()
    hy_df[['HY Tot Index','US Trs Tot Index','IG Tot Index']] = data_df[['US HY Return','US Int. Trsy Return','US IG Return']].add(1).cumprod()

    #spread_dic = {1: [3,6,9,12,24], 3: [6,9,12,24], 6: [9,12,24], 9: [12,24], 12: [24]}
   
    spread_dic = {1: [12]}
    
    long_short_results = {}
    long_neutral_results = {}

    
    for key, list in spread_dic.items():
        fast = key
        for value in list:
            slow = value
            long_short, long_neutral, long_trsy, long_ig, hy_ig_port, beta =   ma_cross_spread_test(hy_df,fast,slow)  
            
        res = bt.run(long_short,long_neutral,long_trsy,long_ig,hy_ig_port,beta)
        #res.plot() 
        #res.display()

        #res.to_csv(sep=',',path='output.csv')


    data_df['US HY Spread log'] = np.log(data_df['US HY Spread'] * 100)
    hy_df['HY Spread MA'] = (data_df['US HY Spread log'] - pd.expanding_mean(data_df['US HY Spread log'], min_periods=24))/  pd.expanding_std(data_df['US HY Spread log'], min_periods=24)
    
    
    hy_df['HY Spread Z EMA'] = (data_df['US HY Spread log'] - pd.ewma(data_df['US HY Spread log'], min_periods = 24, halflife=12)) / pd.ewmstd(data_df['US HY Spread log'], halflife=12)

    hy_df['US HY Spread log'] = data_df['US HY Spread log']
    
    ax1 = hy_df['HY Spread Roll Z'].plot()
    hy_df['US HY Spread log'].plot(ax=ax1,secondary_y = True)
    
    
    
    plt.show()

    
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
