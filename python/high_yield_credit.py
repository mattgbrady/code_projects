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

def spread_val_score(data_df):  

    data_df = np.log(data_df * 100)
    data_df['spread_z_ema'] = (data_df - pd.ewma(data_df, min_periods = 60, halflife=60)) / pd.ewmstd(data_df, halflife=60)
    
    data_df['spread_z_ema'].dropna(inplace=True)
    
    data_df['spread_z_ema'].plot()
    plt.show()
    
    return data_df['spread_z_ema']
    

def trsy_bool_position(data_df):
    
    date_array = data_df.index.values
    
    trsy_bool_array = []
    for date in range(0,len(date_array)):

        score = data_df.ix[date].values[0]

        if score == 1:
            trsy_bool_array.extend([-1])
        elif score == -1:
            trsy_bool_array.extend([1])
        elif score == 0:
            trsy_bool_array.extend([0])
        else:
            temp = score * -1
            trsy_bool_array.extend([0])

    trsy_bool_df = pd.DataFrame(trsy_bool_array,index=date_array )
    return trsy_bool_df
        

def spread_val_backtest(data_df_score, return_series_df):
    
    
    signal_bool = spread_wghts(data_df_score)
    
    signal_bool.columns = [return_series_df.columns.values[0]]
    
    trsy_bool = trsy_bool_position(signal_bool)
    trsy_bool.columns = [return_series_df.columns.values[1]]
    
    combined_positions = pd.concat([signal_bool,trsy_bool], axis=1)
    

    combined_positions = combined_positions.shift(1)
     
    start_date = np.min(combined_positions.index.values)
    end_date = np.max(combined_positions.index.values)
    
    return_series_df = return_series_df.ix[start_date:end_date]

    s1 = bt.Strategy('Spread Valuation', [bt.algos.WeighTarget(combined_positions),
                               bt.algos.Rebalance()])

    strategy = bt.Backtest(s1, return_series_df)
    

    
    res = bt.run(strategy)
    
    res.plot()
    res.plot_weights()
    res.display()

    return res
     
    pass    

def spread_wghts(data_df_score):

    hp_months_cheap = 6
    hp_months_rich = 6

    counter = 0
    
    date_array = data_df_score.index.values
    
    signal = []
    
    signal_abs_threshold_cheap= 1.5
    signal_abs_threshold_rich= 1.5
    
  
    for date in range(0,len(date_array)):
  
        

        if len(data_df_score) - 1 >= counter:
            
            score = data_df_score.ix[counter]

            if score > signal_abs_threshold_cheap:
                counter = counter+hp_months_cheap
                temp_list = [1] * hp_months_cheap
                signal.extend(temp_list)
                
            elif score < (signal_abs_threshold_rich * -1):
                counter = counter+hp_months_rich
                temp_list = [-1] * hp_months_rich 
                signal.extend(temp_list)
                
            elif (score <= signal_abs_threshold_cheap) and score >= (signal_abs_threshold_rich * -1):
                counter = counter + 1
                signal.extend([0])
   
    
    signal = signal[:len(date_array)]  
    
        
    return pd.DataFrame(signal,columns=['score'], index=data_df_score.index.values)
 
    
def spread_crossover(data_df,slow=1,fast=12):
    
    
    
    spread_log = pd.DataFrame(np.log(data_df.ix[:,0] * 100))
    
    
    data_df['spread_z_ma'] = (spread_log - pd.expanding_mean(spread_log, min_periods=24))/  pd.expanding_std(spread_log, min_periods=24)
    
    data_df['spread_z_ema'] = (spread_log - pd.ewma(spread_log, min_periods = 24, halflife=12)) / pd.ewmstd(spread_log, halflife=12)
    
    
    data_df['spread_z_ema'] = pd.rolling_mean(data_df['spread_z_ema'], window=3)
     
    data_df['slow'] = pd.rolling_mean(data_df['US HY Spread'],slow)

    data_df['fast'] = pd.rolling_mean(data_df['US HY Spread'],fast)
    
    data_df['diff'] = (data_df['slow'] - data_df['fast']) * -1
    
    data_df['diff'] = data_df['diff'] + 1
    
    data_df['diff'] = np.log(data_df['diff'])
    
    data_df['tren_z_ma'] = (data_df['diff'] - pd.expanding_mean(data_df['diff'], min_periods=24))/  pd.expanding_std(data_df['diff'], min_periods=24)
    
    data_df['tren_z_ma']  = pd.rolling_mean(data_df['tren_z_ma'], window=3)
    trend_valuation_df = pd.concat([data_df['spread_z_ema'],data_df['tren_z_ma']], axis=1)

    
    trend_valuation_df.dropna(inplace=True)
    trend_valuation_df.plot()
    plt.show()
    
    algo_wghts_df = pd.DataFrame()
    wghts_array = []
    
    valuation_threshold_cheap = 1
    valuation_threshold_rich = -1.0
    trend_threshold_tightening = 0.1
    trend_threshold_widening = -0.1
    
    data_df['spread_z_ma'].plot()
    plt.show()
    
    

    for score in trend_valuation_df.values:
        valuation_score = score[0]
        trend_score = score[1]
        
        if (trend_score >= -0.2 and valuation_score >= -1):
            wghts_array.append(min(1,abs(trend_score-valuation_score) / 1))
        else:
            wghts_array.append(0)
        #elif trend_score <= -0.1 and valuation_score <= valuation_threshold_cheap:
        #    wghts_array.append(-1)
        #elif valuation_score >= valuation_threshold_cheap:
        #    wghts_array.append(1)
        #else:
        #    wghts_array.append(0)   
    
    wghts_df = pd.DataFrame(wghts_array, index = trend_valuation_df.index)
    
 

    long = wghts_df[wghts_df == 1].count()[0] / len(trend_valuation_df)
    neutral = wghts_df[wghts_df == 0].count()[0] / len(trend_valuation_df)
    short = wghts_df[wghts_df == -1].count()[0] / len(trend_valuation_df)
    
    wghts_df.columns = [data_df.columns.values[1]]
    
    wghts_df = wghts_df.shift(1)
    
    
    s1 = bt.Strategy('Valuation & Trend ', [bt.algos.WeighTarget(wghts_df),
                               bt.algos.Rebalance()])
    
    return_data = data_df.ix[:,1].to_frame()
    return_data.columns = [data_df.columns.values[1]]

    strategy = bt.Backtest(s1, return_data)
    
    res = bt.run(strategy)
    
    res.plot()
    res.display()
    print(long,neutral,short)


   
    
      
'''            
            
        if trend_score >= trend_threshold and valuation_score >= (valuation_threshold * -1):
            wghts_array.append(1)
        elif trend_score <= (trend_threshold * -1) and valuation_score <= valuation_threshold:
            wghts_array.append(-1)
        elif trend_score <= trend_threshold and trend_score >= (trend_threshold * -1):
            wghts_array.append(1)
        else:
            wghts_array.append(0)
  

    
    data_df['spread_level_wght'] = (np.abs(data_df['spread_z_ema']) / 5) * data_df['spread_z_ema']
    data_df['spread_level_wght'] = data_df['spread_level_wght'].clip(-1, 1)
    
    data_df['spread_level_wght']  = np.where(np.abs(data_df['spread_level_wght']) == 1, data_df['spread_level_wght'], 0)
    
    
    asset_wght = data_df['spread_level_wght'].to_frame()
    asset_wght.columns = [data_df.columns.values[1]]
    
    
    cash_wght = (data_df['spread_level_wght'] * -1).to_frame()
    cash_wght.columns = [data_df.columns.values[2]]
 
    combine_asset_csh_wght = pd.concat([asset_wght,cash_wght], axis=1)
    
    combine_asset_csh_wght = combine_asset_csh_wght.shift(1)
    
    s1 = bt.Strategy('Long/Short Spread Level ', [bt.algos.WeighTarget(combine_asset_csh_wght),
                               bt.algos.Rebalance()])   
                               
    return_data = data_df.ix[:,1:3]

    strategy = bt.Backtest(s1, return_data)
    
    res = bt.run(strategy)

    
    res.plot()
 
    pass
'''   
    
    
    
def main():
    
    
    data = pd.read_excel('us_hy_credit.xlsx', index_col=0)
    data_df = data.copy()
    data_df[['HY Tot Index','US Trs Tot Index','IG Tot Index','Cash Tot Index']] = data_df[['US HY Return','US Int. Trsy Return','US IG Return','Cash Return']].add(1).cumprod()


    hy_spread = data_df['US HY Spread'].to_frame()
    hy_return_series =  data_df[['US HY Return','US Int. Trsy Return']].add(1).cumprod()

    
    hy_val_score = spread_val_score(hy_spread)
    
    hy_val_res = spread_val_backtest(hy_val_score,hy_return_series)
    
    hy_val_res.plot()
    
    #spread_crossover(hy_df)

''' 
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
