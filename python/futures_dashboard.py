# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 17:23:22 2016

@author: Matt Brady
"""
import bbgREST as bbg
import pandas as pd
import quandl
import time
import plotly.plotly as py
import plotly.tools as tl
import plotly.graph_objs as go
from plotly.graph_objs import *
from datetime import datetime
import numpy as np
import bt as bt

import pandas.io.data as web

tl.set_credentials_file(username='mattgbrady', api_key='cmjs1osucd')

class futures():
    
    def __init__(self):
        
        self.name = 'hi'
        
def get_data(ticker_list):
    
    dt_start = pd.datetime(1992,12,1)

    bl_data = bbg.get_histData(ticker_list,['PX_LAST'],dt_start,pd.datetime.today().date(),freq='DAILY')
    
    bl_data = bl_data.pivot(index='date', columns='Ticker', values='PX_LAST')
    return bl_data

def beta(data, name='Strategy'):
    s = bt.Strategy(name, [bt.algos.RunOnce(),
                           bt.algos.SelectAll(),
                           bt.algos.WeighEqually(),
                           bt.algos.Rebalance()])
    
    return bt.Backtest(s, data)   

def download_data():
    
    futures_tickers = pd.read_csv('futures_tickers.csv')
    futures_tickers['number'] = futures_tickers['number'].to_string()
    tickers_download = futures_tickers['database'] + '/' + futures_tickers['exchange'] + '_' + futures_tickers['symbol'] + str(1) + '_' + futures_tickers['method'] 

    tickers_download = tickers_download.tolist()

    quandl.ApiConfig.api_key = 'YMqrB1SXyjSUkTHYwpJ2'

    merged_data = quandl.get(tickers_download)
    
    merged_data.to_csv('quandl_data.csv')

def get_data_csv():
    
    raw_data = pd.read_csv('quandl_data.csv',index_col=0)
    

    
    tickers = raw_data.columns.tolist()

    tickers = [x for x in tickers if "- Settle" in x]
    
    raw_data = raw_data[tickers]
    
   
    tickers_included = pd.read_csv('futures_tickers.csv')
    tickers_included = tickers_included[tickers_included['include'] == True]
    tickers_included['long_ticker_name'] = tickers_included['database'] + '/' + tickers_included['exchange'] + '_' + tickers_included['symbol'] + str(1) + '_' + tickers_included['method']  + ' - Settle'
    

    raw_data = raw_data[tickers_included['long_ticker_name'].tolist()]
    raw_data.index = pd.to_datetime(raw_data.index)
    
    raw_data.columns = tickers_included['short_name'].values.tolist()
    

    raw_data = raw_data.reindex(pd.date_range(raw_data.index[0],raw_data.index[-1],freq='B'),method='ffill')

    return raw_data

def trailing_return_score(raw_data,period=252):
    
    trailing_returns_df = pd.DataFrame()
    
    trailing_returns_df = raw_data.pct_change(periods=period)
    
    return trailing_returns_df
    

def plot_bar(data_df):
    
    new_df = pd.DataFrame(data=data_df.values[0], index=data_df.columns.tolist(),columns=['1 year return']).sort(columns='1 year return',ascending=False)

    data = [go.Bar(
                x= new_df.index.values,
                y= new_df[new_df.columns[0]].values
    
    
    )]

   
    layout = go.Layout(title='1 Year Total Return as of ' + data_df.index[0].strftime('%m-%d-%Y'))
   
    fig = go.Figure(data=data,layout=layout)
    py.iplot(fig,filename='one_yr_total_return')   


def drop_down_scatter(data_df):
    

    parser = {}
    loop_list = []
    true_false_dict = {}
    #stopped here
    default_list
    counter=0
    for column in data_df.columns:
        
        temp_df = data_df[column].dropna()
        name = 'trace' + str(counter)
        parser[name] = Scatter(
           x=temp_df.index, y=temp_df.values
        
        )
        loop_list.append(parser[name])
        counter = counter + 1
    fig = Figure(data=loop_list)
    py.iplot(fig)

    
def main():
    #download_data()
    raw_data = get_data_csv()
    trailing_1yr_return = trailing_return_score(raw_data)
    #current = plot_bar(trailing_1yr_return.iloc[-1:])
    drop_down_scatter(raw_data)


main()


'''
data_df = pd.read_csv('raw_data.csv',index_col=0)
data_df.index = pd.to_datetime(data_df.index)

data_df = data_df.reindex(pd.date_range(data_df.index[0],data_df.index[-1],freq='B'),method='ffill')
data_df_monthly = data_df.reindex(pd.date_range(data_df.index[0],data_df.index[-1],freq='M'),method='ffill')
print(data_df)
st = '19830331'
en = '20160731'

data_df['shift'] = data_df['SCF/CME_CL1_FR - Settle'].shift(252)
data_df_monthly['shift'] = data_df_monthly['SCF/CME_CL1_FR - Settle'].shift(12)
signal_bool = (data_df['SCF/CME_CL1_FR - Settle']>= data_df['shift']).to_frame()
 
signal_bool = signal_bool.shift(1)
    

 
signal_bool.dropna(inplace=True)

signal_bool = signal_bool[st:en]

signal_bool_m = (data_df_monthly['SCF/CME_CL1_FR - Settle']>= data_df_monthly['shift']).to_frame()
 
signal_bool_m = signal_bool_m.shift(1)
    

 
signal_bool_m.dropna(inplace=True)


asset_monthly_price= data_df['SCF/CME_CL1_FR - Settle'].reindex(pd.date_range(data_df['SCF/CME_CL1_FR - Settle'].index[0],data_df['SCF/CME_CL1_FR - Settle'].index[-1],freq='M'),method='ffill')



asset_return_df_1m = asset_monthly_price.pct_change(periods=1).dropna().to_frame()
asset_return_df_1m.columns = ['Crude']

asset_return_df_12m = asset_monthly_price.pct_change(periods=12).dropna().to_frame()
asset_return_df_12m.columns = ['Crude']

signal_bool_m = (asset_return_df_12m['Crude']>=0).to_frame()
 
signal_bool_m = signal_bool_m.shift(1)
    

 
signal_bool_m.dropna(inplace=True)
signal_bool_m = signal_bool_m[st:en]

ls_weights_df_m = pd.DataFrame(np.where(signal_bool_m == True,1,-1), index=signal_bool_m.index)
ls_weights_df_m.columns = ['Crude']
print(ls_weights_df_m)

asset_return_df = data_df['SCF/CME_CL1_FR - Settle'].pct_change(periods=1).dropna().to_frame()
asset_return_df.columns = ['Crude']

ls_weights_df = pd.DataFrame(np.where(signal_bool == True,1,-1), index=signal_bool.index)
ls_weights_df.columns = ['Crude']

long_short_portfolio_m = asset_return_df_1m[st:en] * ls_weights_df_m

print(asset_return_df_1m[st:en] )
long_short_portfolio_m =  long_short_portfolio_m.add(1).cumprod()
long_short_portfolio_m.dropna(inplace=True)


long_short_portfolio = asset_return_df[st:en] * ls_weights_df



long_short_portfolio =  long_short_portfolio.add(1).cumprod()


long_short_portfolio = long_short_portfolio[st:en]
long_short_portfolio_m = long_short_portfolio_m[st:en]


long_short_portfolio_results = beta(long_short_portfolio, name='Crude Long/Short 252 Days')
long_short_portfolio_results_m = beta(long_short_portfolio_m, name='Crude Long/Short 12M')

res = bt.run(long_short_portfolio_results,long_short_portfolio_results_m)    

res.plot()

res.display()

long_weights_df = pd.DataFrame(np.where(ls_weights_df == 1,1,0), index=ls_weights_df.index)
long_weights_df.dropna(inplace=True)
long_weights_df.columns = ['Crude Long']

short_weights_df = pd.DataFrame(np.where(ls_weights_df == -1,-1,0), index=ls_weights_df.index)
short_weights_df.dropna(inplace=True)
short_weights_df.columns = ['Crude Short']

asset_return_df = data_df['SCF/CME_CL1_FR - Settle'].pct_change(periods=1).to_frame()
asset_return_df.columns = ['Crude']

long_short_portfolio = asset_return_df * ls_weights_df
long_short_portfolio.dropna(inplace=True)
long_short_portfolio.columns = ['Crude Long/Short']
long_short_portfolio.dropna(inplace=True)

long_only_combined = pd.concat([long_weights_df,asset_return_df],axis=1)
long_only_combined.dropna(axis=0,inplace=True)

long_portfolio = (long_only_combined['Crude Long'] * long_only_combined['Crude']).to_frame()
long_portfolio.columns = ['Crude Long Only']
long_portfolio.dropna(inplace=True)


short_only_combined = pd.concat([short_weights_df,asset_return_df],axis=1)
short_only_combined.dropna(axis=0,inplace=True)

short_portfolio = (short_only_combined['Crude Short'] * short_only_combined['Crude']).to_frame()
short_portfolio.columns = ['Crude Short Only']
short_portfolio.dropna(inplace=True)

three_portfolios = pd.concat([long_short_portfolio,long_portfolio,short_portfolio],axis=1)
three_portfolios.fillna(value=0,inplace=True)


long_short_portfolio = beta(three_portfolios['Crude Long/Short'], name='Crude Long/Short')
long_portfolio = beta(three_portfolios['Crude Long Only'], name='Crude Long')
short_portfolio = beta(three_portfolios['Crude Short Only'], name='Crude Short')
'''
#print(long_short_portfolio)
#res = bt.run(long_short_portfolio)    
    
#res.plot()
#res.display()



'''
py.sign_in('mattgbrady','cmjs1osucd')


quandl.ApiConfig.api_key = 'YMqrB1SXyjSUkTHYwpJ2'




ticker_list = ['SCF/CME_CL1_FW','SCF/CME_CL1_FR','SCF/CME_CL1_FB']

merged_data = quandl.get(ticker_list)



raw_data = merged_data[['SCF/CME_CL1_FW - Settle','SCF/CME_CL1_FR - Settle','SCF/CME_CL1_FB - Settle']]


raw_data.to_csv('raw_data.csv')

crude_df = get_data(['CL1 COMB Comdty'])
crude_df.index = pd.to_datetime(crude_df.index)


data_df = pd.read_csv('raw_data.csv',index_col=0)
data_df.index = pd.to_datetime(data_df.index)
#for each in data_df.index:
#    print(type(each))

data_df_combined = pd.concat([data_df,crude_df],axis=1)

data_df_combined.plot()

new_df_pr = (data_df_combined.reindex(pd.date_range(crude_df.index[0],crude_df.index[-1],freq='B'))).fillna(method='ffill')

new_df_tr = new_df_pr.add(1).cumprod()

new_df_pr['CL1 COMB Comdty'].to_csv('test.csv')

data_df_rolling_three = new_df_pr.pct_change(periods=252)

data_df_shift = data_df.shift(252)

data_df_diff = data_df - data_df_shift
data_df_diff.dropna(inplace=True)



data_format = [
    go.Scatter(
        x=data_df_rolling_three.index, # assign x as the dataframe column 'x'
        y=data_df_rolling_three['SCF/CME_CL1_FR - Settle'],
        name='Backwards Ratio'
    ),
    go.Scatter(
        x=data_df_rolling_three.index, # assign x as the dataframe column 'x'
        y=data_df_rolling_three['CL1 COMB Comdty'],
        name='CL1 Bloomberg Ratio Adj'
    ),
]

data_format_2 = [
    go.Scatter(
        x=new_df_tr.index, # assign x as the dataframe column 'x'
        y=new_df_tr['SCF/CME_CL1_FR - Settle'],
        name='Backwards Ratio'
    ),
    go.Scatter(
        x=new_df_tr.index, # assign x as the dataframe column 'x'
        y=new_df_tr['CL1 COMB Comdty'],
        name='CL1 Bloomberg Ratio Adj'
    ),
]

layout = dict(
    title='Time series with range slider and selectors',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(step='all'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(count=1,
                     label='1y',
                     step='year',
                     stepmode='backward'),
                dict(count=5,
                    label='5yr',
                    step='year',
                    stepmode='backward'),
                dict(count=10,
                    label='10yr',
                    step='year',
                    stepmode='backward')
            ])
        ),
        rangeslider=dict(),
        type='date'
    )
)

fig = dict(data=data_format, layout=layout)
py.iplot(fig, filename='Rolling Three Year Return', validate=False, layout=layout)
fig = dict(data=data_format_2, layout=layout)
py.iplot(fig, filename='Cumulative Total Return', validate=False, layout=layout)



data_format = [
    go.Scatter(
        x=data_df.index, # assign x as the dataframe column 'x'
        y=data_df['SCF/CME_CL1_FW - Settle'],
        name='Calendar Weighted'
    ),
    go.Scatter(
        x=data_df.index, # assign x as the dataframe column 'x'
        y=data_df['SCF/CME_CL1_FR - Settle'],
        name='Backwards Ratio'
    ),
    go.Scatter(
        x=data_df.index, # assign x as the dataframe column 'x'
        y=data_df['SCF/CME_CL1_FB - Settle'],
        name='Backwards Panama'
    )]
fig = dict(data=data_format, layout=layout)
py.iplot(fig, filename='Crude Price', validate=False, layout=layout)


table_df = pd.DataFrame()

table_df = data_df[['SCF/CME_CL1_OB - Settle','one_year']].dropna()

table_df['Percentage'] = (data_df['one_year'] / data_df['SCF/CME_CL1_OB - Settle']) - 1


table_df = table_df[-252:]

table_df.columns = ['Crude CL1','Trailing One Year Crude CL1', 'Percentage Difference']

table = FF.create_table(table_df,index=True, index_title='Date')
py.iplot(table, filename='crude_table')
'''