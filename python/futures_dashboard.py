# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 17:23:22 2016

@author: Matt Brady
"""

import pandas as pd
import quandl
import time
import plotly.plotly as py
import plotly.tools as tl
from plotly.tools import FigureFactory as FF 
import plotly.graph_objs as go
from plotly.graph_objs import *
from datetime import datetime
import numpy as np
import bt as bt

import pandas.io.data as web
import dropbox

tl.set_credentials_file(username='mattgbrady', api_key='cmjs1osucd')


quandl.ApiConfig.api_key = 'YMqrB1SXyjSUkTHYwpJ2'

class futures():
    
    def __init__(self):
        
        self.name = 'hi'
        

def beta(data, name='Strategy'):
    s = bt.Strategy(name, [bt.algos.RunOnce(),
                           bt.algos.SelectAll(),
                           bt.algos.WeighEqually(),
                           bt.algos.Rebalance()])
    
    bt_engine = bt.Backtest(s, data) 
    res = bt.run(bt_engine) 
    return res  

def download_data():
    
    futures_tickers = pd.read_csv('futures_tickers.csv')
    tickers_download = futures_tickers['database_ticker']
    
    print(len(tickers_download))
    tickers_download = tickers_download.tolist()

    quandl.ApiConfig.api_key = 'YMqrB1SXyjSUkTHYwpJ2'

    merged_data = quandl.get(tickers_download)
    
    merged_data.fillna(method='ffill',limit=2,inplace=True)
    
    merged_data.to_csv('quandl_data.csv')

def get_data_csv():
    
    raw_data = pd.read_csv('quandl_data.csv',index_col=0)
    
    
    tickers = raw_data.columns.tolist()
    print(len(tickers))
    
    build_df = pd.DataFrame()
    counter = 0
    for ticker in tickers:
        counter = counter + 1
        print(counter)
        hyphen_location = ticker.find('-')
        ticker_name = ticker[0:hyphen_location-1]
        ticker_type = ticker[hyphen_location+2:len(ticker)]
        loop_df = pd.DataFrame(raw_data[ticker])
        loop_df['ticker'] = ticker_name
        loop_df['type'] = ticker_type
        loop_df.columns = ['value','ticker','type']
        loop_df.dropna(inplace=True)
        build_df = pd.concat([build_df,loop_df],axis=0)

    build_df.to_csv('quandl_data_formatted.csv')
'''
    tickers = [x for x in tickers if "- Settle" in x]
    
    raw_data = raw_data[tickers]
    
   
    tickers_included = pd.read_csv('futures_tickers.csv')
    tickers_included = tickers_included[tickers_included['include'] == True]
    tickers_included['long_ticker_name'] = tickers_included['database_ticker'] +  ' - Settle'
    

    raw_data = raw_data[tickers_included['long_ticker_name'].tolist()]
    raw_data.index = pd.to_datetime(raw_data.index)
    
    raw_data.columns = tickers_included['short_name_unique'].values.tolist()
    

    raw_data = raw_data.reindex(pd.date_range(raw_data.index[0],raw_data.index[-1],freq='B'),method='ffill')

    return raw_data
'''
def trailing_return_score(raw_data,period=252):
    
    trailing_returns_df = pd.DataFrame()
    
    trailing_returns_df = raw_data.pct_change(periods=period)
    
    return trailing_returns_df
    

def plot_bar(data_df):
    
    new_df = pd.DataFrame(data=data_df.values[0], index=data_df.columns.tolist(),columns=['1 year return']).sort(columns='1 year return',ascending=False)

    data = [go.Bar(
                x= new_df.index.values,
                y= new_df[new_df.columns[0]].values
                #orientation = 'h'
    
    
    )]

   
    layout = go.Layout(title='1 Year Total Return as of ' + data_df.index[0].strftime('%m-%d-%Y'),
                       yaxis=dict(tickformat = '%'))
                       

    fig = go.Figure(data=data,layout=layout)
    py.iplot(fig,filename='one_yr_total_return')   


def drop_down_scatter(data_df):
    

    parser = {}
    loop_list = []
    #list([dict(arg=[],lable=,method='restyle',),dict(,,,)])
    #stopped here
    default_list = [False] * len(data_df.columns.tolist())
    plotly_tuple = ()
    counter=0
    button_list = []
    #for column in data_df.columns

    for column in data_df.columns:
        temp_df = data_df[column].dropna()
        name = 'trace' + str(counter)
        parser[name] = Scatter(
           x=temp_df.index, y=temp_df.values, name=temp_df.name)

        loop_list.append(parser[name])
        counter = counter + 1

    counter = 0
    loop_list_menus = []
    for column in data_df.columns:
        loop_list_menus = default_list.copy()
        loop_list_menus[counter] = True
        counter = counter + 1
        button_list.append(dict(args=['visible', loop_list_menus],label=column,method='restyle'))
    

    
    layout = Layout(title='C1 Ratio Adj. 1st of Month Roll',autosize=False,margin=go.Margin(l=5,r=50,pad=0),
                    
                    updatemenus=list([dict(
                                            x=-0.5,
                                            y=5,
                                            yanchor='top',
                                            buttons=button_list)]))


    fig = Figure(data=loop_list,layout=layout)
    

    
    
    py.iplot(fig,filename='drop_down_chart')

def one_year_test(data_df):
    pass
    
def create_table(data_df):
    
    table = FF.create_table(table_df,index=True, index_title='Date')
    py.iplot(table, filename='crude_table')
    

def atr():
    
    
    data_df = pd.read_csv('quandl_data_formatted.csv',index_col=0)
    ticker_info_df = pd.read_csv('futures_tickers.csv',index_col=0)
    
    ticker_info_df = ticker_info_df[ticker_info_df['include'] == True]
    
    
    data_df['short_name'] = data_df['ticker'].map(ticker_info_df['short_name_unique'])
    
    ticker_list = ticker_info_df['short_name_unique'].unique()
    
    print(len(ticker_list))
 
    outer_loop_df = pd.DataFrame()
    for ticker in ticker_list:
        print(ticker)
        loop_df = data_df[data_df['short_name'] == ticker]

        loop_df.pop('ticker')
        loop_df = loop_df.pivot(columns='type',values='value')

        #find max 
        loop_df['range_high_low'] = loop_df['High'] - loop_df['Low']
        loop_df['range_high'] = abs(loop_df['High'] - loop_df['Settle'].shift(1))     
        loop_df['range_low'] = abs(loop_df['Low'] - loop_df['Settle'].shift(1))
        
        loop_df['ATR'] = loop_df[['range_high_low','range_high','range_low']].max(axis=1)

        loop_df['ema_atr'] = pd.ewma(loop_df['ATR'], min_periods = 100, halflife=100)
        
        loop_df['ema_atr_relative'] = loop_df['ema_atr'] / loop_df['Settle']
        
        loop_df['ema_art_percentile'] = loop_df['ema_atr_relative'].rank(pct=True)
        
        loop_df = loop_df[['ema_atr','ema_atr_relative']]
        
        inner_loop_df = pd.DataFrame()
        for column in loop_df.columns:
            
            temp_df = loop_df[column].to_frame()
            
            temp_df['type'] = column
       
            temp_df.columns = ['values','type']
            temp_df.reset_index(inplace=True)
            
            inner_loop_df = inner_loop_df.append(temp_df)
            
        
        inner_loop_df['ticker'] = ticker
        
        inner_loop_df.dropna(axis=0,inplace=True)
    
        
    
        outer_loop_df = outer_loop_df.append(inner_loop_df)
        
        #print(outer_loop_df.loc[outer_loop_df['type'] == 'ema_atr','ATR','Rank'])

      
    outer_loop_df.set_index('Date',inplace=True)
    outer_loop_df.to_csv('art.csv')
   
        #df_1 = pd.DataFrame(loop_df['ema_atr'], index=loop_df.index)
        
        #df_2 = pd.DataFrame(loop_df['ema_art_percentile'], index=loop_df.index)

        #combined_df = pd.append([df_1,df_2],axis=1)
        
        #print(loop_df.stack(level=['ema_atr','ema_art_percentile']))


def momentum_test(back_test_df,look_back):

        back_test_df['one year ago price'] = back_test_df['value'].shift(look_back)
        
        back_test_df['signal'] = back_test_df['value']- back_test_df['one year ago price']
        
        back_test_df.dropna(inplace=True)
        back_test_df['weights'] = np.where(back_test_df['signal'] > 0, 1, -1)
        back_test_df['weights'] = back_test_df['weights'].shift(1)
        
        
        asset_return_df_1w= back_test_df['value'].pct_change(periods=1).dropna().to_frame()
        asset_return_df_1w.columns = ['weights']
    
        portfolio_returns = asset_return_df_1w * back_test_df['weights'].to_frame()
        
        portfolio_returns.dropna(inplace=True)
        portfolio_returns =  portfolio_returns.add(1).cumprod()
        
        portfolio_returns.plot(logy=True)
        
        return portfolio_returns
    

def single_backtest():
    
    data_df = pd.read_csv('quandl_data_formatted.csv',index_col=0)
    

    data_df.index = pd.to_datetime(data_df.index)

    
   
    ticker_info_df = pd.read_csv('futures_tickers.csv',index_col=0)
    
    ticker_info_df = ticker_info_df[ticker_info_df['include'] == True]
    
    
    data_df['short_name'] = data_df['ticker'].map(ticker_info_df['short_name_unique'])
    data_df_settle = data_df[data_df.type == 'Settle']
    
    #re)indexdata_df_settle = data_df_settle.re
    
    ticker_list = ticker_info_df['short_name_unique'].unique()
    
    performance_char = []
    performance_df = pd.DataFrame()
    index_labels = ['Start Data','End Date','Total Return','Annualized Return','1yr Return','5yr Return',
                            'Sharpe','Volatility','Max Drawdown',
                            'Average  Drawdown','Best Year','Worst Year','12m Win %']
   
    for ticker in range(0,1):
        
        
        ticker = ticker_list[ticker]
        temp_df = data_df_settle[data_df_settle.short_name == ticker]
        temp_df = temp_df.reindex(pd.date_range(temp_df.index[0],temp_df.index[-1],freq='W'),method='ffill')
      
        back_test_df = temp_df['value'].to_frame()
        
        index_total_rt_52 = momentum_test(back_test_df,52)
        index_total_rt_52 = momentum_test(back_test_df,26)
        index_total_rt_52 = momentum_test(back_test_df,20)
        index_total_rt_52 = momentum_test(back_test_df,12)
        index_total_rt_52 = momentum_test(back_test_df,1)

        
'''
        res = beta(portfolio_returns, name=ticker)
       
        total_return = res.stats.loc['total_return',:][0]
        one_year = res.stats.loc['one_year',:][0]
        five_year = res.stats.loc['five_year',:][0]
        
        max_drawdown = res.stats.loc['max_drawdown',:][0]
        avg_drawdown = res.stats.loc['avg_drawdown',:][0]
        monthly_sharpe = res.stats.loc['monthly_sharpe',:][0]
        volatility_m = res.stats.loc['monthly_vol',:][0]
        
        win_12m = res.stats.loc['twelve_month_win_perc',:][0]
        
        best_yr = res.stats.loc['best_year',:][0]
        worst_yr = res.stats.loc['worst_year',:][0]
        
        num_of_time =  len(res.prices) - 1
        ann_return = ((total_return + 1) ** (52/num_of_time)) - 1
        
        start_date = res.stats.loc['start',:][0]
        end_date = res.stats.loc['end',:][0]
     
        performance_char = [start_date,end_date,total_return,ann_return,one_year,five_year,
                            monthly_sharpe,volatility_m,max_drawdown,
                            avg_drawdown,best_yr,worst_yr,win_12m]
        

               
        temp_df = pd.DataFrame(performance_char,index=index_labels,columns=[ticker])
        
        performance_df = pd.concat([performance_df,temp_df],axis=1)
        res.display()
        
    #performance_df.to_csv('performance_stats')
    print(performance_df)
'''
      
def main():
    #download_data()
    #raw_data = get_data_csv()
    #atr()
    single_backtest()


def upload_dropbox():
    """
    Connect and authenticate with dropbox
    """
    app_key = 'k2sjusuvoxu48je'
    app_secret = 'jdehhyp54lmrdix'
    key = '4oW3DTXjnQsAAAAAAAAHNfQqyCb8ZPgFgoTsrNAplsygt_TonjYVZLGn0oKkWn6Z'
    dbx = dropbox.Dropbox(key)
    #'aGKqdaqoKwAAAAAAAAAAIovGNRj-sUb7BZtTsLX01r8y3UG0WV1aHax_4hKp2C6v'
    f = open('futures_tickers.csv', 'rb')

    #client.put_file('/futures_dashboard.csv', f)
    dbx.files_upload(f,'/test.csv',dropbox.files.WriteMode('overwrite', value=None))
'''
    trailing_1yr_return = trailing_return_score(raw_data)
    current = plot_bar(trailing_1yr_return.iloc[-1:])
    #drop_down_scatter(raw_data)
    new_df = pd.Series(trailing_1yr_return.iloc[-1:].values[0],index=trailing_1yr_return.iloc[-1:].columns.tolist())
    new_df = new_df.to_frame()
    new_df.columns = ['One Year Return']
    new_df.sort(columns=['One Year Return'],inplace=True,ascending=False)
    print(new_df)
    #table_values = pd.DataFrame(trailing_1yr_return.iloc[-1:].values,index=trailing_1yr_return.columns.tolist(),columns=['Future'])
    layout = go.Layout(yaxis=dict(tickformat = '%'))
    table = FF.create_table(new_df,index=True)
    new_df.to_csv('test.csv')
    py.iplot(table, filename='crude_table')
''' 

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
#print(long_short_portfolio)
#res = bt.run(long_short_portfolio)    
    
#res.plot()
#res.display()









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