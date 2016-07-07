# -*- coding: utf-8 -*-
"""
Created on Fri May  6 22:17:54 2016

@author: ubuntu
"""
import bbgREST as bbg
from pandas_datareader import data as pdr

def data_api(source_str,q_spec,start_dt,end_dt):
    if source_str=='Bloomberg':
        bbg_type = q_spec['src_flds']['bbg_type']
        ticker = q_spec['src_flds']['bbg_ticker']
        field = q_spec['src_flds']['bbg_field']
        overrides = q_spec['src_flds']['overrides']
        if overrides is None:
            overrides=[]
        
        if bbg_type=='BDH':         
            data_df = bbg.get_histData([ticker],[field],start_dt,end_dt,overrides=overrides)
            data_df = data_df[['date',q_spec['src_flds']['bbg_field']]]
            data_df.rename(columns={'date':'date',field:'value'}, inplace=True)
            data_type = 'TS'            
            data = data_df
        
        elif bbg_type=='BDP':        
            data_df = bbg.get_refData([ticker],[field],overrides=overrides)
            print('BDP storage not yet implemented')
            data = None
            data_type = None
        
        elif q_spec['src_flds']['bbg_qtype']=='intradayTick':
            data_df = bbg.get_intradayTickData([ticker],start_dt,end_dt)
            print('intradayTick not yet implemented')
            data = None
            data_type = None
        
        elif q_spec['src_flds']['bbg_qtype']=='intradayBars':
            data_df = bbg.get_intradayData([ticker],start_dt,end_dt)
            print('intradayBars not yet implemented')
            data = None
            data_type = None
    
    elif source_str=='FRED':
        ticker = q_spec['src_flds']['fred_ticker']
        data_df = pdr.DataReader(ticker,'fred',start_dt,end_dt)
        data_df = data_df.reset_index().dropna()
        data_df.rename(columns={'DATE':'date',ticker:'value'}, inplace=True)
        data_type = 'TS'
        data = data_df
    return data_type, data