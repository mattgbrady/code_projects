# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 11:35:52 2016

@author: ABerner
"""

import pandas as pd
import numpy as np
import bbgREST as bbg
import psycopg2 as pg
from psycopg2.extensions import register_adapter, AsIs

def adapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)
register_adapter(np.int64, adapt_numpy_int64)

import ast

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

psql_creds = {
    'host': 'localhost',
    'port': 5433,    
    'user': 'postgres',
    'dbname': 'postgres'
}

ticker_file = '../SQL/ticker_input.csv'
factor_file = '../SQL/factor_input.csv'

def update_data(creds,tckr_file):
    conn = pg.connect(**creds)
    curs = conn.cursor()
    
    freq_dict = {
        'D':'DAILY',
        'W':'WEEKLY',
        'M':'MONTHLY',
        'Y':'YEARLY'
    }
    
    tickers = pd.read_csv(tckr_file,parse_dates=[3,4])
    tickers['dt_end'] = tickers['dt_end'].map({'Inf':pd.datetime.today().date()-pd.offsets.BDay(1)})
    tickers['freq'] = tickers['freq'].map(freq_dict)
    
    for row_ind, row in tickers.iterrows():
        print(row['Ticker'],row['Field'])
        if row['Source']=='BDH':
            if type(row['Field'])!=str:
                print('Field not specified for '+row['Ticker'])
                continue
            else:
                try:
                    tmpdf = bbg.get_histData([row['Ticker']],[row['Field']],row['dt_start'],row['dt_end'],freq=row['freq'])
                    if tmpdf.empty:
                        print('Request returns no data for '+row['Ticker']+', '+row['Field'])
                        continue
                    else:
                        curs.execute("""
                            INSERT INTO data_sources (source) VALUES (%s) ON CONFLICT (source) DO NOTHING;
                            """, [row['Source']])
                        curs.execute("""
                            SELECT id FROM data_sources WHERE source=%s;                    
                            """, [row['Source']])
                        src_id = curs.fetchone()[0]
                        
                        curs.execute("""
                            INSERT INTO tickers (ticker) VALUES (%s) ON CONFLICT (ticker) DO NOTHING;
                            """, [row['Ticker']])
                        curs.execute("""
                            SELECT id FROM tickers WHERE ticker=%s;                    
                            """, [row['Ticker']])
                        tckr_id = curs.fetchone()[0]                                        
                        
                        curs.execute("""
                            INSERT INTO fields (ticker_id, field) VALUES (%s, %s) ON CONFLICT (ticker_id, field) DO NOTHING;
                            """, [tckr_id, row['Field']])
                        curs.execute("""
                            SELECT id FROM fields WHERE ticker_id=%s AND field=%s;
                            """,[tckr_id, row['Field']])
                        fld_id = curs.fetchone()[0]
                        
                        tmpdts = pd.DatetimeIndex(tmpdf['date']).to_pydatetime()
                        tmpvals = tmpdf[row['Field']].values
                        n = len(tmpvals)
                        record_list_template = ','.join(['%s']*n)
                        args = list(zip([fld_id]*n,[src_id]*n,tmpvals,tmpdts))                    
                        
                        if is_number(str(tmpvals[0])):
                            curs.execute("""
                                INSERT INTO ticker_values (field_id, source_id, mkt_number, mkt_dt) 
                                  VALUES {0} ON CONFLICT (field_id, mkt_number, mkt_dt) DO NOTHING;
                                  """.format(record_list_template),args)
                        elif type(tmpvals[0])==str:
                             curs.execute("""
                                INSERT INTO ticker_values (field_id, source_id, mkt_text, mkt_dt) 
                                  VALUES {0} ON CONFLICT (field_id, mkt_text, mkt_dt) DO NOTHING;
                                  """.format(record_list_template),args)
                        else:
                            print("Error inserting; bad or no data")
                                         
                except AttributeError as e:
                    print(e)
                    continue
        else:
            print('Only BDH currently implemented.')
            continue
        
    curs.close()
    conn.commit()
    conn.close()
    return
    
    
def update_factors(creds, fctr_file):
    conn = pg.connect(**creds)
    curs = conn.cursor()
    
    factors = pd.read_csv(fctr_file)
    fskip = ''
    
    for row_ind, row in factors.iterrows():
        if row['factor_name']==fskip:
            print("Skipping ",row['factor_name'])
            continue
        elif row['method']==np.nan:
            print('No method specified for ',row['factor_name'],', skipping factor')
            fskip=row['factor_name']
            continue
        
        print(row['factor_name'],row['dependent_factor'],row['dependent_ticker'],row['method'])
        
        curs.execute("""
            INSERT INTO factors (name, method) VALUES (%s,%s) ON CONFLICT (name, method) DO NOTHING;
            """, [row['factor_name'],row['method']])
        curs.execute("""
            SELECT id FROM factors WHERE name=%s AND method=%s;                    
            """, [row['factor_name'],row['method']])
        fctr_id = curs.fetchone()[0]        
        
        if type(row['dependent_factor'])==float:

            curs.execute("""
            SELECT id FROM tickers WHERE ticker=%s;                    
            """, [row['dependent_ticker']])
            tckr_id = curs.fetchone()

            if tckr_id is None:
                print("ticker ",row['dependent_ticker']," is not in the db, skipping factor")
                fskip = row['factor_name']
                continue

            curs.execute("""
            SELECT id FROM fields WHERE ticker_id=%s AND field=%s;                    
            """, [tckr_id,row['dependent_field']])
            fld_id = curs.fetchone()
            
            if fld_id is None:
                print("field ",row['dependent_ticker'],", ",row['dependent_field']," is not in the db, skipping factor")
                fskip = row['factor_name']
                continue
            
            mthd_prm_ordr = row['param_order']
            if np.isnan(mthd_prm_ordr):
                mthd_prm_ordr = None
            curs.execute("""
                INSERT INTO factor_dependents (factor_id, field_id, method_param_order) VALUES (%s,%s,%s) ON CONFLICT (factor_id, field_id) DO NOTHING;
                """, [fctr_id, fld_id, mthd_prm_ordr])            

                        
        else:
            
            curs.execute("""
               SELECT id FROM factors WHERE name=%s;                    
               """, [row['dependent_factor']])
            chld_fctr_id = curs.fetchone()

            if chld_fctr_id is None:
                print(row['dependent_factor']," is not in the db, skipping factor")
                fskip = row['factor_name']
                continue 
            
            mthd_prm_ordr = row['param_order']
            if np.isnan(mthd_prm_ordr):
                mthd_prm_ordr = None
            curs.execute("""
                INSERT INTO factor_dependents (factor_id, child_factor_id, method_param_order) VALUES (%s,%s,%s) 
                ON CONFLICT (factor_id, child_factor_id) DO NOTHING;
                """, [fctr_id, chld_fctr_id, mthd_prm_ordr])            
            
        if type(row['attributes'])!=float:
            attribute_dict = ast.literal_eval(row['attributes'])
            for key, value in attribute_dict.items():
                curs.execute("""
                    INSERT INTO factor_method_attributes (factor_id, attribute_key, attribute_value) VALUES (%s,%s,%s) 
                    ON CONFLICT (factor_id, attribute_key, attribute_value) DO NOTHING;
                    """, [fctr_id, key, value])
        
    curs.close()
    conn.commit()
    conn.close()
    return
    
                
            
            

update_data(psql_creds,ticker_file)

#update_factors(psql_creds,factor_file)

#query = 
                