# -*- coding: utf-8 -*-
#!/usr/bin/env python2

# usage: python HistoricalDataRequest.py <host-ip>
from __future__ import print_function

import json
try:
    import urllib2 as urllib
except ImportError:
    import urllib.request as urllib

import ast
import pandas as pd
import numpy as np


#host = 'dellt17003221.corp.wurts.com:4567'
host = '192.168.67.134:4567'
#host = '169.254.48.234'

'''
#Historical Data Example: Download daily OHLC for three currency pairs
histData = {
    "securities": ["AUD Curncy", "NZD Curncy", "CAD Curncy"],
    "fields": ["OPEN", "PX_HIGH", "PX_LOW", "PX_LAST"],
    "startDate": "20130101",
    "endDate": "20150721",
    "periodicitySelection": "DAILY"
}

#Reference Data Example: Download index constituents as of specific date
refData = {
    "securities": ["SPX Index","S5UTIL Index"],
    "fields": ["DS184"],
    "overrides": [{"fieldId":"END_DATE_OVERRIDE","value":"20150105"}]
}

#Intraday Data Example: Download E-Mini intraday. Note that BBG only maintains a 
#140-day rolling history for intraday data
intradayData = {
    "security": "ES1 Index",
    "eventType": "TRADE", 
    "interval": 5,     
    "startDateTime": pd.datetime(2015,5,21,0,0).isoformat(),
    "endDateTime": pd.datetime(2015,7,24,13,0).isoformat()
}

#Intraday TickData Example: Download E-Mini intraday. Note that BBG only maintains a 
#140-day rolling history for intraday data
intradayTickData = {
    "security": "ES1 Index",
    "eventTypes": ["TRADE"],  
    "startDateTime": pd.datetime(2015,5,21,0,0).isoformat(),
    "endDateTime": pd.datetime(2015,7,24,13,0).isoformat()
}
'''

def histDataReq(host,data):
    data = json.dumps(data)
    bin_data = data.encode('utf8')
    clen = len(bin_data)
    req = urllib.Request('http://{}/request?ns=blp&service=refdata&type=HistoricalDataRequest'.format(host),
                         bin_data,{'Content-Type': 'application/json', 'Content-Length': clen})

    try:
        res = urllib.urlopen(req)
    except Exception as e:
        e
        print(e)
        return 1
    return res

    
def refDataReq(host,data):
    data = json.dumps(data)
    bin_data = data.encode('utf8')
    clen = len(bin_data)
    req = urllib.Request('http://{}/request?ns=blp&service=refdata&type=ReferenceDataRequest'.format(host),
                         bin_data,{'Content-Type': 'application/json', 'Content-Length': clen})

    try:
        res = urllib.urlopen(req)
    except Exception as e:
        e
        print(e)
        return 1
    return res


def intradayDataReq(host,data,interval=5):
    interval = max([min([1024,int(interval)]),1])    
    data['interval']=interval
    
    data = json.dumps(data)
    bin_data = data.encode('utf8')
    clen = len(bin_data)
    req = urllib.Request('http://{}/request?ns=blp&service=refdata&type=IntradayBarRequest'.format(host),
                         bin_data,{'Content-Type': 'application/json', 'Content-Length': clen})
    
    try:
        res = urllib.urlopen(req)
    except Exception as e:
        e
        print(e)
        return 1
    return res
    
    
def intradayTickDataReq(host,data):  
    data = json.dumps(data)
    bin_data = data.encode('utf8')
    clen = len(bin_data)
    req = urllib.Request('http://{}/request?ns=blp&service=refdata&type=IntradayTickRequest'.format(host),
                         bin_data,{'Content-Type': 'application/json', 'Content-Length': clen})
    
    try:
        res = urllib.urlopen(req)
    except Exception as e:
        e
        print(e)
        return 1
    return res


def parse_histDataReq(res):
    dataDict = ast.literal_eval(res.read().decode('utf8').replace(":true",":True"))
    outdf = pd.DataFrame()    
    tempdf = pd.DataFrame()
    for item in dataDict['data']:
        try:
            tempdf = pd.DataFrame(data = item['securityData']['fieldData'])
            tempdf['Ticker'] = item['securityData']['security']
            outdf = pd.concat([outdf,tempdf]).reset_index(drop=True)
            outdf['date'] = pd.to_datetime(outdf['date'])
        except Exception as e:
            e
            print(e)
            break
    return outdf
    
    
def parse_refDataReq(res):
    dataDict = ast.literal_eval(res.read().decode('utf8').replace(":true",":True"))
    outdf = pd.DataFrame()    
    tempdf = pd.DataFrame()
    for item in dataDict['data']:
        try:
            for field in item['securityData'][0]['fieldData']:
                if type(field)==dict:
                    tempdf = pd.DataFrame(data = item['securityData'][0]['fieldData'][field])
                else:
                    tempdf = pd.DataFrame(data = item['securityData'][0]['fieldData'], index=[0])
                tempdf['Ticker'] = item['securityData'][0]['security']
                outdf = pd.concat([outdf,tempdf]).reset_index(drop=True)
        except Exception as e:
            e
            print(e)
            break
    return outdf

    
def parse_intradayDataReq(res):
    dataDict = ast.literal_eval(res.read().decode('utf8').replace(":true",":True"))
    outdf = pd.DataFrame()
    try:
        outdf = pd.DataFrame(data = dataDict['data'][0]['barData']['barTickData'])
    except Exception as e:
        e
        print(e)
    return outdf


def parse_intradayTickDataReq(res):
    dataDict = ast.literal_eval(res.read().decode('utf8').replace(":true",":True"))
    outdf = pd.DataFrame()
    try:
        outdf = pd.DataFrame(data = dataDict['data'][0]['tickData']['tickData'])
    except Exception as e:
        e
        print(e)
    return outdf
    

def get_intradayTickData(tickers,startTime,endTime,events=['TRADE'],hrblk=1):
    df = pd.DataFrame()
    for ticker in tickers:
        tmpdf = pd.DataFrame()
        tmpdf1 = pd.DataFrame()
        tmpstartTime = startTime.isoformat()
        tmpendTime = (np.min([startTime+pd.datetools.offsets.Hour(hrblk),endTime])).isoformat()        
        #endTime = endTime.isoformat()
        print(ticker)
        loop = True
        while loop:
            intradayTickData = {"security": ticker,
                            "eventTypes": events,  
                            "startDateTime": tmpstartTime,#pd.datetime(2015,5,21,0,0).isoformat(),
                            "endDateTime": tmpendTime}#pd.datetime(2015,7,24,13,0).isoformat()
            res = intradayTickDataReq(host,intradayTickData)
            tmpdf1 = parse_intradayTickDataReq(res)
            if (not tmpdf1.empty):
                tmpdf1['time'] = pd.to_datetime(tmpdf1['time'])
                tmpdf = pd.concat([tmpdf,tmpdf1]).reset_index(drop=True)
                tmptime = pd.to_datetime(tmpdf1['time'].values[-1])
            else:
                tmpdf1=pd.DataFrame(data=pd.to_datetime(tmpstartTime)+pd.datetools.offsets.Hour(hrblk),columns=['time'],index=[0])
                tmptime = pd.to_datetime(tmpdf1['time'].values[-1])
            if ((tmptime<pd.to_datetime(endTime)) and (tmptime!=pd.to_datetime(tmpstartTime))):
                tmpendTime=(np.min([endTime,tmptime+pd.datetools.offsets.Hour(hrblk)])).isoformat()
                tmpstartTime=(pd.to_datetime(tmpdf1['time'].values[-1])+pd.datetools.offsets.Second()).isoformat()
            else:
                tmpdf['ticker'] = ticker
                df = pd.concat([df,tmpdf])              
                break
            print(tmpstartTime)
    return df    
    
    
def get_intradayData(tickers,startTime,endTime,event='TRADE',interval=5):
    df = pd.DataFrame()
    for ticker in tickers:
        tmpdf = pd.DataFrame()
        tmpdf1 = pd.DataFrame()
        tmpstartTime = startTime.isoformat()
        endTime = endTime.isoformat()
        print(ticker)
        for i in range(58):
            intradayData = {"security": ticker,
                            "eventType": event, 
                            "interval": interval,     
                            "startDateTime": tmpstartTime,#pd.datetime(2015,5,21,0,0).isoformat(),
                            "endDateTime": endTime}#pd.datetime(2015,7,24,13,0).isoformat()
            res = intradayDataReq(host,intradayData,intradayData['interval'])
            tmpdf1 = parse_intradayDataReq(res)
            tmpdf1['time'] = pd.to_datetime(tmpdf1['time'])
            tmpdf = pd.concat([tmpdf,tmpdf1]).reset_index(drop=True)
            tmptime = pd.to_datetime(tmpdf1['time'].values[-1])
            if ((tmptime<pd.to_datetime(endTime)) and (tmptime!=pd.to_datetime(tmpstartTime))):
                tmpstartTime=pd.to_datetime(tmpdf['time'].values[-1]).isoformat()
            else:
                tmpdf = tmpdf.drop_duplicates()
                tmpdf['ticker'] = ticker
                df = pd.concat([df,tmpdf])
                break
            print(i)
    return df


def get_histData(tickers,fields,startTime,endTime,freq='DAILY',overrides=[]):
    df = pd.DataFrame()
    histData = {"securities": tickers,
                "fields": fields,
                "startDate": startTime.strftime("%Y%m%d"),
                "endDate": endTime.strftime("%Y%m%d"),
                "periodicitySelection": freq,
                "overrides": overrides}

    res = histDataReq(host,histData)
    print(res)
    df = parse_histDataReq(res)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df
    
    
def get_refData(tickers,fields,overrides=[]):
    df = pd.DataFrame()
    refData = {"securities": tickers,
               "fields": fields,
                "overrides": []}
    for item in overrides:
        refData['overrides'].append({'fieldId':item[0],'value':item[1]})
    res = refDataReq(host,refData)
    df = parse_refDataReq(res)
    return df



        
        
