#%%
import time
import pandas as pd
import datetime
from dateutil import parser
from oandapyV20 import API 
import oandapyV20.endpoints.instruments as instruments


account_id = '001-004-5534327-001'
access_token='ef35b35715762224cdc3b30fc0fbe3b2-77f270a7e174e687ad25279855e92a67'

client=API(access_token)


def getdata(begin_time, end_time, instrument, granularity ='M1',):

    '''
    Creates a .csv file for fx data. Colums are: time, open, high, low, close, volume.
    begin_time and end_time should be given in this format: %Y-%m-%d %H:%M:%S. 
    Granularity is set as 5s but can be changed. Step of 21600 to 6h in UNIX_time. Depends on granularity. 
    For 5s 6 hours is maximum granularity time. For 1m 21600*12 for 5m 21600*12*5.
    '''
    step=21600*12


  
    def time_converter(unix_time):

        pattern = '%Y-%m-%d %H:%M:%S'
        epoch = int(time.mktime(time.strptime(unix_time, pattern)))
        return(epoch)

    begin_unix = time_converter(begin_time)
    end_unix = time_converter(end_time)


    i=begin_unix+step
    dataset=pd.DataFrame()
    params={"from": str(i-step),
        "to": str(i),
        "granularity":granularity,
        "price":'A' } # 'A' stands for ask price; 
                      # if you want to get Bid use 'B' instead or 'AB' for both.
    while i<=end_unix:

        params['from']=str(i-step)
        params['to']=str(i)
        r=instruments.InstrumentsCandles(instrument, params=params)
        data = client.request(r)
        results= [{"time":x['time'],"open":float(x['ask']['o']),"high":float(x['ask']['h']),
              "low":float(x['ask']['l']),"close":float(x['ask']['c']),"volume":float(x['volume'])} for x in data['candles']]
    
        df = pd.DataFrame(results)
    
        if (dataset.empty): 
            dataset=df.copy()
    
        else: 
            dataset=dataset.append(df, ignore_index=True)
    
        if(i+step)>=end_unix:
            params['from']=str(i)
            params['to']=str(end_unix)
            r=instruments.InstrumentsCandles(instrument,params=params)
            data = client.request(r)
            results= [{"time":x['time'],"open":float(x['ask']['o']),"high":float(x['ask']['h']),
                  "low":float(x['ask']['l']),"close":float(x['ask']['c'])} for x in data['candles']]
            df = pd.DataFrame(results)
            i+=step
            dataset=dataset.append(df, ignore_index=True)
    
        if len(dataset)>2000000:
            dataset.to_csv(instrument+"_"+granularity+"_"+dataset['time'][0].split('T')[0]+"_"+dataset['time'][len(dataset)-1].split('T')[0]+'.csv',index=False)
            dataset=pd.DataFrame()
        i+=step
    dataset.to_csv(instrument+"_"+granularity+"_"+dataset['time'][0].split('T')[0]+"_"+dataset['time'][len(dataset)-1].split('T')[0]+'.csv',index=False)

# %%
getdata("2018-01-01 00:00:00", "2019-01-01 00:00:00", "GBP_USD")
# %%
