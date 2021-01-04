#%%

from oandapyV20 import API 
import oandapyV20.endpoints.instruments as instruments
import pandas as pd
import datetime
from dateutil import parser



account_id = '001-004-5534327-001'
access_token='ef35b35715762224cdc3b30fc0fbe3b2-77f270a7e174e687ad25279855e92a67'

client=API(access_token)
#%%
import time
def time_converter(unix_time):

    pattern = '%Y-%m-%d %H:%M:%S'
    epoch = int(time.mktime(time.strptime(unix_time, pattern)))
    return(epoch)

# %%

step=21600 # equals to 6h in UNIX_time.Depends on granulariy. 
# for 5s 6 hours is maximum granularity time.
# for 1m 21600*12 for 5m 21600*12*5.
granularity="S5"
begin_unix=time_converter("2018-01-01 00:00:00")
end_unix=time_converter("2018-01-07 00:00:00")

print(begin_unix, end_unix)
