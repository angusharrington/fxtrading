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