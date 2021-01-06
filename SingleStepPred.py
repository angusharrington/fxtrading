#%%
import time
import pandas as pd
import datetime
from dateutil import parser
from oandapyV20 import API 
import oandapyV20.endpoints.instruments as instruments
import matplotlib.pyplot as plt


df = pd.read_csv(r'GBP_USD_M1_2018-01-01_2018-12-31.csv')
df = pd.DataFrame(df, columns=['time','open','high','low','close','volume'])

high = df.high.tolist()
low = df.low.tolist()

HLavg = [(high[i]+low[i])/2 for i in range(len(high))]
times = df.time.tolist()
print(times[:5])

# %%
def centered_moving_avg(odd_window_size, HLavg, times):
    mid_idx = int(odd_window_size/2)
    j = 0
    moving_avg = []
    while odd_window_size+j-1 != len(HLavg):
        point = 0
        for i in range(odd_window_size):
            point += HLavg[i+j]

        moving_avg.append(point/odd_window_size)
        j += 1

    times = times[mid_idx: -mid_idx]
    return moving_avg, times

mov_avg, mov_times = centered_moving_avg(7, HLavg, times)

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping



# %%
maximum, minimum = max(mov_avg), min(mov_avg)

scaled_prices = np.array([((t - minimum)/(maximum-minimum)) for t in mov_avg])

# %%
plt.plot(3)
plt.clf()
plt.plot(mov_times[:1000], scaled_prices[:1000], label='GBP/USD HL avg. ($)')
plt.legend()
plt.show()
# %%
batch_size = 32
window_size = int(256) # must be a multiple of batch_size
validation_size = 8192 * batch_size # must be a multiple of batch_size
test_size = 8192 * batch_size # must be a multiple of batch_size

import numpy as np

def get_train(values, window_size):
    X, y = [], []
    len_values = len(values)
    for i in range(window_size, len_values):
        X.append(values[i-window_size:i])
        y.append(values[i])
    X, y = np.asarray(X), np.asarray(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    print(f"X {X.shape}, y {y.shape}")
    return X, y

def get_val(values, window_size):
    X = []
    len_values = len(values)
    for i in range(window_size, len_values):
        X.append(values[i-window_size:i])
    X = np.asarray(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    y = values[-X.shape[0]:]
    print(f"X {X.shape}, y {y.shape}")
    return X, y

train = scaled_prices
test = scaled_prices
train = train[:int(0.8*len(scaled_prices))]
test = scaled_prices[int(0.8*len(scaled_prices)):]
# %%
X, y = get_train(train, window_size)
X_val, y_val = get_val(test, window_size)

# %%
import pandas as pd
from collections import OrderedDict
import numpy as np
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential(
    [
    layers.LSTM(76, input_shape=(X.shape[1], 1), return_sequences = False),
    layers.Dropout(0.2),
	layers.Dense(1)
    ]
)

print(model.summary())
# %%
model.compile(loss="mse", optimizer="adam", metrics=["categorical_accuracy"])
model.fit(X, y, validation_data=(X_val, y_val), batch_size=batch_size, epochs=20, shuffle=False, verbose=2)


# %%
