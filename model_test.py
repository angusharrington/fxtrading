#%%
print('hello')

#%%
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
#%%

d1f = pd.read_csv('GBP_USD_M1_2019-01-01_2019-01-10.csv')
d1f = pd.DataFrame(d1f, columns=['time','open','high','low','close','volume'])

high = d1f.high.tolist()
low = d1f.low.tolist()

HLavg1 = [(high[i]+low[i])/2 for i in range(len(high))]
times1 = d1f.time.tolist()

mov_avg1, mov_times1 = centered_moving_avg(7, HLavg1, times1)
maximum1, minimum1 = max(mov_avg1), min(mov_avg1)

scaled_prices1 = np.array([((t - minimum1)/(maximum1-minimum1)) for t in mov_avg1])
window_size = int(256)
X_val1, y_val1 = get_val(scaled_prices1, window_size)

#%%
from tensorflow.keras.models import load_model

y_pred = []
model = load_model('model_lstm_singlesteppred')

model.evaluate(X_val1, y_val1, verbose =1)
# %%
y_pred = model.predict(X_val1)



# %%
plt.plot(1)
plt.plot(y_pred[0:250], label = 'pred')
plt.plot(X_val1[1], label = 'real')

plt.legend()
plt.show()
# %%
print(y_pred[:100])
print(y_val1[:100])
# %%
