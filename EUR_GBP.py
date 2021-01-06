#%%
import pandas as pd
from collections import OrderedDict
import numpy as np
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
# %%

column_names = ["DateTime", "Price", 'x', 'x1', 'x2', 'x3']
data = pd.read_csv('GBP_USD_M1_2019-01-01_2019-01-10.csv', names=column_names)
price = data.Price.tolist()

slices = []
preds = []
i = 0
j = 0
sliced = []
while j < len(price) - 100:

    sliced.append(price[j])
    i += 1
    j += 5
    if i == 9:
        norm_sliced = np.array([((t - min(sliced))/(max(sliced)-min(sliced))) for t in sliced])

        if (price[j +5] - min(sliced))/(max(sliced)-min(sliced)) > sliced[-1]:
            preds.append(1)
        else:
            preds.append(0)

        slices.append(norm_sliced)
        j = j - 44
        i = 0
        sliced = []
        norm_sliced = []

x_train = np.array(slices[:int(0.7*len(preds))])
y_train = np.array(preds[:int(0.7*len(preds))])

x_test = np.array(slices[int(0.7*len(preds)):])
y_test = np.array(preds[int(0.7*len(preds)):])

input_shape = x_train[0].shape
#%%
print(y_train[0])
#%%
model = keras.Sequential(
    [
    keras.Input(shape=input_shape),
	layers.Dense(60, input_dim=9, activation='relu'),
	layers.Dense(30,  activation='relu'),
    layers.Dropout(0.5),
	layers.Dense(2, activation="softmax"),
    ]
)

batch_size = 1
epochs = 10

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)






# %%
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# %%
score = model.evaluate(np.array([[0.2, 0, 0.2, 0.3, 0.5, 0.9, 1.0, 0.9, 0.5]]), np.array([1]), verbose=0)
pred = model.predict(np.array([[0.2, 0, 0.2, 0.3, 0.2, 0.9, 1.0, 0.9, 0.5]]))

# %%
print(pred)
# %%
print(y_test[0])
# %%
