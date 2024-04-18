# https://www.kaggle.com/datasets/rakannimer/air-passengers?resource=download

# https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/

import matplotlib.pyplot as plt

import pandas as pd

df= pd.read_csv('AirPassengers.csv')

timeseries= df[["#Passengers"]].values.astype('float32')

plt.plot(timeseries)
plt.show()

train_size= int(len(timeseries)*0.67)
test_size= len(timeseries) -train_size
train, test= timeseries[:train_size],timeseries[train_size:]



