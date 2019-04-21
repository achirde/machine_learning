import sklearn as sk
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import os

def measure_accuracy(y_true, y_pred):
    mse = np.array([])

    for i in range(len(y_true)):
        diff = ((y_true['close'][i] - y_pred['close'][i])/y_true['close'][i])**2
        mse = np.append(mse, diff)
    return mse.mean()


data = pd.read_csv('files/train_timeseries2.csv', index_col=0)
data_test = pd.read_csv('files/test_timeseries2.csv', index_col=0)


# print(data.head())

X_data = data.loc[:, data.columns != 'close']
y_data = data.loc[:, data.columns == 'close'] 

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)

# print(len(X_train), len(X_test))

reg = RandomForestRegressor(n_estimators=100)

reg.fit(X_train, y_train.values.ravel())
y_pred = reg.predict(X_test)
y_pred = y_pred.round(decimals=2)
score = reg.score(X_test, y_test)

y_pred_df = pd.DataFrame(y_pred, columns=['close'], index=y_test.index)

y_sub = reg.predict(data_test)
y_sub = y_sub.round(decimals=2)
y_sub_df = pd.DataFrame(y_sub, columns=['close'], index=data_test.index)

score = mean_squared_error(y_test, y_pred)
score = measure_accuracy(y_test, y_pred_df)


y_pred_df.to_csv('./files/y_predicted.csv')
y_test.to_csv('./files/y_test.csv')
y_sub_df.to_csv('./files/test_submission2.csv')
print(score)

