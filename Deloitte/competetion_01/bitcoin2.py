import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Calculate Accuracy
def measure_accuracy(y_true, y_pred):
    mse = np.array([])

    for i in range(len(y_true)):
        diff = ((y_true['close'][i] - y_pred['close'][i])/y_true['close'][i])
        mse = np.append(mse, diff)
    return mse.mean()

data = pd.read_csv('files/train_timeseries2.csv', index_col=0)
data_sub = pd.read_csv('files/test_timeseries2.csv', index_col=0)

# print(data.isna().sum())

data_train = data.sample(frac=0.8, random_state=0)
data_test = data.drop(data_train.index)
# print(len(data_train), len(data_test))

train_stat = data_train.describe()
train_stat.pop('close')
train_stat = train_stat.transpose()
# print(train_stat)

y_train = data_train.pop("close")
y_test  = data_test.pop("close")

# Normalize Data
def norm(x):
    return (x-train_stat['mean'])/train_stat['std']

normed_X_train = norm(data_train)
normed_X_test  = norm(data_test)

reg = RandomForestRegressor(n_estimators=100)
reg.fit(normed_X_train, y_train)

y_pred = reg.predict(normed_X_test)
y_pred = y_pred.round(decimals=2)
score = reg.score(normed_X_test, y_test)

y_pred_df = pd.DataFrame(y_pred, columns=['close'], index=y_test.index)

normalized_sub = norm(data_sub)
y_sub = reg.predict(normalized_sub)
y_sub = y_sub.round(decimals=2)
y_sub_df = pd.DataFrame(y_sub, columns=['close'], index=data_sub.index)

y_test_df = pd.DataFrame(y_test, columns=['close'], index=y_test.index)
myscore = measure_accuracy(y_test_df, y_pred_df)
# print(y_pred_df.head())


y_pred_df.to_csv('./files/y_predicted_2.csv')
y_test_df.to_csv('./files/y_test_2.csv')
y_sub_df.to_csv('./files/test_submission2_2.csv')


print(score, '\n', myscore)

