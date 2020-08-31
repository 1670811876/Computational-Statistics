# -*- coding: utf-8 -*-
"""
Created on Sun 6月7日 14:40:35 2020 
@author: Alvin
""" 
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['simhei']  
plt.rcParams['axes.unicode_minus'] = False 
import tushare as ts
import pandas as pd
import numpy as np 
ts_code = '600000'
data_ = ts.get_hist_data(ts_code)  # acquire stock data
data_ = pd.DataFrame(data_[['close']])  # acquire closing price
data_.sort_index(inplace=True)  
data_['ratio'] = [0.0000000000000000000] * len(data_)
data_['date'] = data_.index
data_.index = range(data_.shape[0])  # calculate the share price ratio
for i in range(data_.shape[0] - 1): 
    data_['ratio'][i+1] = np.log(data_['close'][i+1] / data_['close'][i])  # calculate the elemental parameters
T = 30  # the cycle is 30 days (one month)
k = 3  # the first circle
data_train = data_.iloc[(k-1)*T:(k*T - 1), :]  # select the past month as the training set
data_test = data_.iloc[(k-1)*T:(k+1)*T - 1, :]  # select the past February as the test set
mean = data_train['ratio'].sum() / T  # calculate the sample mean
S_2 = ((data_train['ratio'] - mean)**2).sum()/(T-1)  # calculate the sample variance
miu = (mean + S_2 / 2) / 1  # stock price drift rate
sigma = np.sqrt(S_2)  # stock price volatility

# use geometric Brownian motion model to predict the future period stock price
S_0 = list(data_train['close'])[-1]  # initial stock price
data_test['fore_price'] = data_test['close']  # predicted stock price
for t in range(1, data_train.shape[0] + 1): 
    Bt = np.random.normal(0, 1, t)[0]  # stochastic value of Brownian motion
    data_test['fore_price'][t + k * T - 1] = S_0 * np.exp(sigma * Bt + (miu - S_2 / 2) * t)  # predict stock price

days = np.array(data_test['date'], dtype="datetime64[ms]")
plt.figure(num=k, figsize=(15,8))
plt.plot(days,data_test['fore_price'], label='predicted stock price')
plt.plot(days,data_test['close'], label='actual stock price')
plt.xlabel('date', fontsize=20)
plt.ylabel('price', fontsize=20)
plt.legend(loc='upper right')
plt.xticks(rotation=270)
