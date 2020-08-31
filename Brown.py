# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 14:40:35 2020

@author: 64376
"""

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['simhei']#用于正常显示中文标签
plt.rcParams['axes.unicode_minus']=False#用于正常显示负号
import tushare as ts
import pandas as pd
import numpy as np

#获取股票数据
ts_code='600000'
data_=ts.get_hist_data(ts_code)#获得一天数据

data_=pd.DataFrame(data_[['close']])#获得其中的收盘价
data_.sort_index(inplace=True)#按照日期做升序处理
data_['ratio']=[0.0000000000000000000]*len(data_)
data_['date']=data_.index
data_.index=range(data_.shape[0])

#计算股价比率
for i in range(data_.shape[0]-1):
    data_['ratio'][i+1]=np.log(data_['close'][i+1]/data_['close'][i])

#计算基础参数
T=30#周期为30天（一个月）
k=3#第一个周期
data_train=data_.iloc[(k-1)*T:(k*T-1),:]#选取过去一月的时间作为训练集
data_test=data_.iloc[(k-1)*T:(k+1)*T-1,:]##选取过去第二月的时间作为测试集
mean=data_train['ratio'].sum()/T#计算样本均值
S_2=((data_train['ratio']-mean)**2).sum()/(T-1)#计算样本方差
miu=(mean+S_2/2)/1#股价漂移率
sigma=np.sqrt(S_2)#股价波动率

#通过几何布朗运动模型预测未来周期股价
S_0=list(data_train['close'])[-1]#初始股价
data_test['fore_price']=data_test['close']#预测的股价
for t in range(1,data_train.shape[0]+1):
    Bt = np.random.normal(0, 1,t)[0]#布朗运动随机数值
    data_test['fore_price'][t+k*T-1]=S_0*np.exp(sigma*Bt+(miu-S_2/2)*t)#预测股价
data_test['fore_price']

#画出预测值和真实值对比图
days = np.array(data_test['date'], dtype="datetime64[ms]")
plt.figure(num=k,figsize=(15,8))
plt.plot(days,data_test['fore_price'],label='预测股价')
plt.plot(days,data_test['close'],label='实际股价')
plt.xlabel('日期',fontsize=20)
plt.ylabel('股价',fontsize=20)
plt.title("股票“"+ts_code+"”"+str(T)+'天预测值和真实值对比图',fontsize=20)
plt.legend(loc='upper right')
plt.xticks(rotation=270)
