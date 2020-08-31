# -*- coding: utf-8 -*-
"""
Created on Sun 6月7日 19:46:13 2020 
@author: Alvin
"""
import tushare as ts
import warnings
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
from tqdm import tqdm 

ts_code = '600000'
data_ = ts.get_hist_data(ts_code)
data_['date'] = data_.index  # Supress warning in hmmlearn
warnings.filterwarnings("ignore")
plt.style.use('ggplot') 
class StockPredictor(object):
    def __init__(self, data, test_size=0.33, n_hidden_states=4, n_latency_days=10, n_steps_frac_change=50, n_steps_frac_high=10, n_steps_frac_low=10): 
        self.data = data 
        self.n_latency_days = n_latency_days 
        self.hmm = GaussianHMM(n_components=n_hidden_states) 
        self._split_train_test_data(test_size) 
        self._compute_all_possible_outcomes( n_steps_frac_change, n_steps_frac_high, n_steps_frac_low) 
        
    def _split_train_test_data(self, test_size): 
        data = self.data _train_data, 
        test_data = train_test_split( data, test_size=test_size, shuffle=False) 
        self._train_data = _train_data self._test_data = test_data 
    
    @staticmethod 
    def _extract_features(data): 
        open_price = np.array(data['open']) 
        close_price = np.array(data['close']) 
        high_price = np.array(data['high']) 
        low_price = np.array(data['low']) 
        frac_change = (close_price - open_price) / open_price 
        frac_high = (high_price - open_price) / open_price 
        frac_low = (open_price - low_price) / open_price 
        return np.column_stack((frac_change, frac_high, frac_low)) 
    
    def fit(self): 
        feature_vector = StockPredictor._extract_features(self._train_data) 
        self.hmm.fit(feature_vector) 
        
    def _compute_all_possible_outcomes(self, n_steps_frac_change, n_steps_frac_high, n_steps_frac_low): 
        frac_change_range = np.linspace(-0.1, 0.1, n_steps_frac_change) 
        frac_high_range = np.linspace(0, 0.1, n_steps_frac_high) 
        frac_low_range = np.linspace(0, 0.1, n_steps_frac_low) 
        self._possible_outcomes = np.array(list(itertools.product( frac_change_range, frac_high_range, frac_low_range))) 
        
    def _get_most_probable_outcome(self, day_index): 
        previous_data_start_index = max(0, day_index - self.n_latency_days) 
        previous_data_end_index = max(0, day_index - 1) 
        previous_data = self._test_data.iloc[previous_data_end_index: previous_data_start_index] 
        previous_data_features = StockPredictor._extract_features( previous_data) 
        outcome_score = [] 
        for possible_outcome in self._possible_outcomes: 
            total_data = np.row_stack((previous_data_features, possible_outcome)) 
            outcome_score.append(self.hmm.score(total_data)) 
            most_probable_outcome = self._possible_outcomes[np.argmax( outcome_score)] 
        return most_probable_outcome 
    
    def predict_close_price(self, day_index): 
        open_price = self._test_data.iloc[day_index]['open'] 
        predicted_frac_change, _, _ = self._get_most_probable_outcome( day_index) return open_price * (1 + predicted_frac_change) 
    
    def predict_close_prices_for_days(self, days, with_plot=False): 
        predicted_close_prices = [] 
        for day_index in tqdm(range(days)): 
            predicted_close_prices.append(self.predict_close_price(day_index)) 
            if with_plot: 
                test_data = self._test_data[0: days] 
                days = np.array(test_data['date'], dtype="datetime64[ms]") 
                actual_close_prices = test_data['close'] 
                fig = plt.figure() plt.plot(days, actual_close_prices, '-', label="actual") 
                plt.plot(days, predicted_close_prices, '-', label="predicted") 
                fig.autofmt_xdate() 
                plt.legend() 
                plt.show() 
        return predicted_close_prices stock_predictor = StockPredictor(data=data_)
    
  
stock_predictor.fit()
stock_predictor.predict_close_prices_for_days(200, with_plot=True)
