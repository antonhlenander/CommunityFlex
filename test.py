from agents import DSO
from datamanager import DataManager
import numpy as np
import matplotlib.pyplot as plt

residual_demand = 25
daily_prices = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2]
var = 1

daily_residual_demand = [5,5,5,5,5]



def compute_capacity_limitation():
    max_price = np.max(daily_prices)
    min_price = np.min(daily_prices)
    mid_price = (max_price + min_price)/2
    avg_cap = residual_demand/24
    prices = np.array(daily_prices)
    capacity_limitation = avg_cap - var * avg_cap * 2 * (prices - mid_price) / (max_price - min_price)
    return capacity_limitation


daily_caps = np.divide(daily_residual_demand, residual_demand)

print(daily_caps)