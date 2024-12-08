from agents import DSO
from datamanager import DataManager
import numpy as np
import matplotlib.pyplot as plt

residual_demand = 25
daily_prices = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2]
var = 1



def compute_capacity_limitation():
    max_price = np.max(daily_prices)
    min_price = np.min(daily_prices)
    mid_price = (max_price + min_price)/2
    avg_cap = residual_demand/24
    prices = np.array(daily_prices)
    capacity_limitation = avg_cap - var * avg_cap * 2 * (prices - mid_price) / (max_price - min_price)
    return capacity_limitation


print(compute_capacity_limitation())