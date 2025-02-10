from agents import DSO
from datamanager import DataManager
import numpy as np
import matplotlib.pyplot as plt

residual_demand = 25
daily_prices = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2]
var = 0.8

daily_residual_demand = [5,5,5,5,5]

dm = DataManager()

agent_list = ["H1", "H2", "H3", "H4", "H5"]

demand_df = dm.demand_df

max_demand = 0

# print(demand_df)

print(dm.get_all_maxdemand())


# for i in range(24):
#     hour_demand = 0
#     for agent in agent_list:
#         hour_demand += demand_df[agent].iloc[i]
#     print(hour_demand)

#     max_demand = max(max_demand, hour_demand)

# print("max demand:", max_demand)