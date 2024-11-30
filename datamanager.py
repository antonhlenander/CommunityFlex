import prepros as pp
import pandas as pd
import numpy as np

subfolder = 'data/fullyearPV_singleDemand/'
demand_path = f'{subfolder}demandprofiles.csv'
prod_path = f'{subfolder}PV.csv'
price_path = f'{subfolder}price_data.csv'

class DataManager:
    def __init__(self, demand_path=demand_path, prod_path=prod_path, price_path=price_path):
        self.demand_df: pd.DataFrame = pp.load_demand_profile(demand_path)
        print("Demand profiles loaded!")
        self.prod_df = pp.load_production_data(prod_path)
        print("Production profiles loaded!")
        # self.price_df: pd.DataFrame = pp.load_price_data(price_path)

    def get_agent_demand(self, aid, step, noise_std=0.1):
        noise = 0
        base_demand = self.demand_df[aid].iloc[step]
        if base_demand > 0:
            noise = np.random.normal(0, noise_std * base_demand)  # Add noise proportional to demand
        return max(base_demand + noise, 0)
    
    # Right now there is no unique agent production
    def get_agent_production(self, aid, step, noise_std=0.0):
        noise = 0
        base_prod = self.prod_df[aid].iloc[step]
        # Temporary solar hack 
        # base_prod = base_prod * 10
        if base_prod > 0:
            noise = np.random.normal(0, noise_std * base_prod) 
        return max(base_prod + noise, 0)

    def get_spot_price(self, step, currency='EUR'):
        # TODO: implement zones if import/export influences pricing
        """Returns the spot price of electricity at a given timestamp in either EUR or DKK"""
        if currency == 'EUR':
            return self.price_df['SpotPriceEUR'].iloc[step]
        elif currency == 'DKK':
            return self.price_df['SpotPriceDKK'].iloc[step]
        
    def get_agent_battery_capacity(self, aid):
        return self.batt_cap_df[aid].iloc[0]
    
    def get_agent_maxdemand(self, aid):
        max = self.demand_df[aid].max()
        return max*1.1
    
    def get_all_maxdemand(self):
        max = self.demand_df.max().max()
        return max*1.1
    
    def get_all_maxprod(self):
        max_df = self.prod_df.iloc[:, 1:]
        max = max_df.max().max()
        return max*3
    
    def get_agent_maxproduction(self, aid):
        return self.prod_df[aid].max()
    
    def get_all_maxcap(self):
        max = self.batt_cap_df.max().max()
        return max
    
    def rotate(self):
        print("Rotating demand profiles")
        self.demand_df.columns = [f'H{(int(col[1:]) % 14) + 1}' for col in self.demand_df.columns]

# dm = DataManager()
# print(dm.demand_df.head())
# for i in range(14):
#     dm.rotate()
#     print(dm.demand_df.head())


