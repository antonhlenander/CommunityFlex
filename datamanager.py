import prepros as pp
import pandas as pd
import numpy as np

subfolder = 'data/fullyearPV_singleDemand/'
demand_path = f'{subfolder}demandprofiles.csv'
prod_path = f'{subfolder}PV.csv'
price_path = f'{subfolder}elspotprices.csv'
cap_path = f'data/eval/caps.csv'

class DataManager:
    def __init__(self, cap_path=cap_path, demand_path=demand_path, prod_path=prod_path, price_path=price_path):
        self.demand_df: pd.DataFrame = pp.load_demand_profile(demand_path)
        print("Demand profiles loaded!")
        self.prod_df = pp.load_production_data(prod_path)
        print("Production profiles loaded!")
        self.price_df = pp.load_price_data(price_path)
        self.cap_df = pp.load_cap_data(cap_path)

    def get_agent_daily_demand(self, aid):
        demand_profile = self.demand_df[aid].values
        return demand_profile
    
    def get_agent_daily_prod(self, aid):
        prod_profile = self.prod_df[aid].values
        return prod_profile
    
    def get_all_daily_demand(self):
        return self.demand_df.sum().sum()
    
    # def get_all_daily_prod(self):
    #     return self.prod_df.sum().sum()
    
    def get_price_array(self):
        return self.price_df.values
    
    def get_all_max_price(self):
        return self.price_df.max()

    # Old methods from other implementation

    def get_agent_demand(self, aid, step, noise_std=0.0):
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
        return max
    
    def get_all_maxdemand(self):
        max = self.demand_df.max().max()
        return max
    
    def get_all_maxprod(self):
        max_df = self.prod_df.iloc[:, 1:]
        max = max_df.max().max()
        return max*3
    
    def get_agent_maxproduction(self, aid):
        return self.prod_df[aid].max()
    
    def get_all_maxcap(self):
        max = self.batt_cap_df.max().max()
        return max
    
    def get_agent_cap(self, aid, rollout):
        return self.cap_df[aid].iloc[rollout]


    def rotate(self):
        print("Rotating demand profiles")
        self.demand_df.columns = [f'H{(int(col[1:]) % 14) + 1}' for col in self.demand_df.columns]

# dm = DataManager()
# prices = dm.get_price_array()



