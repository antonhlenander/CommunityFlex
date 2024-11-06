import prepros as pp
import pandas as pd

class DataManager:
    def __init__(self, cons_path, prod_path, price_path):
        self.cons_df: pd.DataFrame = pp.load_consumption_data(cons_path)
        self.prod_df: pd.DataFrame = pp.load_production_data(prod_path)
        self.price_df: pd.DataFrame = pp.load_price_data(price_path)

    def get_agent_demand(self, step, agent_key):
        return self.cons_df[agent_key].iloc[step]
    
    def get_agent_production(self, step, agent_key):
        return self.prod_df[agent_key].iloc[step]

    def get_spot_price(self, step, currency='EUR'):
        # TODO: implement zones if import/export influences pricing
        """Returns the spot price of electricity at a given timestamp in either EUR or DKK"""
        if currency == 'EUR':
            return self.price_df['SpotPriceEUR'].iloc[step]
        elif currency == 'DKK':
            return self.price_df['SpotPriceDKK'].iloc[step]
    
    def get_agent_maxdemand(self, agent_key):
        return self.cons_df[agent_key].max()
    
    def get_agent_maxproduction(self, agent_key):
        return self.prod_df[agent_key].max()
    
    def get_maxbidprice(self, currency='EUR') -> float:
        if currency == 'EUR':
            return self.price_df['SpotPriceEUR'].max()
        elif currency == 'DKK':
            return self.price_df['SpotPriceDKK'].max()
    