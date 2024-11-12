from dataclasses import dataclass
from datetime import datetime, timedelta
from datamanager import DataManager

import phantom as ph
import gymnasium as gym
import numpy as np
import pandas as pd

import prepros as pp

class CommunityEnv(ph.StackelbergEnv):
    @dataclass(frozen=True)
    class View(ph.EnvView):

        # Current state
        current_price: float # Current price
        current_demand: float
        current_production: float
        current_covered_demand: float
        current_month: int
        current_date: int
        # For now, datasets start on a Monday, such that 1 = Monday, 2 = Tuesday. 
        # With other datasets just remember what day the dataset starts on.
        current_day: int 
        current_hour: int

        current_timestamp: str


        # Statistics
        avg_price: float # Average price

    def __init__(self, num_steps, network, leader_agents, follower_agents, **kwargs):
        self.current_price = 0.0
        self.current_demand = 0.0
        self.current_covered_demand = 0.0
        self.current_capacity = 0.0
        self.current_month = 0
        self.current_date = 0
        self.current_day = 0
        self.current_hour = 0
        self.current_timestamp = ""
        # Init stats
        self.avg_price = 0.0
        super().__init__(
            num_steps=num_steps,
            network=network,
            leader_agents=leader_agents,
            follower_agents=follower_agents,
            **kwargs
        )

    def view(self, neighbour_id=None) -> "CommunityEnv.View":
        return self.View(
            current_price=self.current_price,
            current_demand=self.current_demand,
            current_production=self.current_capacity,
            current_covered_demand=self.current_covered_demand,
            current_month=self.current_month,
            current_date=self.current_date,
            current_day=self.current_day,
            current_hour=self.current_hour,
            current_timestamp=self.current_timestamp,
            # Statistics
            avg_price=self.avg_price,
            **super().view({}).__dict__
            )

    def pre_message_resolution(self):
        super().pre_message_resolution()
        #self.current_price = self.agents["CM"].dynamic_price
        
    
        # # Get the current production of the generators
        # generator_capacities = [
        #     agent.current_production
        #     for agent in self.agents.values()
        #     if isinstance(agent, (GeneratorAgent, StrategicGeneratorAgent))
        # ]
        # # Sum the production
        # self.current_production = np.sum(generator_capacities)

        # Get the current demand of the consumers
        # consumer_demands = [
        #     agent.current_demand
        #     for agent in self.agents.values()
        #     if isinstance(agent, (ConsumerAgent, StrategicConsumerAgent))
        # ]
        # # Sum the demands
        # self.current_demand = np.sum(consumer_demands)
        # # Get the cleared price from the exchange agent
        # self.current_price = self.agents["EX"].clearing_price
        # # Get the agent and ctx




