from dataclasses import dataclass

from datamanager import DataManager

import phantom as ph
import gymnasium as gym
import numpy as np
import pandas as pd

import prepros as pp

from stackelberg_custom import StackelbergEnvCustom

class CommunityEnv(ph.StackelbergEnv):
    # @dataclass(frozen=True)
    # class View(ph.EnvView):
    #     # Current state

    def __init__(self, num_steps, network, leader_agents, follower_agents, starttime, **kwargs):
        # Current time
        super().__init__(
            num_steps=num_steps,
            network=network,
            leader_agents=leader_agents,
            follower_agents=follower_agents,
            **kwargs
        )

    def view(self, neighbour_id=None) -> "CommunityEnv.View":
        return self.View(
            current_time=self.current_time,
            **super().view({}).__dict__
            )
    

    # def post_message_resolution(self):
    #     #super().pre_message_resolution()


class SimpleCommunityEnv(StackelbergEnvCustom):
    @dataclass(frozen=True)
    class View(ph.EnvView):

        # Current state
        current_price: float # Current price
        current_demand: float
        current_production: float

        current_timestamp: str

        # Statistics
        avg_price: float # Average price

    def __init__(self, num_steps, network, leader_agents, follower_agents, **kwargs):
        #self.current_timestamp = ""
        # Init stats
        #self.avg_price = 0.0
        super().__init__(
            num_steps=num_steps,
            network=network,
            leader_agents=leader_agents,
            follower_agents=follower_agents,
            **kwargs
        )

    # def view(self, neighbour_id=None) -> "CommunityEnv.View":
    #     return self.View(
    #         current_timestamp=self.current_timestamp,
    #         # Statistics
    #         avg_price=self.avg_price,
    #         **super().view({}).__dict__
    #         )

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




