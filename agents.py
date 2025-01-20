from dataclasses import dataclass
import phantom as ph
import gymnasium as gym
import numpy as np
import pandas as pd
from phantom.types import AgentID
from typing import Iterable, List, Sequence, Dict, Tuple
from messages import BuyBid, SellBid, ClearedSellBid, ClearedBuyBid, PriceUpdate, DummyMsg
from datamanager import DataManager
from market import Market
from phantom.telemetry import logger
import time
import random
import matplotlib.pyplot as plt
#from datamanager import DataManager

####################
# TSO Class
####################

class DSO():
    "Communicates capacity limitation to the Community Mediator"

    def __init__(self, dm, discount=0.0, local_price=None):
        self.price_array: list = []

        self.dm: DataManager = dm
        # Updates every step
        self.current_grid_price: float = 0

        self.residual_demand: float = 0
        # Update for every day
        self.daily_prices: list = []
        self.all_daily_demand: list = self.dm.get_all_daily_demand()
        self.all_daily_prod: list = []

        self.price_array = self.dm.get_price_array()

        # Hourly tariffs
        self.import_tariffs_winter = [0.2296,0.2296,0.2296,0.2296,0.2296,0.2296,0.6889,0.6889,0.6889,0.6889,0.6889,0.6889,0.6889,0.6889,0.6889,0.6889,0.6889,2.0666,2.0666,2.0666,2.0666,0.6889,0.6889,0.6889]
        self.import_tariffs_summer = [0.2296,0.2296,0.2296,0.2296,0.2296,0.2296,0.3444,0.3444,0.3444,0.3444,0.3444,0.3444,0.3444,0.3444,0.3444,0.3444,0.3444,0.8955,0.8955,0.8955,0.8955,0.3444,0.3444,0.3444]
        self.export_tariff = 0.00375 + 0.000875 + 0.01
    
    def compute_yearly_capacity_limits(self, var, ctx: ph.Context):
        yearly_cap_limits = []
        for h in range(8736):
            if h % 24 == 0:
                self.compute_residual_demand(h, ctx)
                yearly_cap_limits.extend(self.compute_capacity_limitation(h, var, ctx))
        return yearly_cap_limits

    def compute_residual_demand(self, step, ctx: ph.Context):
        # Gets the total demand and supply for the next 24 hours
        total_prod = 0
        views = ctx.agent_views.items()
        for aid, view in views:
            base_prod = self.dm.get_agent_daily_prod(aid, step)
            total_prod += base_prod*view.capacity
        # Compute the residual demand
        residual_demand = self.all_daily_demand - total_prod
        self.residual_demand = max(residual_demand, 0)
        return residual_demand
    
    def compute_capacity_limitation(self, step, var, ctx: ph.Context):
        daily_prices = self.price_array[step:step+24]
        # Hack for out of bounds index
        if len(daily_prices) < 24:
            daily_prices = self.price_array[0:24]
        max_price = np.max(daily_prices)
        min_price = np.min(daily_prices)
        mid_price = (max_price + min_price)/2
        avg_cap = self.residual_demand/24
        capacity_limitation = avg_cap - var * avg_cap * 2 * (daily_prices - mid_price) / (max_price - min_price)
        return capacity_limitation
        
        # The production has to be computed from each agent because of the random capacities.

##############################################################
# Strategic Community Mediator Agent
# Learns a dynamic pricing strategy for the local market
##############################################################
class StrategicCommunityMediator(ph.StrategicAgent):
    "Stategic Community Mediator Agent"

    @dataclass
    class Supertype(ph.Supertype):
        discount: float = 0.5
        cap_var: float = 0.5

    @dataclass(frozen=True)
    class MediatorView(ph.AgentView):
            current_grid_price: float
            current_local_price: float
            current_feedin_price: float
            discount: float
            cap_var: float

    def __init__(self, agent_id, dm, no_agents):
        super().__init__(agent_id)
 
        # Store the DataManager to get historical price data
        self.dm: DataManager = dm
        self.dso = DSO(dm)

        # Currents
        self.current_grid_price: float = 0 # Spot price + import tariff
        self.current_local_price: float = 0 # Dynamic price set by agent
        self.feedin_price: float = 0 # Spot price - export tariff
        self.current_local_tariff: float = 0 # Discounted import tariff
    
        self.price_array: list = []
        self.daily_prices: list = []

        # Cap limit variables
        self.yearly_cap_limits: list = []
        self.next_residual_demand: float
        self.current_cap_limit: float = 0
        self.capacity_balance: float = 0

        # Variables for reward and observation computation
        self.prev_price: float = 0
        self.prev_total_interactions: int = 0

        self.mediator_netloss: float = 0
        self.prev_mediator_netloss: float = 0

        self.prosumers_netloss: float = 0
        self.prev_prosumers_netloss: float = 0

        self.penalized_amount: float = 0
        self.prev_penalty: float = 0
        self.prev_cap_limit: float = 0

        # Aggregates
        self.current_total_load: float = 0
        self.current_total_prod: float = 0
        self.current_total_supply: float = 0
        self.current_total_import: float = 0
        self.current_total_export: float = 0
        self.acc_total_interactions: int = 0
        
        # Daily budget balance
        self.alltime_mediator_payments: float = 0 # Grid side payments
        self.alltime_mediator_income: float = 0 # Community side payments

        self.alltime_prosumers_payments: float = 0 # Community side payments
        self.alltime_prosumers_income: float = 0 # Grid side payments
        self.budget_balance: float = 0 # 
        self.prev_budget_balance: float = 0
        self.normed_balance: float = 0

        # More stats
        self.total_netloss = 0
        self.total_interactions = 0
        self.total_local_bought = 0
        # Normalization constants
        self.all_max_demand = 0
        self.all_max_prod = 0

        self.all_max_daily_demand = 0

        # Finding normalization constants
        self.max_balance = 0
        self.max_reward = 0
        self.min_reward = 0
        self.acc_reward = 0
        self.reward = 0

        # Training stats
        self.different_prices = []
        self.no_different_prices = 0 
        self.random_day = 0

        # Community net loss
        self.community_net_loss: float = 0

        # Community self-sufficiency
        self.community_self_sufficiency: float = 0

        self.price_plot = []
        self.action_plot = []
        self.balance_plot = []

        self.prices = np.ndarray(50)

        #self.action_space = gym.spaces.Discrete(50)
        self.action_space = gym.spaces.Box(low=0.05, high=1, shape=(1,), dtype=np.float32)


    def view(self, neighbour_id=None) -> ph.View:
        return self.MediatorView(
            current_grid_price = self.current_grid_price,
            current_local_price = self.current_local_price,
            current_feedin_price = self.feedin_price,
            discount = self.type.discount,
            cap_var = self.type.cap_var
            )

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                #"next_cap_limits": gym.spaces.Box(low=0.0, high=1.0, shape=(12,), dtype=np.float32),
                "infos": gym.spaces.Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32)
            }
        )

    def pre_message_resolution(self, ctx: ph.Context) -> None:
        # Reset all reward computations stats here at beginning of day
        step = ctx.env_view.current_step
        sim_step = (step + 1) // 2
        hour = ((sim_step-1) % 24) + 1

        # Resetting other stats
        ###############################################################
        self.total_local_bought = 0
        self.current_total_supply = 0
        self.acc_total_interactions = 0
        self.current_total_import = 0
        self.current_total_export = 0

    # Decode actions is the first method that is called in a step
    def decode_action(self, ctx: ph.Context, action):

        #print(f"------------------ MEDIATOR ACTION step {ctx.env_view.current_step} ------------------")
        #self.current_local_price = self.prices[5]
        self.current_local_price = self.max_price * action[0]
        #print(f"ACTION: ", action[0])
        #print("SET PRICE: ", self.current_local_price)
        self.action_plot.append(self.current_local_price)

        msgs = []
        for agent in ctx.neighbour_ids:
            msgs.append(
                (agent,PriceUpdate(self.current_local_price),)
            )

        return msgs

    def handle_batch(
        self, ctx: ph.Context, batch: Sequence[ph.Message]):

        buy_bids = []
        sell_bids = []
        msgs = []

        # Create lists of buy and sell bids
        for message in batch:
            if isinstance(message.payload, BuyBid):
                buy_bids.append(message)
            elif isinstance(message.payload, SellBid):
                sell_bids.append(message)
            else:
                msgs += self.handle_message(ctx, message)

        if len(buy_bids) > 0 or len(sell_bids) > 0:
            msgs = self.market_clearing(buy_bids=buy_bids, sell_bids=sell_bids)

        return msgs


    def market_clearing(
        self, buy_bids: Sequence[ph.Message[BuyBid]], sell_bids: Sequence[ph.Message[SellBid]]):   

        encoded_buy_bids = []
        encoded_sell_bids = []

        # ENCODING
        for bid in buy_bids:
            tuple = (bid.payload.buyer_id, bid.payload.buy_amount)
            encoded_buy_bids.append(tuple)

        for bid in sell_bids:
            tuple = (bid.payload.seller_id, bid.payload.sell_amount)
            encoded_sell_bids.append(tuple)

        # CLEAR BIDS
        cleared_buy_bids, cleared_sell_bids = Market.market_clearing(
            buy_bids=encoded_buy_bids, 
            sell_bids=encoded_sell_bids,
            local_price=self.current_local_price,
            grid_price=self.current_grid_price,
            feedin_price=self.feedin_price,
            local_tariff=self.current_local_tariff
            )

        # DECODING
        msgs = []

        # Create messages for the cleared buy bids
        for cleared_buy_bid in cleared_buy_bids:
            buyer_id, buy_amount, local_amount, grid_amount, prosumer_cost, mediator_cost = cleared_buy_bid
            msgs.append(
                (
                    buyer_id,
                    ClearedBuyBid(buyer_id, buy_amount, round(local_amount, 2), round(grid_amount, 2), round(prosumer_cost, 2), round(mediator_cost, 2)),
                )
            )
            # Update aggregates stats
            self.current_total_import += grid_amount
            self.total_local_bought += local_amount
            self.mediator_netloss += mediator_cost
            self.alltime_mediator_payments += mediator_cost

            self.prosumers_netloss += prosumer_cost
            self.alltime_prosumers_payments += prosumer_cost

        # Create messages for the cleared sell bids
        for cleared_sell_bid in cleared_sell_bids:
            seller_id, sell_amount, local_amount, grid_amount, prosumer_income, mediator_income = cleared_sell_bid
            msgs.append(
                (
                    seller_id,
                    ClearedSellBid(seller_id, sell_amount, round(local_amount, 2), round(grid_amount, 2), round(prosumer_income, 2), round(mediator_income, 2)),
                )
            )
            # Update aggregates stats
            self.current_total_export += grid_amount
            self.mediator_netloss -= mediator_income
            self.alltime_mediator_income += mediator_income

            self.prosumers_netloss -= prosumer_income
            self.alltime_prosumers_income += prosumer_income



        return msgs
    
    def post_message_resolution(self, ctx: ph.Context) -> None:
        step = ctx.env_view.current_step 
        sim_step = (step + 1) // 2
        hour = ((sim_step-1) % 24) + 1

        if hour == 1 and step % 2 == 0:
            # Reset dailies
            #self.alltime_mediator_payments = 0
            #self.alltime_prosumers_payments = 0
            # Reset budget balance
            #self.budget_balance = 0
            # Reset 
            #self.current_total_import = 0
            #self.current_total_export = 0
            #self.total_supply = 0
            self.total_netloss = 0
            self.total_interactions = 0

        # HOURLY COMPUTES AT EVEN STEPS 
        # i.e. at step 2 these are computed, such that CM agent can observe for its action in step 3.
        # these computes are updated after all prosumer agents make their updates, such that aggregates match
        ##################################################################             
                    
        if step % 2 == 0:
            self.penalized_amount = max(self.current_total_import - self.current_cap_limit, 0)
            self.penalty = self.penalized_amount*0
            self.capacity_balance = self.current_total_import - self.current_cap_limit
            #self.mediator_netloss += self.penalty
            #self.alltime_mediator_payments += self.penalty

    def encode_observation(self, ctx: ph.Context):
        step = ctx.env_view.current_step
        sim_step = (step + 1) // 2  + min(self.random_day, 8735)
        hour = ((sim_step-1) % 24) + 1
        hr_idx = sim_step % 24
        day = (sim_step // 24)
        month = day // 30

        print(f"------------------ STEP {ctx.env_view.current_step} MEDIATOR OBSERVATION ------------------")
        # DAILY COMPUTES AT END OF DAY AND AFTER RESET
        # Computations at even step for CM to observe at beginning of the next day
        ###############################################################
        if step == 0:
            self.yearly_cap_limits = self.dso.compute_yearly_capacity_limits(self.type.cap_var, ctx)
            # Extends yearly_cap_limits such that we dont get out of bounds error
            self.yearly_cap_limits.extend([10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10])
            views = ctx.agent_views.items()
            # for key, view in views:
            #     self.current_total_supply += view.supply
            #     self.acc_total_interactions += view.interactions

        if step % 2 == 0:
            self.current_cap_limit = self.yearly_cap_limits[hr_idx]
            next_cap_limits = self.yearly_cap_limits[sim_step:sim_step+12]
            #self.current_grid_price = self.price_array[sim_step] + self.dso.import_tariffs_winter[hr_idx]
            a = 0.45 * (1.1 + np.sin((2 * np.pi * (sim_step) / 17280 / self.squeeze)+self.displacement))
            upper_sin = 0.4 * a * (2.2 + np.sin((2 * np.pi * (sim_step) / 1440 / self.cycles)+self.displacement))
            lower_sin = 0.4 * a * (1.1 + np.sin((2 * np.pi * (sim_step) / 1440 / self.cycles)+self.displacement))
            upper = self.prices[min(int(upper_sin*39), 39)]
            lower = self.prices[min(int(lower_sin*39), 39)]
            daily = ((lower+upper)/2) + ((upper-lower)/2) * np.sin((2*np.pi*(sim_step) / 24))
            #self.price_plot.append(daily)
            #self.current_grid_price = daily
            self.current_grid_price = self.price_array[sim_step] + self.dso.import_tariffs_winter[hr_idx]
            self.price_plot.append(self.current_grid_price)
            self.feedin_price = self.current_grid_price - self.dso.export_tariff
            self.current_local_tariff = self.dso.import_tariffs_winter[hr_idx]*(1-self.type.discount)

            #print(f"DAILY RESIDUAL DEMAND: {self.next_residual_demand}")
            #print(f"NEXT CAP LIMITS: {next_cap_limits}")
            #print(f"SUM OF CAP LIMITS: {np.sum(next_cap_limits)}")
        
        #print(f"CURRENT LOCAL PRICE: {self.current_local_price}")
        #print(f"CURRENT GRID PRICE: {self.current_grid_price}")
        # print(f"CURRENT FEEDIN PRICE: {self.feedin_price}")
        # print(f"CURRENT LOCAL TARIFF: {self.current_local_tariff}")
        #print(f"STEP CURRENT TOTAL IMPORT: {self.current_total_import}")
        #print(f"STEP CAP LIMIT: {self.prev_cap_limit}")

        # Compute the budget balance - positive for profit, negative for loss
        self.budget_balance = self.prosumers_netloss - self.mediator_netloss
        normed_budget_balance = self.budget_balance / (self.mediator_netloss+self.prosumers_netloss+0.000001)
        
        views = ctx.agent_views.items()
        for key, view in views:
            self.current_total_prod += view.current_prod
            self.current_total_load += view.current_load
            self.acc_total_interactions += view.interactions

        marginal_netloss = self.mediator_netloss - self.prev_mediator_netloss
        marginal_interactions = self.acc_total_interactions - self.prev_total_interactions
        prev_price = self.prev_price
        prev_cap_limit = self.prev_cap_limit
        self.prev_price = self.current_local_price
        self.prev_cap_limit = self.current_cap_limit
        self.prev_total_netloss = self.mediator_netloss
        self.prev_total_interactions = self.acc_total_interactions
        
        # Normalization
        max_price = self.max_price
        epsilon = 0.000001

        # print(f"Capacity limits:", next_cap_limits)
        # print(f"Prev price: {prev_price}")
        # print(f"Current local price: {self.current_local_price}")
        # print(f"Current feedin price: {self.feedin_price}")
        # print(f"Current grid price: {self.current_grid_price}")
        # print(f"Current total supply in next step: {self.current_total_supply}")
        #print(f"Import in this ending step {self.current_total_import}")
        # print(f"Penalized amount: {self.penalized_amount}")
        # print(f"The cap limit for the ending step: {prev_cap_limit}")
        # print(f"The cap limit for the coming step: {self.current_cap_limit}")
        # print(f"The budget balance: {self.budget_balance}")
        # time.sleep(0.1)
        # print(f"Marginal net loss: {marginal_netloss}")

        observation = {
            # "next_cap_limits": np.divide
            #     (
            #         next_cap_limits,
            #         self.all_max_daily_demand, 
            #         dtype=np.float32
            #     ),
            "infos": np.array(
                [
                    hr_idx / 24,
                    prev_price / max_price,
                    self.current_local_price / max_price,
                    self.feedin_price / max_price,
                    self.current_grid_price / max_price,
                    #self.current_total_supply / self.all_max_daily_demand,
                    #self.current_total_prod / self.all_max_daily_demand,
                    #self.current_total_load / self.all_max_daily_demand,
                    #self.current_total_import / self.all_max_daily_demand,
                    #self.penalized_amount / self.all_max_daily_demand,
                    # prev_cap_limit / self.all_max_daily_demand,
                    #self.current_cap_limit / self.all_max_daily_demand,
                    #self.alltime_mediator_payments / 400000,
                    #self.alltime_mediator_income / 400000,
                    #self.prosumers_netloss / 400000,
                    abs(min(self.budget_balance, 0))/200000,
                    max(self.budget_balance, 0)/200000,
                    abs(min(normed_budget_balance, 0)),
                    max(normed_budget_balance, 0),
                ],
                dtype=np.float32
                )
            }
        # print(f"Observation: {observation}")
        # time.sleep(0.5)

        for key, value in observation.items():
            observation[key] = np.clip(value, -1, 1)

        return observation
    

    def compute_reward(self, ctx: ph.Context) -> float:
        # Budget balance
        # 1. Minimize cost
        # 2. Balance income and cost        
        # Compute amount of power above current capacity limitation
        # Compute the budget balance - positive for profit, negative for loss

        step = ctx.env_view.current_step
        sim_step = (step + 1) // 2
        hour = ((sim_step-1) % 24) + 1

        self.budget_balance = self.prosumers_netloss - self.mediator_netloss
        normed_budget_balance = self.budget_balance / (self.mediator_netloss+self.prosumers_netloss+0.000001)
        self.normed_balance = normed_budget_balance
    
        
        x = normed_budget_balance
        self.balance_plot.append(x)
        # upper_term  = pow(x, 2)
        # lower_term = 2 * pow(0.1, 2)
        # exp =  upper_term / lower_term
        # euler_exp = 1.3 * pow(np.e, -exp)
        # budget_signal = euler_exp - 0.7

        # MORE SQUEEZED FUNCTON:
        # lower_term = 0.025 + pow(x, 2)
        # budget_signal = (0.05 / lower_term) - 1

        lower_term = 0.1 + pow(x, 2)
        budget_signal = (0.2 / lower_term) - 1
        #print("BUDGET BALANCE: ", self.budget_balance)
        # Compute marginal budget change
        marginal_budget =  abs(self.prev_budget_balance) - abs(self.budget_balance)
        self.prev_budget_balance = self.budget_balance

        normed_marginal_change = marginal_budget / 40

        # if budget_signal > 0:
        #     self.reward = budget_signal + normed_marginal_change
        
        # if budget_signal < 0:
        #     self.reward = budget_signal #normed_marginal_change
        # if step == 48:
        #     if normed_budget_balance > (0.9):
        #         self.reward = 1
        #     else:
        #         self.reward = -1
        #else:
            #self.reward = -(normed_marginal_change

        #factor = pow(hour, 3) / pow(24, 3)
        #self.reward = factor*budget_signal
        self.reward = budget_signal

        #print("MARGINAL CHANGE: ", marginal_budget)
        # print("MEDIATOR NET LOSS: ", self.medif sim_step % 48:iator_netloss)
        # print("PROSUMERS NET LOSS: ", self.prosumers_netloss)
        #print("NORM BALANCE: ", self.normed_balance)
        # print("NORM MARGINAL CHANGE:", normed_marginal_change)
        #print("REWARD: ", self.reward)
    
        #print("REWARD: ", self.reward)
        self.acc_reward += self.reward
    

        # print("hour: ", hour)
        #print("reward: ", self.reward)
        # self.reward = normed_marginal_netloss + budget_signal
        #time.sleep(0.1)
        # Clip reward
        #self.reward = np.clip(self.reward, -1, 1)

        return self.reward
    
    
    def reset(self):
        super().reset()
        # Reset statistics
        self.total_earnings = 0
        # Reset previous variables (needed for reward computations)
        self.prev_price = 0
        self.prev_total_income = 0
        self.prev_total_netloss = 0
        self.prev_total_interactions = 0
        self.current_total_supply = 0
        self.acc_total_interactions = 0
        # Get normalization
        self.all_max_demand = self.dm.get_all_maxdemand()
        self.all_max_prod = self.dm.get_all_maxprod()*14 # TODO:fix 
        self.max_price = self.dm.get_all_max_price() + 2.0666
        self.min_price = self.dm.get_all_min_price() + 0.2296

        self.all_max_daily_demand = self.dm.get_all_max_daily_demand()
        self.alltime_mediator_payments = 0
        self.alltime_mediator_income = 0
        self.alltime_prosumers_payments = 0
        self.alltime_prosumers_income = 0
        self.mediator_netloss = 0
        self.prosumers_netloss = 0
        self.budget_balance = 0
        self.prev_budget_balance = 0
        self.prev_mediator_netloss = 0
        self.different_prices = []
        self.no_different_prices = 0
        self.acc_reward = 0
        self.reward = 0

        self.price_array = self.dm.get_price_array()
        #self.current_grid_price = self.price_array[0] + self.dso.import_tariffs_winter[0]
        self.current_grid_price = self.price_array[0] + self.dso.import_tariffs_winter[0]
        self.feedin_price = self.price_array[0] - self.dso.export_tariff
        self.current_local_tariff = self.dso.import_tariffs_winter[0]
        # TODO: Let's see what happens if max price is doubled
        #self.max_price = self.max_price * 2
        self.prices = np.linspace(0.1, self.max_price, num=50)
        random.seed(time.time())
        self.cycles = random.randint(4, 11)
        self.displacement = random.randint(1, 7)
        self.squeeze = random.randint(1, 2)
        #self.random_day = (random.randint(0, 363)*24)-1
        # PLOTTING
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(self.price_plot, 'r-')
        ax1.plot(self.action_plot, 'g-')
        ax1.set_ylim(0, 16)
        ax2.set_ylim(-1, 1)
        ax2.plot(self.balance_plot, 'b-')
        plt.savefig("price_plot.png")
        plt.close()
        self.price_plot = []
        self.action_plot = []
        self.balance_plot = []
        #print(self.prices)


##############################################################
# Simple Community Mediator Agent
# Sets a fixed price for the local market
##############################################################
class SimpleCommunityMediator(ph.Agent):#
    "Stategic Community Mediator Agent"

    @dataclass
    class Supertype(ph.Supertype):
        discount: float = 0.5
        std_dev: float = 0

    @dataclass(frozen=True)
    class MediatorView(ph.AgentView):
            current_grid_price: float
            current_local_price: float
            current_feedin_price: float
            discount: float


    def __init__(self, agent_id, dm):
        super().__init__(agent_id)

        self.dm: DataManager = dm

        self.import_tariffs = [0.2296,0.2296,0.2296,0.2296,0.2296,0.2296,0.6889,0.6889,0.6889,0.6889,0.6889,0.6889,0.6889,0.6889,0.6889,0.6889,0.6889,2.0666,2.0666,2.0666,2.0666,0.6889,0.6889,0.6889]
        self.export_tariff = 0.00375 + 0.000875 + 0.01

        self.daily_prices: list = []
        # Store the current prices
        self.current_grid_price: float
        self.current_local_price: float 
        self.current_feedin_price: float

        self.total_local_bought: float = 0
        self.current_total_export: float = 0
        self.current_total_import: float = 0

        self.max_price: float = 0

        self.mediator_netloss = 0
        self.alltime_mediator_payments = 0

        self.prosumers_netloss = 0
        self.alltime_prosumers_payments = 0

    def view(self, neighbour_id=None) -> ph.View:
        return self.MediatorView(
            current_grid_price = self.current_grid_price,
            current_local_price = self.current_local_price,
            current_feedin_price = self.current_feedin_price,
            discount = self.type.discount
            )
    def generate_messages(self, ctx):
        if ctx.env_view.current_step % 2 == 1:
            msgs = []
            for agent in ctx.neighbour_ids:
                msgs.append(
                    (agent,PriceUpdate(self.current_local_price),)
                )
            # print(f"-------- Step {ctx.env_view.current_step} CM sending message --------")
            # print(f"Current local price: {self.current_local_price}")
            return msgs


    def post_message_resolution(self, ctx: ph.Context) -> None:
        # Update grid price on even steps since this carries over to the next hour which begins on odd steps
        if ctx.env_view.current_step % 2 == 0: # This really is unnecessary if we only access index, but if we update something else it's probably not that simple
            # Integer division taking into account odd and even steps
            sim_step = (ctx.env_view.current_step + 1) // 2
            self.current_grid_price = self.price_array[sim_step] + self.import_tariffs[sim_step%24]
            price = self.current_grid_price * 2 
            noise = np.random.normal(1, self.type.std_dev)
            self.current_local_price = min(price*noise, self.max_price)
            self.current_feedin_price = self.price_array[sim_step] - self.export_tariff
            #print(f"Simple mediator updated prices: {self.current_grid_price}, {self.current_local_price}, {self.current_feedin_price}")

    def reset(self):
        super().reset()
        self.price_array = self.dm.get_price_array()
        self.current_grid_price = self.price_array[0] + self.import_tariffs[0]
        self.current_local_price = self.current_grid_price * 2
        self.current_feedin_price = self.price_array[0] - self.export_tariff
        self.max_price = self.dm.get_all_max_price() + 2.0666
        self.max_price = self.max_price
        

    def handle_batch(
        self, ctx: ph.Context, batch: Sequence[ph.Message]):
        buy_bids = []
        sell_bids = []
        msgs = []
        # Create lists of buy and sell bids
        for message in batch:
            if isinstance(message.payload, BuyBid):
                buy_bids.append(message)
            elif isinstance(message.payload, SellBid):
                sell_bids.append(message)
            else:
                msgs += self.handle_message(ctx, message)

        if len(buy_bids) > 0 or len(sell_bids) > 0:
            msgs = self.market_clearing(buy_bids=buy_bids, sell_bids=sell_bids)

        return msgs
    
    def market_clearing(
        self, buy_bids: Sequence[ph.Message[BuyBid]], sell_bids: Sequence[ph.Message[SellBid]]):   

        encoded_buy_bids = []
        encoded_sell_bids = []

        # ENCODING
        for bid in buy_bids:
            tuple = (bid.payload.buyer_id, bid.payload.buy_amount)
            encoded_buy_bids.append(tuple)

        for bid in sell_bids:
            tuple = (bid.payload.seller_id, bid.payload.sell_amount)
            encoded_sell_bids.append(tuple)

        # CLEAR BIDS
        cleared_buy_bids, cleared_sell_bids = Market.market_clearing(
            buy_bids=encoded_buy_bids, 
            sell_bids=encoded_sell_bids,
            local_price=self.current_local_price,
            grid_price=self.current_grid_price,
            feedin_price=0,
            local_tariff=0
            )

        # DECODING
        msgs = []
        self.current_total_import = 0
        self.current_total_export = 0
        # Create messages for the cleared buy bids
        for cleared_buy_bid in cleared_buy_bids:
            buyer_id, buy_amount, local_amount, grid_amount, prosumer_cost, mediator_cost = cleared_buy_bid
            msgs.append(
                (
                    buyer_id,
                    ClearedBuyBid(buyer_id, buy_amount, round(local_amount, 2), round(grid_amount, 2), round(prosumer_cost, 2), round(mediator_cost, 2)),
                )
            )
            # Update aggregates stats
            self.current_total_import += grid_amount
            self.total_local_bought += local_amount
            self.mediator_netloss += mediator_cost
            self.alltime_mediator_payments += mediator_cost

            self.prosumers_netloss += prosumer_cost
            self.alltime_prosumers_payments += prosumer_cost
  
        # Create messages for the cleared sell bids
        for cleared_sell_bid in cleared_sell_bids:
            seller_id, sell_amount, local_amount, grid_amount, prosumer_income, mediator_income = cleared_sell_bid
            msgs.append(
                (
                    seller_id,
                    ClearedSellBid(seller_id, sell_amount, round(local_amount, 2), round(grid_amount, 2), round(prosumer_income, 2), round(mediator_income, 2)),
                )
            )
            # Update aggregates stats
            self.current_total_export += grid_amount
            self.mediator_netloss -= mediator_income
            self.alltime_mediator_payments -= mediator_income

            self.prosumers_netloss -= prosumer_income
            self.alltime_prosumers_payments -= prosumer_income

        return msgs


##############################################################
# Strategic RL prosumer agent
##############################################################
class StrategicProsumerAgent(ph.StrategicAgent):

    @dataclass
    class Supertype(ph.Supertype):
        capacity: int = 1
        eta: float = 0.1
        rollout: int = 0

    @dataclass(frozen=True)
    class ProsumerView(ph.AgentView):
            supply: float
            current_prod: float
            current_load: float
            net_loss: float
            income: float
            interactions: float
            capacity: int
            current_import: float
            current_export: float

    def __init__(self, agent_id, mediator_id, data_manager):
        super().__init__(agent_id)

        # Store the ID of the community mediator
        self.mediator_id = mediator_id

        # Store the DataManager
        self.dm: DataManager = data_manager

        # Agent properties
        self.battery_cap: float = 0
        self.charge_rate: float = 0

        # Agent currents
        self.current_load: float = 0 
        self.current_prod: float = 0
        self.current_charge: float = 0
        self.current_supply: float = 0

        self.self_consumption: float = 0
        self.avail_energy: float = 0
        self.surplus_energy: float = 0
        self.current_local_bought: float = 0
        self.current_import: float = 0
        self.current_export: float = 0

        # Agent constraints
        self.remain_batt_cap: float = 0
        self.max_batt_charge: float = 0
        self.max_batt_discharge: float = 0

        # Accumulated statistics
        self.acc_local_market_coin: float = 0
        self.acc_feedin_coin: float = 0
        self.acc_local_market_cost: float = 0
        self.acc_grid_market_cost: float = 0
        self.acc_invalid_actions: int = 0
        self.acc_grid_interactions: int = 0
        self.net_loss: float = 0
        self.acc_reward: float = 0

        # Normalization factors
        self.all_max_load: float = 0
        self.all_max_prod: float = 0
        self.all_max_cap: float = 0
        
        # Utility
        self.utility_prev: float = 0
        self.max_utility: float = 0

        # Reward
        self.reward: float = 0

        # = {Buy, BuyCharge, Sell, SellCharge, Charge, No-op}
        self.action_space = gym.spaces.Discrete(6)

        self.episode = 0

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                # Can include type here as well in the future maybe
                "action_mask": gym.spaces.Box(0, 1, shape=(6,), dtype=np.float32),

                "observations": gym.spaces.Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32),
            }
        )

    def view(self, neighbour_id=None) -> ph.View:
        """@override
        Create the view for the mediator to compute aggregate info.
        """
        return self.ProsumerView(
            supply = self.current_supply,
            current_prod = self.current_prod,
            current_load = self.current_load,
            net_loss = self.net_loss,
            income = self.acc_feedin_coin+self.acc_local_market_coin,
            interactions = self.acc_grid_interactions,
            capacity = self.type.capacity,
            current_import = self.current_import,
            current_export = self.current_export
        )

    def buy_power(self, amount):
        if amount > 0:
            buy_amount = round(amount, 2)
            return [(self.mediator_id, BuyBid(self.id, buy_amount))]


    def sell_power(self, amount):
        if amount > 0:
            return [(self.mediator_id, SellBid(self.id, round(amount, 2)))]
    
    # Charge or decharge battery by a certain amount
    def charge_battery(self, amount):
        # Only charge the battery by a positive amount capped by charge rate and if the battery is not full.
        if amount > 0 and self.current_charge < self.battery_cap:
            self.current_charge += min(self.max_batt_charge, amount)
            self.current_charge = round(self.current_charge, 2)
        else:
            self.acc_invalid_actions += 1
            self.curr_invalid_actions = 1000

    def discharge_battery(self, amount):
        if amount > 0 and self.current_charge >= amount:
            self.current_charge -= min(self.max_batt_discharge, amount)
            self.current_charge = max(self.current_charge, 0)
            self.current_charge = round(self.current_charge, 2)

    def pre_message_resolution(self, ctx: ph.Context) -> None:
        self.self_consumption = 0.0
        self.current_local_bought = 0.0
        # Reset stats
     
        # if ctx.env_view.current_step == 1:
        #     self.episode += 1

    def decode_action(self, ctx: ph.Context, action: np.ndarray):
        #print(action)
        msgs = []
        # print(f"----------- Step {ctx.env_view.current_step} decode action -------------")
        #msgs.extend(self.generate_info_message())

        if action == 0:
           # Buy enough power to cover own deficit
            if self.current_supply >= 0:
                self.acc_invalid_actions += 1
                return msgs
            else:
                msgs.extend(self.buy_power(abs(self.current_supply)))
                return msgs
        
        elif action == 1:
            # Buy power to charge and fill possible deficit
            if self.current_supply >= self.max_batt_charge:
                self.acc_invalid_actions += 1
            else:
                deficit = abs(min(self.current_supply, 0))
                # Capping the buy amount to 1 kWh
                buy_amount = min(1, self.max_batt_charge) + deficit
                msgs.extend(self.buy_power(buy_amount))
                return msgs

        elif action == 2: 
            # Sell own surplus production
            if self.current_supply > 0:
                msgs.extend(self.sell_power(self.current_supply))
                return msgs
            else:
                self.acc_invalid_actions += 1
                return msgs
            
        elif action == 3:            
            # Sell from battery and possible surplus production
            sell_amount = self.max_batt_discharge + self.current_supply
            # Add to self consumption if the agent has negative supply
            self.self_consumption += abs(min(self.current_supply, 0))
            if sell_amount > 0:
                msgs.extend(self.sell_power(sell_amount))
                self.discharge_battery(self.max_batt_discharge)
                return msgs
            else:
                self.acc_invalid_actions += 1
                return msgs
        
        elif action == 4:
            # Charge battery
            # Can only charge if the agent has positive supply
            if self.current_supply > 0:
                self.charge_battery(self.current_supply)
            else:
                self.acc_invalid_actions += 1
            return msgs
        
        elif action == 5:
            # If the agent has negative supply, and it has enough charge to cover it, it will do so
            if self.current_supply < 0 and self.max_batt_discharge >= abs(self.current_supply):
                self.self_consumption += abs(self.current_supply)
                self.discharge_battery(abs(self.current_supply))
                return msgs
            # If the agent has negative supply and it does not have enough charge to cover it, this is an invalid action
            elif self.current_supply < 0 and self.max_batt_discharge < abs(self.current_supply):
                self.acc_invalid_actions += 1
                self.curr_invalid_actions = 1000
                return msgs
            # The agent might just have surplus energy and also choose this action, 
            # then it just does not cooperate, but it is a legal action.
    
    @ph.agents.msg_handler(PriceUpdate)
    def handle_priceupdate(self, _ctx: ph.Context, msg: ph.Message):
        self.current_local_price = msg.payload.current_price

    @ph.agents.msg_handler(ClearedBuyBid)
    def handle_cleared_buybid(self, _ctx: ph.Context, msg: ph.Message):
        # Charge battery if agent bought more than load
        energy_to_charge = msg.payload.buy_amount - abs(min(self.current_supply, 0))
        if energy_to_charge > 0:
            self.charge_battery(energy_to_charge)
        # Update current import
        # Update accumulated statistics
        self.current_import = msg.payload.grid_amount
        self.current_local_bought += msg.payload.local_amount
        self.acc_local_market_cost += msg.payload.prosumer_cost
        if msg.payload.local_amount > 0:
            self.acc_grid_interactions += 1


    @ph.agents.msg_handler(ClearedSellBid)
    def handle_cleared_sellbid(self, _ctx: ph.Context, msg: ph.Message):
        # Calculate the energy to discharge
        # Update statistics
        self.acc_local_market_coin += msg.payload.prosumer_income
        if msg.payload.local_amount > 0:
            # Update accumulated statistics
            self.acc_grid_interactions += 1

    def post_message_resolution(self, ctx: ph.Context):
        # We update everything after the messages have been resolved
        # This is where the the values are updated for the observation in next step
        double_step = ctx.env_view.current_step 
        # only get new values if even step
        if double_step % 2 == 0:
            # Integer division taking into account odd and even steps
            sim_step = (ctx.env_view.current_step + 1) // 2
            # Convert to hours since demand profile is just 24 hours
            hour = sim_step % 24
            # Update current load only
            self.current_load = self.dm.get_agent_demand(self.id, hour)
            # Update production
            self.current_prod = self.dm.get_agent_production(self.id, sim_step)*self.type.capacity
            # Update current own supply
            self.current_supply = round(self.current_prod - self.current_load, 2)
            
        # Update battery constraints
        self.remain_batt_cap = round(self.battery_cap - self.current_charge, 2)
        self.max_batt_charge = round(min(self.remain_batt_cap, self.charge_rate), 2)
        self.max_batt_discharge = round(min(self.current_charge, self.charge_rate), 2)

        # Update statistics
        self.net_loss = (
            self.acc_local_market_cost + self.acc_grid_market_cost 
            - self.acc_local_market_coin - self.acc_feedin_coin
        )
        self.avail_energy = self.current_prod + self.max_batt_discharge
        self.surplus_energy = self.avail_energy - self.current_load
       
        # Update self consumption
        if self.current_supply < 0:
            self.self_consumption += self.current_prod
        else:
            self.self_consumption += self.current_load


    def encode_observation(self, ctx: ph.Context):        
        # Action masking
        # Can agent buy to cover its own deficit?
        # Not if it has enough supply.
        if self.current_supply >= 0:
            buy = 0
        else:
            buy = 1
        # Can agent buy to charge?
        if self.current_supply >= self.max_batt_charge:
            buy_charge = 0
        else:
            buy_charge = 1
        # Can agent sell its own power?
        if self.current_supply <= 0:
            sell = 0
        else:
            sell = 1
        # Can agent sell its charge?
        if self.max_batt_discharge > abs(min(self.current_supply, 0)):
            sell_batt = 1
        else: 
            sell_batt = 0
        # Can agent charge?
        if self.current_supply > 0 and self.max_batt_charge > 0:
            charge = 1
        else:
            charge = 0
        # Can agent do nothing?
        if self.current_supply < 0 and self.max_batt_discharge < abs(min(self.current_supply, 0)):
            noop = 0
        else:
            noop = 1

        # print(f"--------------- STEP {ctx.env_view.current_step} PROSUMER OBSERVATION --------------")
        # print(f"Current local price: ", {self.current_local_price})
        # print(f"Current load: {self.current_load}")
        # print(f"Current production: {self.current_prod}")
        # print(f"Current charge: {self.current_charge}")
        # print(f"Battery cap: {self.all_max_cap}")
        # print(f"Charge rate: {self.charge_rate}")
        # print(f"Acc. income: {self.acc_local_market_coin}")
        # print(f"Acc. cost: {self.acc_local_market_cost}")

        # time.sleep(0.1)

        observation = {
            'observations' : np.array([
                    self.current_local_price / self.max_price,
                    self.current_load / self.all_max_cap,
                    self.current_prod / self.all_max_cap,
                    self.current_charge / self.all_max_cap,
                    self.battery_cap / self.all_max_cap,
                    self.charge_rate / self.all_max_cap,
                    self.acc_local_market_coin / 30000,
                    self.acc_local_market_cost / 30000,
                    self.acc_grid_interactions / 8760], dtype=np.float32),
            'action_mask' : np.array([buy, buy_charge, sell, sell_batt, charge, noop], dtype=np.float32)
        }

        np.clip(observation['observations'], -1, 1, out=observation['observations'])

        return observation

    def compute_reward(self, ctx: ph.Context) -> float:
        I_t = self.acc_local_market_coin + self.acc_feedin_coin
        C_t = self.acc_local_market_cost + self.acc_grid_market_cost
        eta_comp = 1 - self.type.eta
        upper_term = (pow(I_t, eta_comp) - 1)
        utility = (upper_term / eta_comp) - C_t
        # Final reward
        marginal_utility = utility - self.utility_prev 
        # Update utility
        self.utility_prev = utility
        if marginal_utility > 100 or marginal_utility < -100:
            print("!!!!!!!!!!!!!!!MARGINAL UTILITY: ", marginal_utility)
        self.reward = min(marginal_utility/100, 1)
        return self.reward

    def reset(self):
        # Reset to sample type
        super().reset()
        if self.type.rollout == 0:
            print(f"Strategic agent reset with sample capacity: {self.type.capacity} and eta: {self.type.eta}")
        # Reset statistics
        self.acc_local_market_coin = 0
        self.acc_feedin_coin = 0
        self.acc_local_market_cost = 0
        self.acc_grid_market_cost = 0
        self.acc_invalid_actions = 0
        self.acc_grid_interactions = 0
        self.acc_reward = 0
        self.net_loss = 0
        #
        if self.type.rollout == 1:
            self.type.capacity = self.dm.get_agent_cap(self.id, self.episode)
            print(f"Agent {self.id} capacity set to {self.type.capacity} for episode {self.episode}")
        # Get battery capacity
        # Compute battery capacity
        self.battery_cap = 5 * self.type.capacity
        # Compute charge rate
        self.charge_rate = self.battery_cap / 2
        # Get demand data for first step
        self.current_load = self.dm.get_agent_demand(self.id, 0)
        # Get production data for first step
        self.current_prod = self.dm.get_agent_production(self.id, 0)*self.type.capacity
        # Update current own supply
        self.current_supply = round(self.current_prod - self.current_load, 2)
        # Reset battery charge
        self.current_charge = self.battery_cap / 2
        # Reset battery constraints
        self.utility_prev = 0
        # Update battery constraints
        self.remain_batt_cap = round(self.battery_cap - self.current_charge, 2)
        self.max_batt_charge = round(min(self.remain_batt_cap, self.charge_rate), 2)
        self.max_batt_discharge = round(min(self.current_charge, self.charge_rate), 2)
        # Reset and get energy currents
        self.surplus_energy = 0.0
        self.avail_energy = self.current_supply + self.max_batt_discharge
        self.self_consumption = 0.0
        # Normalization factors
        self.all_max_load = self.dm.get_all_maxdemand()
        self.all_max_prod = self.dm.get_all_maxprod()
        self.all_max_cap = 15
        self.max_price = self.dm.get_all_max_price() + 2.0666
        self.max_price = self.max_price * 2

            


##############################################################
# Simple prosumer agent
##############################################################

class SimpleProsumerAgent(ph.Agent):

    @dataclass
    class Supertype(ph.Supertype):
        capacity: int = 1
        eta: float = 0.1
        greed: float = 0.75
        rollout: int = 0

    @dataclass(frozen=True)
    class ProsumerView(ph.AgentView):
            supply: float
            current_prod: float
            current_load: float
            net_loss: float
            income: float
            interactions: float
            capacity: int
            current_import: float
            current_export: float

    def __init__(self, agent_id, mediator_id, data_manager):
        
        # Store the ID of the community mediator
        self.mediator_id = mediator_id

        # Store the DataManager
        self.dm: DataManager = data_manager

        # Agent properties
        self.battery_cap: float = 0
        self.charge_rate: float = 0

        # Agent currents
        self.current_load: float = 0 
        self.current_prod: float = 0
        self.current_charge: float = 0
        self.current_supply: float = 0
        self.self_consumption: float = 0
        self.avail_energy: float = 0
        self.surplus_energy: float = 0
        self.current_local_bought: float = 0
        
        self.current_import: float = 0
        self.current_export: float = 0

        # Agent constraints
        self.remain_batt_cap: float = 0
        self.max_batt_charge: float = 0
        self.max_batt_discharge: float = 0

        # Accumulated statistics
        self.acc_local_market_coin: float = 0
        self.acc_feedin_coin: float = 0
        self.acc_local_market_cost: float = 0
        self.acc_grid_market_cost: float = 0
        self.acc_grid_interactions: int = 0
        self.acc_invalid_actions: int = 0 # just here to not get an error
        self.net_loss: float = 0 

        self.rotate = False

        self.episode = 0
        
        super().__init__(agent_id)

    
    def view(self, neighbour_id=None) -> ph.View:
        """@override
        Create the view for the mediator to compute aggregate info.
        """
        return self.ProsumerView(
            supply = self.current_supply,
            current_prod = self.current_prod,
            current_load = self.current_load,
            net_loss = self.net_loss,
            income = self.acc_feedin_coin+self.acc_local_market_coin,
            interactions = self.acc_grid_interactions,
            capacity = self.type.capacity,
            current_import = self.current_import,
            current_export = self.current_export
        )

    # Charge or decharge battery by a certain amount
    def charge_battery(self, amount):
        # Only charge the battery by a positive amount capped by charge rate and if the battery is not full.
        if amount > 0 and self.current_charge < self.battery_cap:
            self.current_charge += min(self.max_batt_charge, amount)
        else:
            self.acc_invalid_actions += 1

    def discharge_battery(self, amount):
        if amount > 0 and self.current_charge >= amount:
            self.current_charge -= min(self.max_batt_discharge, amount)
            self.current_charge = max(self.current_charge, 0)

    def pre_message_resolution(self, ctx: ph.Context) -> None:
        self.self_consumption = 0.0
        self.current_local_bought = 0.0
        if ctx.env_view.current_step == 1:
            self.episode += 1

    def generate_messages(self, ctx: ph.Context):
        # Evaluate greediness of agent.
        if self.current_charge >= self.type.greed*self.battery_cap:
            # If balanced supply or excess, sell what can be discharged from battery + supply
            if self.current_supply >= 0:
                amount = self.max_batt_discharge + self.current_supply
                if amount > 0:
                    return [(self.mediator_id, SellBid(self.id, amount))]
            # Cases of negative supply:
            elif self.current_supply < 0:
                # If enough charge to cover, discharge the deficit and sell the rest.
                if self.max_batt_discharge >= abs(self.current_supply):
                    # The below is a subtraction for the selfconsumption
                    energy_to_sell = self.max_batt_discharge - abs(self.current_supply)
                    self.discharge_battery(abs(self.current_supply))
                    if energy_to_sell > 0:
                        return [(self.mediator_id, SellBid(self.id, energy_to_sell))]
                # If not enough charge to cover, buy the deficit
                elif self.max_batt_discharge < abs(self.current_supply):
                    return [(self.mediator_id, BuyBid(self.id, abs(self.current_supply)))]

        # In the case of battery charge below threshold
        elif self.current_charge < self.type.greed*self.battery_cap:
            # If balanced supply, do nothing
            if self.current_supply == 0:
                return []
            # Cases of negative supply:
            elif self.current_supply < 0:
                # If enough charge to cover, discharge the deficit
                if self.max_batt_discharge >= abs(self.current_supply):
                    self.discharge_battery(abs(self.current_supply))
                    return []
                # If not enough charge to cover, buy the deficit
                elif self.max_batt_discharge < abs(self.current_supply):
                    return [(self.mediator_id, BuyBid(self.id, abs(self.current_supply)))]
            # Case of positive supply, charge the surplus:
            elif self.current_supply > 0:
                self.charge_battery(self.current_supply)
                return []
            
    @ph.agents.msg_handler(PriceUpdate)
    def handle_priceupdate(self, _ctx: ph.Context, msg: ph.Message):
        self.current_local_price = msg.payload.current_price

    @ph.agents.msg_handler(ClearedBuyBid)
    def handle_cleared_buybid(self, _ctx: ph.Context, msg: ph.Message):
        # Charge battery if agent bought more than load
        energy_to_charge = msg.payload.buy_amount - abs(min(self.current_supply, 0))
        if energy_to_charge > 0:
            self.charge_battery(energy_to_charge)
        # Update current import
        # Update accumulated statistics
        self.current_import = msg.payload.grid_amount
        self.current_local_bought += msg.payload.local_amount
        self.acc_local_market_cost += msg.payload.prosumer_cost
        if msg.payload.local_amount > 0:
            self.acc_grid_interactions += 1


    @ph.agents.msg_handler(ClearedSellBid)
    def handle_cleared_sellbid(self, _ctx: ph.Context, msg: ph.Message):
        # Calculate the energy to discharge
        # Update statistics
        self.acc_local_market_coin += msg.payload.prosumer_income
        if msg.payload.local_amount > 0:
            # Update accumulated statistics
            self.acc_grid_interactions += 1

    def post_message_resolution(self, ctx: ph.Context):
        # We update everything after the messages have been resolved
        # This is where the the values are updated for the observation in next step
        double_step = ctx.env_view.current_step 
        # only get new values if even step
        if double_step % 2 == 0:
            # Integer division taking into account odd and even steps
            sim_step = (ctx.env_view.current_step + 1) // 2
            # Convert to hours since demand profile is just 24 hours
            hour = sim_step % 24
            # Update current load only
            self.current_load = self.dm.get_agent_demand(self.id, hour-1)
            # Update production
            self.current_prod = self.dm.get_agent_production(self.id, sim_step-1)*self.type.capacity
            # Update current own supply
            self.current_supply = round(self.current_prod - self.current_load, 2)
            
        # Update battery constraints
        self.remain_batt_cap = round(self.battery_cap - self.current_charge, 2)
        self.max_batt_charge = round(min(self.remain_batt_cap, self.charge_rate), 2)
        self.max_batt_discharge = round(min(self.current_charge, self.charge_rate), 2)

        # Update statistics
        self.net_loss = (
            self.acc_local_market_cost + self.acc_grid_market_cost 
            - self.acc_local_market_coin - self.acc_feedin_coin
        )
        self.avail_energy = self.current_prod + self.max_batt_discharge
        self.surplus_energy = self.avail_energy - self.current_load
        # Below does not work for strategic agent!
        if self.surplus_energy >= 0:
            self.self_consumption = self.current_load
        else:
            self.self_consumption = self.current_prod
        
        
    def reset(self):
        # Reset for type
        super().reset()
        if self.type.rollout == 0:
            print(f"Simple agent {self.id} sampled with cap: {self.type.capacity}, greed: {self.type.greed} & eta: {self.type.eta}")
        # Reset statistics
        self.acc_local_market_coin = 0.0
        self.acc_feedin_coin = 0.0
        self.acc_local_market_cost = 0.0
        self.acc_grid_market_cost = 0.0
        self.acc_grid_interactions = 0.0
        self.net_loss = 0.0
        #
        if self.type.rollout == 1:
            self.type.capacity = self.dm.get_agent_cap(self.id, self.episode)
            print(f"Agent {self.id} capacity set to {self.type.capacity} for episode {self.episode}")
        # Get battery capacity
        self.battery_cap = 5*self.type.capacity
        # Get charge rate
        self.charge_rate = self.battery_cap / 2
        # Get demand data for first step
        self.current_load = self.dm.get_agent_demand(self.id, 0)
        # Get production data for first step
        self.current_prod = self.dm.get_agent_production(self.id, 0)*self.type.capacity
        # Update current own supply
        self.current_supply = round(self.current_prod - self.current_load, 2)
        # Reset battery charge
        self.current_charge = self.battery_cap / 2
        # Reset battery constraints
        self.remain_batt_cap = round(self.battery_cap - self.current_charge, 2)
        self.max_batt_charge = round(min(self.remain_batt_cap, self.charge_rate), 2)
        self.max_batt_discharge = round(min(self.current_charge, self.charge_rate), 2)
        # 
        self.surplus_energy = 0.0
        self.avail_energy = self.current_supply + self.max_batt_discharge
        #
        self.self_consumption = 0.0
        if self.rotate:
            self.dm.rotate()
   


##############################################################
# Dummy Agent
##############################################################

class DummyAgent(ph.StrategicAgent):
    def __init__(self, agent_id: ph.AgentID):
        super().__init__(agent_id)

        self.obs: float = 0

        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,))

        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,))

    def encode_observation(self, ctx: ph.Context):
        return np.array([0.9])

    def decode_action(self, ctx: ph.Context, action):
        # We perform this action by sending a Bid message to the generator.
        return [("EX", DummyMsg("Hello"))]

    def compute_reward(self, ctx: ph.Context) -> float:
        return 0.9

