from dataclasses import dataclass
import phantom as ph
import gymnasium as gym
import numpy as np
import pandas as pd
from phantom.types import AgentID
from typing import Iterable, List, Sequence, Dict, Tuple
from messages import BuyBid, SellBid, ClearedSellBid, ClearedBuyBid, DummyMsg
from datamanager import DataManager
from market import Market
from phantom.telemetry import logger
import time
import random
#from datamanager import DataManager


##############################################################
# Strategic Community Mediator Agent
# Learns a dynamic pricing strategy for the local market
# TODO: Implement observation space and reward function
##############################################################
class StrategicCommunityMediator(ph.StrategicAgent):
    "Stategic Community Mediator Agent"

    @dataclass(frozen=True)
    class MediatorView(ph.AgentView):
            """
            We expose a view of the grid price, local market price and feed-in price to the agents.
            """
            public_info: dict

    def __init__(self, agent_id, data_manager):
        super().__init__(agent_id)
 
        # Store the DataManager to get historical price data
        self.dm: DataManager = data_manager

        # Store the current grid price
        self.current_grid_price: float = 0
        # Current local price
        self.current_local_price: float = 0
        # Feedin price
        self.feedin_price: float = 0

        # Community net loss
        self.community_net_loss: float = 0

        # Community self-sufficiency
        self.community_self_sufficiency: float = 0
        
        # Include forecast demand data at some point?
        # = [Month, Date, Day, Hour, Cleared Price, Current Production]
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(6,))

        # 10 price brackets for now
        self.action_space = gym.spaces.Discrete(10)


    def view(self, neighbour_id=None) -> ph.View:
        """@override
        Create the view for the agents to see the public information.
        """
        return self.MediatorView(
            public_info={
                "grid_price": self.current_grid_price, 
                "local_price": self.current_local_price, 
                "feedin_price": self.feedin_price
            },
        )

    # Decode actions is the first method that is called in a step
    def decode_action(self, ctx: ph.Context, action):
        # Translate the action to a price (the grid price is the maximum price)
        self.current_local_price = float(action) * self.current_grid_price / 10

    # 2nd step in a step
    # def pre_message_resolution(self, ctx: ph.Context):
        

    def handle_batch(
        self, ctx: ph.Context, batch: Sequence[ph.Message]):
        """@override
        We override the method `handle_batch` to consume all the bids messages
        as one block in order to perform the auction. The batch object contains
        all the messages that were sent to the actor.

        Note:
        -----
        The default logic is to consume each message individually.
        """
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

        if len(buy_bids) > 0 and len(sell_bids) > 0:
            msgs = self.market_clearing(buy_bids=buy_bids, sell_bids=sell_bids)

        for msg in msgs:
            logger.log_msg_send(msg)

        return msgs


    def market_clearing(
        self, buy_bids: Sequence[ph.Message[BuyBid]], sell_bids: Sequence[ph.Message[SellBid]]):   
        """
        Encode and decode buy and sell bids and pass to external market clearing mechanism.

        """
        encoded_buy_bids = []
        encoded_sell_bids = []

        # ENCODING
        for bid in buy_bids:
            tuple = (bid.payload.buyer_id, bid.payload.mwh, bid.payload.price)
            encoded_buy_bids.append(tuple)

        for bid in sell_bids:
            tuple = (bid.payload.seller_id, bid.payload.mwh, bid.payload.price)
            encoded_sell_bids.append(tuple)

        # CLEAR BIDS
        cleared_buy_bids, cleared_sell_bids = Market.market_clearing(sell_bids=encoded_sell_bids, buy_bids=encoded_buy_bids)

        # DECODING
        msgs = []
        for cleared_buy_bid in cleared_buy_bids:
            buyer_id, buy_amount, local_cost, grid_cost = cleared_buy_bid
            decoded_cleared_buy_bid = ClearedBuyBid(buyer_id, buy_amount, local_cost, grid_cost)
            # Create message for both seller and buyer
            msg = (buyer_id, decoded_cleared_buy_bid)
            msgs.extend(msg)
        for cleared_sell_bid in cleared_sell_bids:
            seller_id, sell_amount, local_income, grid_income = cleared_sell_bid
            decoded_cleared_buy_bid = ClearedBuyBid(seller_id, sell_amount, local_income, grid_income)
            # Create message for both seller and buyer
            msg = (buyer_id, decoded_cleared_buy_bid)
            msgs.extend(msg)

        return msgs


    def encode_observation(self, ctx: ph.Context):
        month = ctx.env_view.current_month
        date = ctx.env_view.current_date
        day = ctx.env_view.current_day
        hour = ctx.env_view.current_hour
        # Clip the cleared price to be 0 in case of negative prices
        cleared_price = max(ctx.env_view.current_price, 0) 
        MAX_BID_PRICE = ctx.env_view.MAX_BID_PRICE
        

        return np.array(
            [
                float(month / 12),
                float(date / 31),
                float(day / 7),
                float(hour / 23),
                float(cleared_price / MAX_BID_PRICE),
                float(self.current_production / self.capacity)
            ],
            dtype=np.float32,
        )

    def compute_reward(self, ctx: ph.Context) -> float:
        max_earning = self.capacity * self.marginal_cost
        costs = self.current_production * self.marginal_cost
        profit = self.current_earnings - costs

        # Apply sqrt scaling, handling positive and negative profit separately
        if profit > 0:
            # Logarithmic scaling for positive profits
            scaled_reward = np.sqrt(profit/max_earning)
        elif profit < 0:
            # Logarithmic scaling for negative profits
            scaled_reward = -np.sqrt(abs(profit/max_earning))
        else:
            # If profit is exactly zero, return zero
            scaled_reward = 0

        return scaled_reward

    def reset(self):
        # Reset statistics
        self.total_earnings = 0


##############################################################
# Strategic RL prosumer agent
##############################################################
class StrategicProsumerAgent(ph.StrategicAgent):

    @dataclass
    class ProsumerSupertype(ph.Supertype):
        capacity: int = 1
        eta: float = 0.23

    def __init__(self, agent_id, mediator_id, data_manager):
        super().__init__(agent_id)

        # Store the ID of the community mediator
        self.mediator_id = mediator_id

        # Store the DataManager
        self.dm: DataManager = data_manager

        # Agent properties
        self.battery_cap: float = 0
        self.charge_rate: float = 1

        # Agent currents
        self.current_load: float = 0 
        self.current_prod: float = 0
        self.current_charge: float = 0
        self.current_supply: float = 0

        self.self_consumption: float = 0
        self.avail_energy: float = 0
        self.surplus_energy: float = 0
        self.current_local_bought: float = 0

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

        # Reward
        self.reward: float = 0

        # = {Buy, BuyCharge, Sell, SellCharge, Charge, No-op}
        self.action_space = gym.spaces.Discrete(6)

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                # Can include type here as well in the future maybe
                "action_mask": gym.spaces.Box(0, 1, shape=(6,), dtype=np.float64),

                "observations": gym.spaces.Box(low=0.0, high=1.0, shape=(13,), dtype=np.float64),
            }
        )

    def buy_power(self, amount):
        if amount > 0:
            buy_amount = round(amount, 2)
            return [(self.mediator_id, BuyBid(self.id, buy_amount))]
        else:
            print("NEGATIVE BUY AMOUNT")


    def sell_power(self, amount):
        if amount > 0:
            return [(self.mediator_id, SellBid(self.id, round(amount, 2)))]
        else:
            print("NEGATIVE SELL AMOUNT")
    
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

    def decode_action(self, ctx: ph.Context, action: np.ndarray):
        #print(action)
        if action == 0:
           # Buy enough power to cover own deficit
            if self.current_supply >= 0:
                self.acc_invalid_actions += 1
                return []
            else:
                return self.buy_power(abs(self.current_supply))
        
        elif action == 1:
            # Buy power to charge and fill possible deficit
            if self.current_supply >= self.max_batt_charge:
                self.acc_invalid_actions += 1
            else:
                deficit = abs(min(self.current_supply, 0))
                buy_amount = self.max_batt_charge + deficit
                return self.buy_power(buy_amount)

        elif action == 2: 
            # Sell own surplus production
            if self.current_supply > 0:
                return self.sell_power(self.current_supply)
            else:
                self.acc_invalid_actions += 1
                return []
            
        elif action == 3:
            # Sell from battery and possible surplus production
            sell_amount = self.max_batt_discharge + self.current_supply
            if sell_amount > 0:
                self.discharge_battery(min(self.max_batt_discharge, sell_amount))
                return self.sell_power(sell_amount)
            else:
                self.acc_invalid_actions += 1
                return []
        
        elif action == 4:
            # Charge battery
            # Can only charge if the agent has positive supply
            if self.current_supply > 0:
                self.charge_battery(self.current_supply)
            else:
                self.acc_invalid_actions += 1
            return []
        
        elif action == 5:
            # If the agent has negative supply, and it has enough charge to cover it, it will do so
            if self.current_supply < 0 and self.max_batt_discharge >= abs(self.current_supply):
                self.self_consumption += abs(self.current_supply)
                self.discharge_battery(abs(self.current_supply))
            # If the agent has negative supply and it does not have enough charge to cover it, this is an invalid action
            elif self.current_supply < 0 and self.max_batt_discharge < abs(self.current_supply):
                self.acc_invalid_actions += 1
                self.curr_invalid_actions = 1000
            # The agent might just have surplus energy and also choose this action, 
            # then it just does not cooperate, but it is a legal action.
            return []


    @ph.agents.msg_handler(ClearedBuyBid)
    def handle_cleared_buybid(self, _ctx: ph.Context, msg: ph.Message):
        # Charge battery if agent bought more than load
        energy_to_charge = msg.payload.buy_amount - abs(self.current_supply)
        if energy_to_charge > 0:
            self.charge_battery(energy_to_charge)
        # Update statistics
        self.current_local_bought += msg.payload.local_amount
        self.acc_local_market_cost += msg.payload.local_cost
        self.acc_grid_market_cost += msg.payload.grid_cost
        self.acc_grid_interactions += 1


    @ph.agents.msg_handler(ClearedSellBid)
    def handle_cleared_sellbid(self, _ctx: ph.Context, msg: ph.Message):
        # Calculate the energy to discharge
        # Update statistics
        self.acc_local_market_coin += msg.payload.local_coin
        self.acc_feedin_coin += msg.payload.feedin_coin
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
        if self.current_supply < 0:
            self.self_consumption += self.current_prod

        # Reset invalid actions
        self.curr_invalid_actions = 0


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

        # Get the public info from the mediator
        grid_price = ctx[self.mediator_id].public_info["grid_price"] / 1.8
        local_price = ctx[self.mediator_id].public_info["local_price"] / 1.8
        feedin_price = ctx[self.mediator_id].public_info["feedin_price"] / 1.8

        # Normalization
        current_load = min(self.current_load / self.all_max_load, 1)
        current_prod = min(self.current_prod / self.all_max_prod, 1)
        current_charge = min(self.current_charge / self.all_max_cap, 1)
        battery_cap = min(self.battery_cap / self.all_max_cap, 1)
        charge_rate = min(self.charge_rate / self.all_max_cap, 1)
        acc_local_coin = min(self.acc_local_market_coin / 10000, 1)
        acc_feedin_coin = min(self.acc_feedin_coin / 10000, 1)
        acc_local_market_cost = min(self.acc_local_market_cost / 10000, 1)
        acc_grid_market_cost = min(self.acc_grid_market_cost / 10000, 1)
        acc_grid_interactions = min(self.acc_grid_interactions / 10000, 1)

        observation = {
            'observations' : np.array([
                grid_price, 
                local_price, 
                feedin_price,
                current_load,
                current_prod,
                current_charge,
                battery_cap,
                charge_rate,
                acc_local_coin,
                acc_feedin_coin,
                acc_local_market_cost,
                acc_grid_market_cost,
                acc_grid_interactions], dtype=np.float64),
            'action_mask' : np.array([buy, buy_charge, sell, sell_batt, charge, noop], dtype=np.float64)
        }

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
        # Normalize reward
        self.reward = min(marginal_utility/20, 1)
        return self.reward

    def reset(self):
        # Reset statistics
        self.acc_local_market_coin = 0
        self.acc_feedin_coin = 0
        self.acc_local_market_cost = 0
        self.acc_grid_market_cost = 0
        self.acc_invalid_actions = 0
        self.acc_grid_interactions = 0
        #self.battery_cap = self.dm.get_agent_battery_capacity(self.id)
        self.battery_cap = 5 * self.type.capacity
        # Get charge rate
        self.charge_rate = self.battery_cap / 4
        # Get demand data for first step
        self.current_load = self.dm.get_agent_demand(self.id, 0)
        # Get production data for first step
        self.current_prod = self.dm.get_agent_production(self.id, 0)*self.type
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

        self.surplus_energy = 0.0
        self.avail_energy = self.current_supply + self.max_batt_discharge
        #
        self.self_consumption = 0.0
        # Normalization factors
        self.all_max_load = self.dm.get_all_maxdemand()
        self.all_max_prod = self.dm.get_all_maxprod()
        self.all_max_cap = 5*3
        # Reward scaling
        self.reward = 0
        self.mean = 0.0
        self.var = 0.0
        self.count = 0
        return super().reset()

##############################################################
# Simple Community Mediator Agent
# Sets a fixed price for the local market
##############################################################
class SimpleCommunityMediator(ph.Agent):#
    "Stategic Community Mediator Agent"

    @dataclass(frozen=True)
    class MediatorView(ph.AgentView):
            """
            We expose a view of the grid price, local market price and feed-in price to the agents.
            """
            public_info: dict

    def __init__(self, agent_id, grid_price, feedin_price, train=False):
        super().__init__(agent_id)

        # Store the current grid price
        self.current_grid_price: float = grid_price
        # Current local price
        self.current_local_price: float = 0
        # Feedin tariff
        self.feedin_price: float = feedin_price
        # Training mode
        self.train = train


    def view(self, neighbour_id=None) -> ph.View:
        """@override
        Create the view for the agents to see the public information.
        """
        return self.MediatorView(
            public_info={
                "grid_price": self.current_grid_price, 
                "local_price": self.current_local_price, 
                "feedin_price": self.feedin_price
            },
        )
        
    def handle_batch(
        self, ctx: ph.Context, batch: Sequence[ph.Message]):
        """@override
        We override the method `handle_batch` to consume all the bid messages
        as one block in order to perform the auction. The batch object contains
        all the messages that were sent to the actor.
        Note:
        -----
        The default logic is to consume each message individually.
        """
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
        """
        Encode and decode buy and sell bids and pass to external market clearing mechanism.

        """
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
        cleared_buy_bids, cleared_sell_bids, fraction, self_sufficiency = Market.market_clearing(
            buy_bids=encoded_buy_bids, 
            sell_bids=encoded_sell_bids,
            local_price=self.current_local_price,
            grid_price=self.current_grid_price,
            feedin_price=self.feedin_price)

        # DECODING
        msgs = []
        # Create messages for the cleared buy bids
        for cleared_buy_bid in cleared_buy_bids:
            buyer_id, buy_amount, local_amount, local_cost, grid_cost = cleared_buy_bid
            msgs.append(
                (
                    buyer_id,
                    ClearedBuyBid(buyer_id, buy_amount, round(local_amount, 2), round(local_cost, 2), round(grid_cost, 2)),
                )
            )
        # Create messages for the cleared sell bids
        for cleared_sell_bid in cleared_sell_bids:
            seller_id, sell_amount, local_income, grid_income = cleared_sell_bid
            msgs.append(
                (
                    seller_id,
                    ClearedSellBid(seller_id, sell_amount, round(local_income, 2), round(grid_income, 2)),
                )
            )

        return msgs
    

    def post_message_resolution(self, ctx: ph.Context) -> None:
        if self.train and ctx.env_view.current_step % 4320 == 0:
            self.current_local_price = random.uniform(self.feedin_price, self.current_grid_price)
    
    def reset(self):
        if self.train:
            self.current_local_price = random.uniform(self.feedin_price, self.current_grid_price)
        return super().reset()

##############################################################
# Simple prosumer agent
##############################################################

class SimpleProsumerAgent(ph.Agent):
    def __init__(self, agent_id, mediator_id, data_manager, greed):
        super().__init__(agent_id)

        # Store the ID of the community mediator
        self.mediator_id = mediator_id

        # Store the DataManager
        self.dm: DataManager = data_manager

        # Agent properties
        self.battery_cap: float = 0
        self.charge_rate: float = 0
        self.greed = greed
        self.type: int = 1

        # Agent currents
        self.current_load: float = 0 
        self.current_prod: float = 0
        self.current_charge: float = 0
        self.current_supply: float = 0
        self.self_consumption: float = 0
        self.avail_energy: float = 0
        self.surplus_energy: float = 0
        self.current_local_bought: float = 0

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


    def generate_messages(self, ctx: ph.Context):

        # Evaluate greediness of agent.
        if self.current_charge >= self.greed*self.battery_cap:
            # If balanced supply, sell what can be discharged from battery
            if self.current_supply == 0:
                return [(self.mediator_id, SellBid(self.id, self.max_batt_discharge))]
            # Cases of negative supply:
            elif self.current_supply < 0:
                # If enough charge to cover, discharge the deficit and sell the rest.
                if self.max_batt_discharge >= abs(self.current_supply):
                    # The below is a subtraction for the selfconsumption
                    energy_to_sell = self.max_batt_discharge - abs(self.current_supply)
                    self.discharge_battery(abs(self.current_supply))
                    if energy_to_sell > 0:
                        return [(self.mediator_id, SellBid(self.id, energy_to_sell))]
                # If not enough charge to cover, discharge what's available and buy the remaining
                elif self.max_batt_discharge < abs(self.current_supply):
                    self.discharge_battery(self.max_batt_discharge)
                    return [(self.mediator_id, BuyBid(self.id, abs(self.current_supply+self.max_batt_discharge)))]

        # In the case of battery charge below threshold
        elif self.current_charge < self.greed*self.battery_cap:
            # If balanced supply, do nothing
            if self.current_supply == 0:
                return []
            # Cases of negative supply:
            elif self.current_supply < 0:
                # If enough charge to cover, discharge the deficit
                if self.max_batt_discharge >= abs(self.current_supply):
                    self.discharge_battery(abs(self.current_supply))
                    return []
                # If not enough charge to cover, discharge what's available and buy the remaining
                elif self.max_batt_discharge < abs(self.current_supply):
                    self.discharge_battery(self.max_batt_discharge)
                    return [(self.mediator_id, BuyBid(self.id, abs(self.current_supply+self.max_batt_discharge)))]
            # Case of positive supply, charge the surplus:
            elif self.current_supply > 0:
                self.charge_battery(self.current_supply)
                return []
            

    @ph.agents.msg_handler(ClearedBuyBid)
    def handle_cleared_buybid(self, _ctx: ph.Context, msg: ph.Message):
        # Charge battery if agent bought more than load
        energy_to_charge = msg.payload.buy_amount - abs(self.current_supply)
        if energy_to_charge > 0:
            self.charge_battery(energy_to_charge)
        # Update statistics
        self.current_local_bought += msg.payload.local_amount
        self.acc_local_market_cost += msg.payload.local_cost
        self.acc_grid_market_cost += msg.payload.grid_cost
        self.acc_grid_interactions += 1


    @ph.agents.msg_handler(ClearedSellBid)
    def handle_cleared_sellbid(self, _ctx: ph.Context, msg: ph.Message):
        # Calculate the energy to discharge based on the current supply and sell amount
        energy_to_discharge = msg.payload.sell_amount - self.current_supply
        if energy_to_discharge > 0:
            self.discharge_battery(energy_to_discharge)
        # Update statistics
        self.acc_local_market_coin += msg.payload.local_coin
        self.acc_feedin_coin += msg.payload.feedin_coin
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
            self.current_prod = self.dm.get_agent_production(self.id, sim_step-1)*self.type
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
        if self.surplus_energy < 0:
            self.self_consumption = self.avail_energy
        else :
            self.self_consumption = self.current_load
        
        
    def reset(self):
        # Reset statistics
        self.acc_local_market_coin = 0.0
        self.acc_feedin_coin = 0.0
        self.acc_local_market_cost = 0.0
        self.acc_grid_market_cost = 0.0
        self.acc_grid_interactions = 0.0
        self.net_loss = 0.0
        self.type = random.randint(1, 3)
        # Get battery capacity
        self.battery_cap = 5*self.type
        # Get charge rate
        self.charge_rate = self.battery_cap / 4
        # Get demand data for first step
        self.current_load = self.dm.get_agent_demand(self.id, 0)
        # Get production data for first step
        self.current_prod = self.dm.get_agent_production(self.id, 0)*self.type
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

