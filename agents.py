from dataclasses import dataclass
import phantom as ph
import gymnasium as gym
import numpy as np
import pandas as pd
from phantom.types import AgentID
from typing import Iterable, Sequence, Dict
from messages import BuyBid, SellBid, ClearedSellBid, ClearedBuyBid, DummyMsg
from datamanager import DataManager
#from datamanager import DataManager


##############################################################
# Strategic Community Mediator Agent
# Learns a dynamic pricing strategy for the local market
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
        # Feedin tariff
        self.feedin_tariff: float = 0

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
        return self.StrategicCommunicatorView(
            public_info={
                "grid_price": self.grid_price, 
                "local_price": self.local_price, 
                "feedin_price": self.feedin_price
            },
        )

    # Decode actions is the first method that is called in a step
    def decode_action(self, ctx: ph.Context, action):
        # Set the local price
        self.current_local_price = float(action) * self.current_grid_price / 10

    # 2nd step in a step
    def pre_message_resolution(self, ctx: ph.Context):
        self.satisfied_demand = 0
        self.missed_demand = 0
        self.current_earnings = 0


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

        return msgs

    # 3rd step in a step
    @ph.agents.msg_handler(ClearedBid)
    def handle_cleared_bid(self, _ctx: ph.Context, msg: ph.Message):
        #self.satisfied_demand += msg.payload.mwh
        self.current_earnings += msg.payload.mwh * msg.payload.price
        self.total_earnings += self.current_earnings


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
    def __init__(self, agent_id, agent_key, mediator_id, data_manager, battery: Dict[str, float], production: Dict[str, float]):
        super().__init__(agent_id)

        # Store the ID of the community mediator
        self.mediator = mediator_id

        # Store the DataManager
        self.dm: DataManager = data_manager

        # Agent properties
        self.battery_cap = battery["cap"]
        self.charge_rate = battery["charge_rate"]

        # Agent currents
        self.current_load: float = 0 
        self.current_prod: float = 0
        self.current_charge: float = 0
        self.current_supply: float = 0

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

        # = [Grid price, Local market price, Feed-in price, 
        #   Load, PV supply, State of charge, Battery capacity,
        #   Battery charge/discharge limit, Acc. local market coin, Acc. feed-in coin,
        #   Acc. lokal market cost, Acc. grid market cost, Acc. number of invalid actions,
        #   Acc. number of grid interactions]
        # self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(11,))

        # = {Buy, Sell, Charge, No-op}
        self.action_space = gym.spaces.Discrete(4)

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                # Can include type here as well in the future maybe
                "public_info": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(3,), dtype=np.float64
                ),
                "private_info": gym.spaces.Box(low=0.0, high=1.0, shape=(11,), dtype=np.float64),
                # "action_mask": gym.spaces.Discrete(4),
                # "user_age": gym.spaces.Box(low=0.0, high=100., shape=(1,), dtype=np.float64),
                # "user_zipcode": gym.spaces.Box(low=0.0, high=99999., shape=(1,), dtype=np.float64),
            }
        )

    def buy_power(self, ctx: ph.Context):
        # Check if valid action!
        # Down the line we can just make the action invalid with action masking

        # Cannot buy power if agent has more supply than it can charge
        if self.current_supply >= self.max_batt_charge:
            self.acc_invalid_actions += 1
            return []
        # Can only buy the amount of power that agent cannot supply itself, 
        # bounded by the max amount agent can charge
        elif self.current_supply > 0 and self.current_supply < self.max_batt_charge:
            buy_amount = self.max_batt_charge - self.current_supply
            return [(self.mediator, BuyBid(self.id, buy_amount))]
        # If negative own supply, agent buys the amount it can charge + its deficit in own supply
        else:
            # This case will be if supply is negative or zero
            # i.e. the subtraction is an addition
            buy_amount = self.max_batt_charge - self.current_supply
            return [(self.mediator, BuyBid(self.id, buy_amount))]

    # TODO: this is not done!
    def sell_power(self):
        # Check if valid action!
        # Down the line we can just make the action invalid with action masking
        # If agent has no excess own supply, it cannot sell
        if self.current_supply <= 0:
            self.acc_invalid_actions += 1
            return []
        # If excess supply, agent bids that + what can be discharged from the battery
        else:
            sell_amount = self.current_supply + self.max_batt_discharge
            return [(self.mediator, SellBid(self.id, sell_amount))]
    
    # Charge or decharge battery by a certain amount
    def charge_battery(self, amount):
        # Only charge the battery by a positive amount capped by charge rate and if the battery is not full.
        if amount > 0 and self.current_charge < self.battery_cap:
            self.current_charge =+ amount
        else:
            self.acc_invalid_actions =+ 1

    def discharge_battery(self, amount):
        self.current_charge =- min(self.max_batt_discharge, amount)
        self.current_charge = max(self.current_charge, 0)

    def decode_action(self, ctx: ph.Context, action: np.ndarray):
        if action == 0:
            # Buy power
           return self.buy_power()
        elif action == 1:
            # Sell power
            return self.sell_power()
        elif action == 2:
            # Charge battery
            self.charge_battery(self.max_batt_charge)
            return []
        else:
            # If the agent has negative supply, and it has enough charge to cover it, it will do so
            if self.current_supply < 0 and self.max_batt_discharge >= abs(self.current_supply):
                self.current_charge =- min(self.max_batt_discharge, abs(self.current_supply))
            # If the agent has negative supply and it does not have enough charge to cover it, this is an invalid action
            elif self.current_supply < 0 and self.max_batt_discharge < abs(self.current_supply):
                self.acc_invalid_actions =+ 1
            # The agent might just have surplus energy and also choose this action, 
            # then it just does not cooperate, but it is a legal action.
            return []

    def pre_message_resolution(self, ctx: ph.Context):
        # This is after the agent has performed its action i.e. sent messages.
        # But it is before the agent has received the result of its actions!
        # Placeholder code:
        age = ctx[self.mediator_id].public_info[self._current_user_id][
            "age"
        ]

    @ph.agents.msg_handler(ClearedBuyBid)
    def handle_cleared_buybid(self, _ctx: ph.Context, msg: ph.Message):
        # Charge battery if agent bought more than load
        energy_to_charge = msg.payload.buy_amount - self.current_load
        if energy_to_charge > 0:
            self.charge_battery(energy_to_charge)
        # Update statistics
        self.acc_local_market_cost =+ msg.payload.local_cost
        self.acc_grid_market_cost =+ msg.payload.grid_cost
        self.acc_grid_interactions =+ 1


    @ph.agents.msg_handler(ClearedSellBid)
    def handle_cleared_sellbid(self, _ctx: ph.Context, msg: ph.Message):
        # Discharge battery if agent sold more than producing
        energy_to_discharge = msg.payload.sell_amount - self.current_prod
        if energy_to_discharge > 0:
            self.discharge_battery(energy_to_discharge)
        # Update statistics
        self.acc_local_market_coin =+ msg.payload.local_coin
        self.acc_feedin_coin =+ msg.payload.feedin_coin
        self.acc_grid_interactions =+ 1

    def post_message_resolution(self, ctx: ph.Context):
        # We update everything after the messages have been resolved
        # This is where the the values are updated for the observation in next step
        
        # Integer division taking into account odd and even steps
        step = (ctx.env_view.current_step + 1) // 2
        # Convert to hours since demand profile is just 24 hours
        hour = step % 24
        # Update current load
        self.current_load = self.dm.get_agent_demand(hour, self.id)
        # Update production
        self.current_prod = self.dm.get_agent_production(hour, self.id)
        # Update current own supply
        self.current_supply = self.current_prod - self.current_load
        # Update battery constraints
        self.remain_batt_cap = self.battery_cap - self.current_charge
        self.max_batt_charge = min(self.remain_batt_cap, self.charge_rate)
        self.max_batt_discharge = min(self.current_charge, self.charge_rate)


    def encode_observation(self, ctx: ph.Context):
        # Encoding of observations for the action in next step
        # = [Grid price, Local market price, Feed-in price, 
        #   Load, PV supply, State of charge, Battery capacity,
        #   Battery charge/discharge limit, Acc. local market coin, Acc. feed-in coin,
        #   Acc. lokal market cost, Acc. grid market cost, Acc. number of invalid actions,
        #   Acc. number of grid interactions]

        # Get the public info from the mediator
        grid_price = ctx[self.mediator_id].public_info["grid_price"]
        local_price = ctx[self.mediator_id].public_info["local_price"]
        feedin_price = ctx[self.mediator_id].public_info["feedin_price"]

        # Possible we need to normalize!
        observation = {
            'public_info' : np.array([grid_price, local_price, feedin_price], dtype=np.float64),
            'private_info' : np.array([
                self.current_demand,
                self.current_prod,
                self.current_charge,
                self.battery_cap,
                self.charge_rate,
                self.acc_local_market_coin,
                self.acc_feedin_coin,
                self.acc_local_market_cost,
                self.acc_grid_market_cost,
                self.acc_invalid_actions,
                self.acc_grid_interactions],  dtype=np.float64
            ),
            #'action_mask' : np.array([1, 0, 1, 1], dtype=np.float64)  # Actions 1 is invalid
        }

        return observation

    def compute_reward(self, ctx: ph.Context) -> float:
        # Reward for satisfying demand

        # Final reward
        return normalized_satisfied_demand - normalized_missed_demand - price_distance_penalty


    def reset(self):
        # Reset statistics
        self.acc_local_market_coin = 0
        self.acc_feedin_coin = 0
        self.acc_local_market_cost = 0
        self.acc_grid_market_cost = 0
        self.acc_invalid_actions = 0
        self.acc_grid_interactions = 0
        # Get demand data for first step
        self.current_demand = self.dm.get_agent_demand(0, self.id)
        # Get production data for first step
        self.current_prod = self.dm.get_agent_production(0, self.id)
        # Reset battery charge
        self.current_charge = self.battery_cap / 2


##############################################################
# Simple prosumer agent
##############################################################
class ConsumerAgent(ph.Agent):
    def __init__(self, agent_id, agent_key, exchange_id, data_manager):
        super().__init__(agent_id)

        self.agent_key = agent_key
        self.exchange_id = exchange_id

        self.dm: DataManager = data_manager

        self.current_demand: float = 0
        self.current_valuation_pr_MWh: float = 0

        self.satisfied_demand: float = 0
        self.missed_demand: float = 0

    def pre_message_resolution(self, ctx: ph.Context):
        # Reset statistics before message resolution
        self.satisfied_demand = 0
        self.missed_demand = 0

    def generate_messages(self, ctx: ph.Context):
        return [(self.exchange_id, BuyBid(self.id, self.current_demand, self.current_valuation_pr_MWh))]
    
    @ph.agents.msg_handler(ClearedBid)
    def handle_cleared_bid(self, _ctx: ph.Context, msg: ph.Message):
        self.satisfied_demand += msg.payload.mwh
     
    def post_message_resolution(self, ctx: ph.Context):
        # Compute missed demand stat
        self.missed_demand = self.current_demand - self.satisfied_demand

        # Get demand and price data for NEXT step except for at last step
        self.current_demand = self.dm.get_agent_demand(ctx.env_view.current_step, self.agent_key)
        self.current_valuation_pr_MWh = self.dm.get_spot_price(ctx.env_view.current_step)
     
    def reset(self):
        # Reset statistics
        self.satisfied_demand = 0
        self.missed_demand = 0
        # Load demand data for first step
        self.current_demand = self.dm.get_agent_demand(0, self.agent_key)
        # Get price data for first step
        self.current_valuation_pr_MWh = self.dm.get_spot_price(0)


##############################################################
# Exchange Agent
##############################################################

class ExchangeAgent(ph.Agent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id)

        # The current price of the market
        self.clearing_price: float = 0

    @ph.agents.msg_handler(BuyBid)
    def handle_buy_bid(self, ctx: ph.Context, message: ph.Message):
        # Handle a buy bid
        return

    @ph.agents.msg_handler(SellBid)
    def handle_sell_bid(self, ctx: ph.Context, message: ph.Message):
        # Handle a sell bid
        return
    
    @ph.agents.msg_handler(DummyMsg)
    def handle_sell_bid(self, ctx: ph.Context, message: ph.Message):
        # Handle a dummy msg
        return

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
        cleared_bids, clearing_price = Market.market_clearing(supply_bids=encoded_sell_bids, demand_bids=encoded_buy_bids)
        # Update clearing price
        self.clearing_price = clearing_price
        # DECODING
        msgs = []

        for cleared_bid in cleared_bids:
            seller_id, buyer_id, mwh, price = cleared_bid
            decoded_cleared_bid = ClearedBid(seller_id=seller_id, buyer_id=buyer_id, mwh=mwh, price=price)
            # Create message for both seller and buyer
            msg1 = (seller_id, decoded_cleared_bid)
            msg2 = (buyer_id, decoded_cleared_bid)
            #logger.debug("Cleared bid between: %s and %s for %s MWh at cost: %s", seller_id, buyer_id, mwh, price)
            msgs.extend((msg1, msg2))  

        return msgs
    
    def reset(self):
        self.clearing_price = 0


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

