import phantom as ph
import gymnasium as gym
import numpy as np
import pandas as pd
from phantom.types import AgentID
from typing import Iterable, Sequence
from market import Market
from messages import BuyBid, SellBid, ClearedBid, DummyMsg
from datamanager import DataManager
#from datamanager import DataManager

##############################################################
# Generator Agent
##############################################################
class GeneratorAgent(ph.Agent):
    def __init__(self, agent_id, agent_key, exchange_id, data_manager):
        super().__init__(agent_id)

        self.agent_key = agent_key

        # Store the ID of the Exchange that Bids go through
        self.exchange_id = exchange_id

        self.dm: DataManager = data_manager
        # How much capacity can be supplied at current step and price
        self.current_production: float = 0
        self.current_price_pr_MWh: float = 0

        # How much capacity that was sold
        self.supplied_production: float = 0

        # ...and how much capacity per step that was missed due to not bidding low enough.
        self.surplus_production: float = 0

    def generate_messages(self, ctx: ph.Context):
        if not self.current_production == 0:
            return [(self.exchange_id, SellBid(self.id, self.current_production, self.current_price_pr_MWh))]
        
    @ph.agents.msg_handler(ClearedBid)
    def handle_cleared_bid(self, _ctx: ph.Context, msg: ph.Message):
        self.supplied_production += msg.payload.mwh
        #logger.debug("Generator Agent %s supplies: %s to %s at price %s", self.id, msg.payload.mwh, msg.payload.buyer_id, msg.payload.price)

    def pre_message_resolution(self, ctx: ph.Context):
        # Reset statistics before message resolution
        self.supplied_production = 0
        self.missed_production = 0

    def post_message_resolution(self, ctx: ph.Context):
        # Compute how much capacity was not sold
        self.surplus_production = self.current_production - self.supplied_production
        # Get capacity and price data for NEXT step except for at last step
        self.current_production = self.dm.get_agent_production(ctx.env_view.current_step, self.agent_key)
        self.current_price_pr_MWh = self.dm.get_spot_price(ctx.env_view.current_step)
        
    def reset(self):
        # Reset statistics
        self.supplied_production = 0
        self.missed_production = 0
        # Load demand data for first step
        self.current_production = self.dm.get_agent_production(0, self.agent_key)
        # Get price data for first step
        self.current_price_pr_MWh = self.dm.get_spot_price(0)

##############################################################
# Strategic RL Generator agent 
# Does not learn bidding! 
# Learns to adjust current production, sets price from marginal cost
##############################################################
class StrategicGeneratorAgent(ph.StrategicAgent):
    def __init__(self, agent_id, agent_key, exchange_id, data_manager, capacity: float, min_load: float, start_up: float, ramp_rate: float, marginal_cost: float):
        super().__init__(agent_id)

        # Agent key for data retrieval
        self.agent_key = agent_key
        # Store the ID of the Exchange that Bids go through
        self.exchange_id = exchange_id
        # Store the DataManager
        self.dm: DataManager = data_manager
        # The plant capacity
        self.capacity = capacity
        # Minimum load
        self.min_load = min_load
        # Start up time
        self.start_up = start_up
        # The max increase in production per step (MWh/hour)
        self.ramp_rate = ramp_rate
        # Cost of production
        self.marginal_cost: float = marginal_cost

        # How much the plant is currently producing
        self.current_production: float = 0
        
        # Include demand as observation at some point?

        # Total earnings
        self.total_earnings: float = 0
        # How much the plant is currently earning
        self.current_earnings: float = 0
        
        # Include forecast demand data at some point?
        # = [Month, Date, Day, Hour, Cleared Price, Current Production]
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(6,))

        # = [Increase in production]
        # The action space is the increase in production per step, normalized
        # X / self.ramp_rate = Action
        #self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))

        # = 1 = Increase production, 0 = maintain production, -1 = decrease production
        self.action_space = gym.spaces.Discrete(3, start=-1)


    # Decode actions is the first method that is called in a step
    def decode_action(self, ctx: ph.Context, action):
        # Adjust production based on ramp rate and capacity limits
        delta_production = action * self.ramp_rate
        #price = action[1] * ctx.env_view.MAX_BID_PRICE
        
        # Ensure the new production is within valid bounds (min_load <= current_production <= capacity)
        self.current_production = max(self.min_load, min(self.capacity, self.current_production + delta_production))
        
        # Generate a sell bid for the new production level
        return [(self.exchange_id, SellBid(self.id, self.current_production, self.marginal_cost))]

    # 2nd step in a step
    def pre_message_resolution(self, ctx: ph.Context):
        self.satisfied_demand = 0
        self.missed_demand = 0
        self.current_earnings = 0

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
# Consumer Agent
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
# Strategic RL customer agent
##############################################################
class StrategicConsumerAgent(ph.StrategicAgent):
    def __init__(self, agent_id, agent_key, exchange_id, data_manager):
        super().__init__(agent_id)

        # Store the ID of the Exchange that Bids go through
        self.exchange_id = exchange_id

        # Store the DataManager
        self.dm: DataManager = data_manager

        # How much demand the customer has at current step.
        self.current_demand: float = 0
        self.current_valuation_pr_MWh: float = 0

        # How much demand was satisfied at current time step.
        self.satisfied_demand: float = 0

        # ...and how much demand per step that was missed due to not bidding high enough.
        self.missed_demand: float = 0

        # = [Demand, Satisfied Demand, Missed Demand, Month, Date, Day, Hour, Cleared Price]
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(7,))

        # = [Bidding price]
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,))


    def decode_action(self, ctx: ph.Context, action: np.ndarray):
        self.current_valuation_pr_MWh = action[0] * self.MAX_BID_PRICE
        # We perform this action by sending a BuyBid to the Exchange.
        return [(self.exchange_id, BuyBid(self.id, self.current_demand, self.current_valuation_pr_MWh))]


    def pre_message_resolution(self, ctx: ph.Context):
        self.satisfied_demand = 0
        self.missed_demand = 0


    @ph.agents.msg_handler(ClearedBid)
    def handle_cleared_bid(self, _ctx: ph.Context, msg: ph.Message):
        self.satisfied_demand += msg.payload.mwh


    def post_message_resolution(self, ctx: ph.Context):
        # Compute missed demand stat
        self.missed_demand = max(self.current_demand - self.satisfied_demand, 0)

        # Get demand and price data for NEXT step except for at last step
        self.current_demand = self.dm.get_agent_demand(ctx.env_view.current_step, self.agent_key)
        self.current_valuation_pr_MWh = self.dm.get_spot_price(ctx.env_view.current_step)

    def encode_observation(self, ctx: ph.Context):
        month = ctx.env_view.current_month
        day = ctx.env_view.current_day
        hour = ctx.env_view.current_hour
        # Clip the cleared price to be 0 in case of negative prices
        cleared_price = max(ctx.env_view.current_price, 0) 
        self.MAX_DEMAND = ctx.env_view.CUST_MAX_DEMAND
        self.MAX_BID_PRICE = ctx.env_view.MAX_BID_PRICE

        return np.array(
            [
                self.current_demand / self.MAX_DEMAND,
                self.satisfied_demand / self.MAX_DEMAND,
                self.missed_demand / self.MAX_DEMAND,
                float(month / 12),
                float(day / 7),
                float(hour / 23),
                float(cleared_price / self.MAX_BID_PRICE)
            ],
            dtype=np.float32,
        )

    def compute_reward(self, ctx: ph.Context) -> float:
        # Reward for satisfying demand
        normalized_satisfied_demand = self.satisfied_demand / ctx.env_view.CUST_MAX_DEMAND
        normalized_missed_demand = self.missed_demand / ctx.env_view.CUST_MAX_DEMAND
        normalized_valuation = self.current_valuation_pr_MWh / ctx.env_view.MAX_BID_PRICE

        self.current_valuation_pr_MWh
        # Reward for satisfying demand
        normalized_satisfied_demand = self.satisfied_demand / ctx.env_view.CUST_MAX_DEMAND
        normalized_missed_demand = self.missed_demand / ctx.env_view.CUST_MAX_DEMAND
        normalized_valuation = self.current_valuation_pr_MWh / ctx.env_view.MAX_BID_PRICE

        cleared_price = max(ctx.env_view.current_price, 0) 
        normalized_cleared_price = cleared_price / ctx.env_view.MAX_BID_PRICE

        # Calculate the distance between the current valuation and the cleared price
        price_distance = abs(normalized_valuation - normalized_cleared_price)

        # Reward for being close to the cleared price
        price_distance_penalty = price_distance ** 0.5  # Square root to penalize larger distances more

        # Final reward
        return normalized_satisfied_demand - normalized_missed_demand - price_distance_penalty


    def reset(self):
        # Reset statistics
        self.satisfied_demand = 0
        self.missed_demand = 0
        # Get demand data for first step
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

