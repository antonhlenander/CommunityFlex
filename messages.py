import phantom as ph
# Message Payloads
##############################################################


@ph.msg_payload()
class BuyBid:
    """
    A bid to buy a certain amount of kwh.

    Attributes:
    -----------
    customer_id str:    customer id
    e_amount (float):       the amount of kwh
    time:               possibly timestamp for bid?
    """

    buyer_id: str
    buy_amount: float
    

@ph.msg_payload()
class SellBid:
    """
    A bid to sell a certain amount of kwh.

    Attributes:
    -----------
    seller_id (str):    seller id
    e_amount (float):     the amount of kwh
    time:               possibly timestamp for bid?
    """
    
    seller_id: str
    sell_amount: float
    

@ph.msg_payload()
class ClearedBuyBid:
    buyer_id: str
    buy_amount: float
    local_amount: float
    grid_amount: float
    prosumer_cost: float # Dynamic price * amount
    mediator_cost: float


@ph.msg_payload()
class ClearedSellBid:
    seller_id: str
    sell_amount: float
    local_amount: float
    grid_amount: float
    prosumer_income: float # Local price 
    mediator_income: float # Spot price - export tariff

@ph.msg_payload()
class PriceUpdate:
    """
    Update the price of electricity.

    Attributes:
    -----------
    new_price (float):  new price of electricity
    """
    current_price: float
    current_feedin_price: float
   
@ph.msg_payload()
class DummyMsg:
    """
    Empty message
    """

    msg: str