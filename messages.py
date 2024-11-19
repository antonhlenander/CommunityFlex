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
    """
    A cleared bid designating the amount of kwh at which price
    and the id of buyer and seller.

    Attributes:
    -----------
    amount (float):     the amount of kwh
    price (float):      price of bid
    time:               possibly timestamp for bid?
    local_cost:         cost of the amount of local energy bought
    grid_cost:          cost of the amount of grid energy bought
    """

    buyer_id: str
    buy_amount: float
    local_amount: float
    local_cost: float
    grid_cost: float


@ph.msg_payload()
class ClearedSellBid:
    """
    A cleared bid designating the amount of kwh at which price
    and the id of buyer and seller.

    Attributes:
    -----------
    amount (kwh):       the amount of kwh
    price (float):      price of bid
    seller_id (int):    customer id
    time:               possibly timestamp for bid?
    """

    seller_id: str
    sell_amount: float
    local_coin: float
    feedin_coin: float

@ph.msg_payload()
class DummyMsg:
    """
    Empty message
    """

    msg: str