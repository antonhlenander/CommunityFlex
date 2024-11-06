import phantom as ph
# Message Payloads
##############################################################


@ph.msg_payload()
class BuyBid:
    """
    A bid to buy a certain amount of MWh at a certain price.

    Attributes:
    -----------
    customer_id str:    customer id
    size (int):         the amount of MWh
    price (float):      price of bid
    time:               possibly timestamp for bid?
    """

    buyer_id: str
    mwh: float
    price: float
    

@ph.msg_payload()
class SellBid:
    """
    A bid to sell a certain amount of MWh at a certain price.

    Attributes:
    -----------
    seller_id (str):    seller id
    size (int):         the amount of MWh
    price (float):      price of bid
    time:               possibly timestamp for bid?
    """
    
    seller_id: str
    mwh: float
    price: float
    

@ph.msg_payload()
class ClearedBid:
    """
    A cleared bid designating the amount of MWh at which price
    and the id of buyer and seller.

    Attributes:
    -----------
    size (int):         the amount of MWh
    price (float):      price of bid
    customer_id (int)
    seller_id (int):    customer id
    time:               possibly timestamp for bid?
    """

    buyer_id: str
    seller_id: str
    mwh: float
    price: float

@ph.msg_payload()
class DummyMsg:
    """
    Empty message
    """

    msg: str