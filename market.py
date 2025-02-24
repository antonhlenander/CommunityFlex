class Market():

    def market_clearing(buy_bids, sell_bids, local_price, grid_price, feedin_price, local_tariff):
        """ Buy bids: list of tuples (buyer_id, buy_amount)
            Sell bids: list of tuples (seller_id, sell_amount) 
            Cleared buy bids: list of tuples (buyer_id, buy_amount, local_amount, local_cost, grid_cost)
            Cleared sell bids: list of tuples (seller_id, sell_amount, local_amount, local_income, grid_income)
        """
        

        total_demand = sum(bid[1] for bid in buy_bids)
        total_supply = sum(bid[1] for bid in sell_bids)

        cleared_buy_bids = []
        cleared_sell_bids = []

        fraction = 0.0

        # This has been modified for the Baseline case where agents cannot sell to each other!!!

        # If demand greater than or equal to supply
        # The grid is used to meet the remaining demand
        for bid in buy_bids:
            buyer_id = bid[0]
            buy_amount = bid[1]
            local_amount = 0
            grid_amount = buy_amount
            prosumer_cost = buy_amount * local_price
            mediator_cost = 0
            cleared_buy_bids.append((buyer_id, buy_amount, local_amount, grid_amount, prosumer_cost, mediator_cost))
        for bid in sell_bids:
            # All sell bids were bought locally
            seller_id = bid[0]
            sell_amount = bid[1]
            local_amount = 0
            grid_amount = sell_amount
            prosumer_income = sell_amount * feedin_price
            mediator_income = 0
            cleared_sell_bids.append((seller_id, sell_amount, local_amount, grid_amount, prosumer_income, mediator_income))

        return cleared_buy_bids, cleared_sell_bids, total_demand, total_supply

# Example check

# buy_bids = [("G1", 150), ("G2", 50), ("G3", 500)] # 700
                    

# sell_bids = [("D1", 250), ("D2", 300), ("D3", 120)] # 670

# cleared_buy_bids, cleared_sell_bids, fraction, self_sufficient = Market.market_clearing(buy_bids, sell_bids, 1.8, 3.0, 0.5)

# print("Local Price: 1.8, Grid Price: 3.0, Feedin Price: 0.5")
# print("Self Sufficiency:", self_sufficient)
# print("Fraction:", fraction)
# print("Cleared Buy Bids:", cleared_buy_bids)
# print("Cleared Sell Bids:", cleared_sell_bids)

