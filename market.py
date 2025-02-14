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

        # If demand greater than or equal to supply
        # The grid is used to meet the remaining demand
        if total_supply <= total_demand:
            fraction = total_supply / total_demand
            for bid in buy_bids:
                buyer_id = bid[0]
                buy_amount = bid[1]
                local_amount = buy_amount * fraction
                grid_amount = buy_amount * (1-fraction)
                prosumer_cost = buy_amount * local_price
                mediator_cost = (grid_amount * grid_price) + (local_amount * local_tariff)
                cleared_buy_bids.append((buyer_id, buy_amount, local_amount, grid_amount, prosumer_cost, mediator_cost))
            for bid in sell_bids:
                # All sell bids were bought locally
                seller_id = bid[0]
                sell_amount = bid[1]
                local_amount = sell_amount
                grid_amount = 0
                prosumer_income = sell_amount * local_price
                mediator_income = 0
                cleared_sell_bids.append((seller_id, sell_amount, local_amount, grid_amount, prosumer_income, mediator_income))

        # If supply exceeds demand
        # The excess energy is sold to the grid
        elif total_supply > total_demand:
            fraction = total_demand / total_supply
            for bid in buy_bids:
                # All buy bids supplied with purely local energy
                buyer_id = bid[0]
                buy_amount = bid[1]
                local_amount = buy_amount
                grid_amount = 0
                prosumer_cost = buy_amount * local_price
                mediator_cost = local_amount * local_tariff
                cleared_buy_bids.append((buyer_id, buy_amount, local_amount, grid_amount, prosumer_cost, mediator_cost))
            for bid in sell_bids:
                seller_id = bid[0]
                sell_amount = bid[1]
                local_amount = sell_amount * fraction
                grid_amount = sell_amount * (1-fraction)
                prosumer_income = sell_amount * local_price
                mediator_income = grid_amount * feedin_price
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

