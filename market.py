class Market():

    def market_clearing(buy_bids, sell_bids, local_price, grid_price, feedin_price):
        """ Buy bids: list of tuples (buyer_id, buy_amount)
            Sell bids: list of tuples (seller_id, sell_amount) 
            Cleared buy bids: list of tuples (buyer_id, buy_amount, local_cost, grid_cost)
            Cleared sell bids: list of tuples (seller_id, sell_amount, local_income, grid_income)
        """
        

        total_demand = sum(bid[1] for bid in buy_bids)
        total_supply = sum(bid[1] for bid in sell_bids)

        cleared_buy_bids = []
        cleared_sell_bids = []

        fraction = 0.0
        self_sufficient = False

        # Check if supply exceeds demand
        # TODO: Edge cases where supply or demand is zero?
        if total_supply <= total_demand:
            fraction = total_supply / total_demand
            self_sufficient = True if fraction == 1.0 else False
            for bid in buy_bids:
                local_cost = bid[1] * fraction * local_price
                grid_cost = bid[1] * (1-fraction) * grid_price
                cleared_buy_bids.append((bid[0], bid[1], local_cost, grid_cost))
            for bid in sell_bids:
                cleared_sell_bids.append((bid[0], bid[1], bid[1]*local_price, 0))

        # If supply exceeds demand
        elif total_supply > total_demand:
            fraction = total_demand / total_supply
            self_sufficient = True
            for bid in buy_bids:
                # All buy bids supplied with local energy
                cleared_buy_bids.append((bid[0], bid[1], bid[1]*local_price, 0))
            for bid in sell_bids:
                local_income = bid[1] * fraction * local_price
                grid_income = bid[1] * (1-fraction) * feedin_price
                cleared_sell_bids.append((bid[0], bid[1], local_income, grid_income))

        return cleared_buy_bids, cleared_sell_bids, fraction, self_sufficient

# Example check

# buy_bids = [("G1", 150), ("G2", 50), ("G3", 500)] # 700
                    

# sell_bids = [("D1", 250), ("D2", 300), ("D3", 120)] # 670

# cleared_buy_bids, cleared_sell_bids, fraction, self_sufficient = Market.market_clearing(buy_bids, sell_bids, 1.8, 3.0, 0.5)

# print("Local Price: 1.8, Grid Price: 3.0, Feedin Price: 0.5")
# print("Self Sufficiency:", self_sufficient)
# print("Fraction:", fraction)
# print("Cleared Buy Bids:", cleared_buy_bids)
# print("Cleared Sell Bids:", cleared_sell_bids)

