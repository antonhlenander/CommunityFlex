class Market():

    def market_clearing(supply_bids, demand_bids):
        
        # Step 1: Sort bids by price
        supply_bids.sort(key=lambda x: x[2])  # Sort by price ascending (cheapest first)
        demand_bids.sort(key=lambda x: x[2], reverse=True)  # Sort by price descending (most expensive first)

        # Step 2: Matching process
        cleared_bids = []
        clearing_price: float = None
        i, j = 0, 0  # indices for supply and demand lists

        # Step 3: Traverse and match bids
        while i < len(supply_bids) and j < len(demand_bids):
            supply_id, supply_mwh, supply_price = supply_bids[i]
            demand_id, demand_mwh, demand_price = demand_bids[j]

            # If the demand price is less than the supply price, stop clearing (no match possible)
            if demand_price < supply_price:
                break

            # Determine the amount to match (min of available supply and demand)
            match_mwh = min(supply_mwh, demand_mwh)

            # Record the match and the increasing clearing price
            clearing_price = min(supply_price, demand_price)  # Clearing price is final at last possible match
            cleared_bids.append((supply_id, demand_id, match_mwh)) # Clearing price is added after the loop

            # Update the remaining supply and demand
            supply_bids[i] = (supply_id, supply_mwh - match_mwh, supply_price)
            demand_bids[j] = (demand_id, demand_mwh - match_mwh, demand_price)

            # Move to the next bid if fully fulfilled
            if supply_bids[i][1] == 0:
                i += 1
            if demand_bids[j][1] == 0:
                j += 1


        # Add the final clearing price to all cleared bid tuples
        cleared_bids = [tuple + (clearing_price,) for tuple in cleared_bids]

        return cleared_bids, clearing_price


# Example Usage (id, MWh, price)
# As example in Pierre Pinson lectures


# supply_bids = [("G1", 120, 0), ("G2", 50, 0), ("G3", 200, 15.), 
#             ("G4", 400, 30), ("G5", 60, 32.5), ("G6", 50, 34),
#             ("G7", 60, 36), ("G8", 100, 37.5), ("G9", 70, 39),
#             ("G10", 50, 40), ("G11", 70, 60), ("G12", 45, 70),
#             ("G13", 50, 100), ("G14", 60, 150), ("G15", 50, 200)
#             ]
                    
# supply_bids = [(bid_id, mwh, float(price)) for bid_id, mwh, price in supply_bids]


# demand_bids = [("D1", 250, 200), ("D2", 300, 110), ("D3", 120, 100), 
#             ("D4", 80, 90), ("D5", 40, 85), ("D6", 70, 75),
#             ("D7", 60, 65), ("D8", 45, 40), ("D9", 30, 38),
#             ("D10", 35, 31), ("D11", 25, 24), ("D12", 10, 16),
#                 ]

# demand_bids = [(bid_id, mwh, float(price)) for bid_id, mwh, price in demand_bids]

# cleared_bids, clearing_price = Market.market_clearing(supply_bids, demand_bids)

# print("Cleared Bids:", cleared_bids)
# print("Market Clearing Price:", clearing_price)
