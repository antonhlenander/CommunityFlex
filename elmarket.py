import sys

import matplotlib.pyplot as plt
import phantom as ph
import pandas as pd
import numpy as np
from agents import DummyAgent, ExchangeAgent, GeneratorAgent, ConsumerAgent, StrategicConsumerAgent
from elmarket_env import EL_Clearing_Env

##############################################################
# PATHS
##############################################################

consumption_path = "consumption_data.csv"

production_path = "production_data.csv"

price_path = "price_data.csv"

##############################################################
# MODIFY CONSUMPTION DATA
##############################################################

print('Loading consumption data...')
# Load CSV file
consumption_data = pd.read_csv(consumption_path, sep=',')

def sum_consumption_by_category(data, category):
    """Sum consumption data by HourDK for a specific HousingCategory."""
    filtered_data = data[data['HousingCategory'] == category]
    # Group by 'HourDK' and sum the 'ConsumptionkWh' values
    filtered_data['ConsumptionkWh'] = filtered_data.groupby('HourDK')['ConsumptionkWh'].transform('sum')
    # Drop duplicates based on 'HourDK' and return only the 'ConsumptionkWh' values
    filtered_data = filtered_data.drop_duplicates(subset='HourDK').reset_index(drop=True)
    # Optionally, only return the 'ConsumptionkWh' values
    # filtered_data = filtered_data['ConsumptionkWh'].values
    return filtered_data

# Define the categories
categories = [
    'Erhverv', 
    'Etageejendom', 
    'Fritidshuse', 
    'Parcel- og rækkehuse',
    'Andet',
    'Ukendt'
]

print('Summing consumption by category...')
# Get consumption for each category
# NOTE: Industrial consumption includes public buildings.
# Specification of public consumption is available at:
# www.energidataservice.dk/tso-electricity/ConsumptionIndustry
industrial_consumption = sum_consumption_by_category(consumption_data, categories[0])
private_consumption1 = sum_consumption_by_category(consumption_data, categories[1])
private_consumption2 = sum_consumption_by_category(consumption_data, categories[2])
private_consumption3 = sum_consumption_by_category(consumption_data, categories[3])
other_consumption = sum_consumption_by_category(consumption_data, categories[4])
unknown_consumption = sum_consumption_by_category(consumption_data, categories[5])


##############################################################
# MODIFY PRODUCTION DATA
##############################################################

# Mean loss for transmission and distribution
MEAN_LOSS = 1 - 0.055

# Load CSV file
production_data = pd.read_csv(production_path, sep=',')

# Replace commas with periods in the columns and convert to float
# Assuming production_data is already loaded as a DataFrame
# Apply the operation to every row from the 4th column and up
production_data.iloc[:, 3:] = production_data.iloc[:, 3:].applymap(lambda x: float(str(x).replace(',', '.')))

# Extract the columns that will be used as index columns
index_columns = production_data.iloc[:, :3].copy()

# Wind power data from all offshore plants
offshore_wind_data = index_columns.copy()
offshore_wind_data['Capacity'] = (production_data['OffshoreWindGe100MW_MWh'] + production_data['OffshoreWindLt100MW_MWh']) * MEAN_LOSS

# Onshore wind power data from from all onshore plants
onshore_wind_data = index_columns.copy()
onshore_wind_data['Capacity'] = (production_data['OnshoreWindGe50kW_MWh'] + production_data['OnshoreWindLt50kW_MWh']) * MEAN_LOSS

# Solar power data from all solar plants
solar_data = index_columns.copy()
solar_data['Capacity'] = (
    production_data['SolarPowerGe40kW_MWh'] +
    production_data['SolarPowerLt10kW_MWh'] +
    production_data['SolarPowerGe10Lt40kW_MWh'] -
    production_data['SolarPowerSelfConMWh']
) * MEAN_LOSS

# Data from all central power data
centralpower_data = index_columns.copy()
centralpower_data['Capacity'] = production_data['CentralPowerMWh'] * MEAN_LOSS

# Data from all local power plants + unknown production
localpower_data = index_columns.copy()
localpower_data['Capacity'] = (production_data['LocalPowerMWh'] + production_data['UnknownProdMWh']) * MEAN_LOSS

# Data from all commercial power plants (waste incineration, etc.)
# Self consumption subtracted
commercial_data = index_columns.copy()
commercial_data['Capacity'] = (production_data['CommercialPowerMWh'] - production_data['LocalPowerSelfConMWh']) * MEAN_LOSS

##############################################################
# GET EXPORT DATA
##############################################################

exchange_sweden = index_columns.copy()
exchange_sweden['Capacity'] = production_data['ExchangeSE_MWh']

exchange_germany = index_columns.copy()
exchange_germany['Capacity'] = production_data['ExchangeGE_MWh']

exchange_greatbelt = index_columns.copy()
exchange_greatbelt['Capacity'] = production_data['ExchangeGreatBelt_MWh']

##############################################################
# MODIFY PRICE DATA
##############################################################

# Load CSV file
price_data = pd.read_csv(price_path, sep=',')

# Filter rows where 'PriceArea' is "DK2"
price_data_dk2 = price_data[price_data['PriceArea'] == 'DK2'].reset_index(drop=True)

# Replace commas with periods in the columns and convert to float
# Assuming production_data is already loaded as a DataFrame
# Apply the operation to every row from the 4th column and up
price_data_dk2.iloc[:, 3:] = price_data_dk2.iloc[:, 3:].applymap(lambda x: float(str(x).replace(',', '.')))


##############################################################
# PARAMS
##############################################################
NUM_EPISODE_STEPS = 24 * 180
CUST_MAX_DEMAND = private_consumption1['ConsumptionkWh'].max() / 1000 # Convert to MWh
MAX_BID_PRICE = price_data_dk2['SpotPriceEUR'].max()+100 # Allow agent to bid higher than all other agents


##############################################################
# NETWORK SETUP
##############################################################

# Initialize consumption Agents
# industrial_consumer = ConsumerAgent("IC", "EX", demand_data=industrial_consumption, price_data=price_data_dk2)


strategic_consumer1 = StrategicConsumerAgent("S1", "EX", demand_data=private_consumption1)
simple_consumer2 = ConsumerAgent("C2", "EX", demand_data=private_consumption2, price_data=price_data_dk2)
simple_consumer3 = ConsumerAgent("C3", "EX", demand_data=private_consumption3, price_data=price_data_dk2)
simple_consumer4 = ConsumerAgent("C4", "EX", demand_data=other_consumption, price_data=price_data_dk2)
simple_consumer5 = ConsumerAgent("C5", "EX", demand_data=unknown_consumption, price_data=price_data_dk2)

# Initialize production Agents
offshore_wind_agent = GeneratorAgent("G1", "EX", capacity_data=offshore_wind_data, price_data=price_data_dk2)
onshore_wind_agent = GeneratorAgent("G2", "EX", capacity_data=onshore_wind_data, price_data=price_data_dk2)
solar_agent = GeneratorAgent("G3", "EX", capacity_data=solar_data, price_data=price_data_dk2)
centralpower_agent = GeneratorAgent("G4", "EX", capacity_data=centralpower_data, price_data=price_data_dk2)
localpower_agent = GeneratorAgent("G5", "EX", capacity_data=localpower_data, price_data=price_data_dk2)
commercialpower_agent = GeneratorAgent("G6", "EX", capacity_data=commercial_data, price_data=price_data_dk2)

# Initiate DummyAgents and ExchangeAgent
dummy_agent = DummyAgent("DD")
exchange_agent = ExchangeAgent("EX")

consumer_agents = [
    strategic_consumer1, simple_consumer2, simple_consumer3, simple_consumer4, simple_consumer5
]

generator_agents = [
    offshore_wind_agent, onshore_wind_agent, solar_agent, centralpower_agent, localpower_agent, commercialpower_agent
]

# Define Network and create connections between Actors
agents = consumer_agents + generator_agents + [exchange_agent] 
network = ph.Network(agents)

# Connect the agents
for agent in consumer_agents:
    network.add_connection("EX", agent.id)
for agent in generator_agents:
    network.add_connection("EX", agent.id)


##############################################################
# SETUP ENVIRONMENT
##############################################################
env = EL_Clearing_Env(num_steps=NUM_EPISODE_STEPS, network=network, CUST_MAX_DEMAND=CUST_MAX_DEMAND, MAX_BID_PRICE=MAX_BID_PRICE)


##############################################################
# RUN VARIABLES
##############################################################
observations, _ = env.reset()
rewards = {}
infos = {}
metrics = {
    "ENV/current_demand": ph.metrics.SimpleEnvMetric("current_demand"),
    "ENV/current_capacity": ph.metrics.SimpleEnvMetric("current_capacity"),
    "ENV/current_price": ph.metrics.SimpleEnvMetric("current_price"),
    "S1/current_demand": ph.metrics.SimpleAgentMetric("S1", "current_demand"),
    "S1/satisfied_demand": ph.metrics.SimpleAgentMetric("S1", "satisfied_demand"),
    "S1/missed_demand": ph.metrics.SimpleAgentMetric("S1", "missed_demand"),
    "C2/satisfied_demand": ph.metrics.SimpleAgentMetric("C2", "satisfied_demand"),
    "C2/missed_demand": ph.metrics.SimpleAgentMetric("C2", "missed_demand"),
    "C3/missed_demand": ph.metrics.SimpleAgentMetric("C3", "missed_demand"),
    "G1/missed_capacity": ph.metrics.SimpleAgentMetric("G1", "missed_capacity"),
    "G2/missed_capacity": ph.metrics.SimpleAgentMetric("G2", "missed_capacity"),
    "G3/missed_capacity": ph.metrics.SimpleAgentMetric("G3", "missed_capacity"),
    "G4/missed_capacity": ph.metrics.SimpleAgentMetric("G4", "missed_capacity"),
}

##############################################################
# LOGGING
##############################################################
ph.telemetry.logger.configure_print_logging(print_messages=True, metrics=metrics, enable=True)

##############################################################
# EXECUTE
##############################################################

if sys.argv[1] == "train":
    ph.utils.rllib.train(
        algorithm="PPO",
        env_class=EL_Clearing_Env,
        env_config={
            'num_steps': NUM_EPISODE_STEPS,
            'network': network,
            'CUST_MAX_DEMAND': CUST_MAX_DEMAND,
            'MAX_BID_PRICE': MAX_BID_PRICE
        },
        iterations=500,
        checkpoint_freq=50,
        policies={"customer_policy": ["S1"]},
        metrics=metrics,
        results_dir="~/ray_results/electricity_market",
        num_workers=1
    )

elif sys.argv[1] == "rollout":
    results = ph.utils.rllib.rollout(
        directory="~/ray_results/electricity_market/LATEST",
        env_config={
            'num_steps': NUM_EPISODE_STEPS,
            'network': network,
            'CUST_MAX_DEMAND': CUST_MAX_DEMAND,
            'MAX_BID_PRICE': MAX_BID_PRICE
        },
        num_repeats=1,
        num_workers=1,
        metrics=metrics,
    )

    results = list(results)

    customer_actions = []
    customer_demand = []
    customer_satisfied_demand = []
    customer_missed_demand = []
    real_prices = []

    for rollout in results:
        # Adds all customer actions sequentially to one big list
        # TODO: Take mean of all values across each timestep (vertically)
        # Episode1: a1, a2, a3, a4, a5
        # Episode2: a1, a2, a3, a4, a5
        # Mean:     m1, m2, m3, m4, m5
        customer_actions += list(
            MAX_BID_PRICE*x[0] for x in rollout.actions_for_agent("S1")
        )
        customer_demand += list(rollout.metrics["S1/current_demand"])
        customer_satisfied_demand += list(rollout.metrics["S1/satisfied_demand"])
        customer_missed_demand += list(rollout.metrics["S1/missed_demand"])
        real_prices += list(rollout.metrics["ENV/current_price"])


    
    # Plot agent acions for each step
    plt.plot(customer_actions)
    plt.plot(real_prices)
    plt.title("Customer action at each hour (price bid)")
    plt.xlabel("Hour")
    plt.ylabel("Price")
    plt.show()



elif sys.argv[1] == "simple":
    while env.current_step < env.num_steps:

        actions = {}
        for aid, obs in observations.items():
            agent = env.agents[aid]
            if isinstance(agent, DummyAgent):
                actions[aid] = 0.9

        #print("\nactions:")
        #print(actions)

        step = env.step(actions)
        observations = step.observations
        rewards = step.rewards
        infos = step.infos