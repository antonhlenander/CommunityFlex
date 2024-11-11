import sys
import matplotlib.pyplot as plt
import phantom as ph
import pandas as pd
import numpy as np
from agents import DummyAgent, ExchangeAgent, ConsumerAgent, StrategicConsumerAgent, StrategicGeneratorAgent
from environment import ElMarketEnv
from datamanager import DataManager

##############################################################
# PATHS
##############################################################


# Mean loss for transmission and distribution
MEAN_LOSS = 1 - 0.055

##############################################################
# PARAMS
##############################################################

##############################################################
# NETWORK SETUP
##############################################################

simple_consumer1 = ConsumerAgent('C1', 'Etageejendom', 'EX', dm)
simple_consumer2 = ConsumerAgent('C2', 'Fritidshuse', 'EX', dm)
simple_consumer3 = ConsumerAgent('C3', 'Parcel- og rækkehuse',"EX", dm)
simple_consumer4 = ConsumerAgent('C4', 'Andet', 'EX', dm)
simple_consumer5 = ConsumerAgent('C5', 'Ukendt','EX', dm)
industrial_consumer = ConsumerAgent('C6', 'Erhverv', 'EX', dm)

# Initialize strategic production agent
strategic_generator = StrategicGeneratorAgent('SG', 'CentralPower', 'EX', dm, capacity=747, min_load=0, start_up=0, ramp_rate=231, marginal_cost=100)

# Initialize simple production agents
offshore_wind_agent = GeneratorAgent('G1', 'OffshoreWind', 'EX', dm)
onshore_wind_agent = GeneratorAgent('G2', 'OnshoreWind', 'EX', dm)
solar_agent = GeneratorAgent('G3', 'SolarPower','EX', dm)
localpower_agent = GeneratorAgent('G5', 'LocalPower', 'EX', dm)
commercialpower_agent = GeneratorAgent('G6', 'CommercialPower','EX', dm)

# Initiate DummyAgents and ExchangeAgent
dummy_agent = DummyAgent("DD")
exchange_agent = ExchangeAgent("EX")

consumer_agents = [
    simple_consumer1, simple_consumer2, simple_consumer3, simple_consumer4, simple_consumer5, industrial_consumer
]

generator_agents = [
    strategic_generator, offshore_wind_agent, onshore_wind_agent, solar_agent, localpower_agent, commercialpower_agent
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
env = ElMarketEnv(num_steps=NUM_EPISODE_STEPS, network=network, CUST_MAX_DEMAND=CUST_MAX_DEMAND, MAX_BID_PRICE=MAX_BID_PRICE)


##############################################################
# RUN VARIABLES
##############################################################
observations, _ = env.reset()
rewards = {}
infos = {}
metrics = {
    "ENV/current_demand": ph.metrics.SimpleEnvMetric("current_demand"),
    "ENV/current_production": ph.metrics.SimpleEnvMetric("current_production"),
    "ENV/current_price": ph.metrics.SimpleEnvMetric("current_price"),

    "SG/current_production": ph.metrics.SimpleAgentMetric("SG", "current_production"),

    "C2/satisfied_demand": ph.metrics.SimpleAgentMetric("C2", "satisfied_demand"),
    "C2/missed_demand": ph.metrics.SimpleAgentMetric("C2", "missed_demand"),
    "C3/missed_demand": ph.metrics.SimpleAgentMetric("C3", "missed_demand"),
    
    "G1/missed_production": ph.metrics.SimpleAgentMetric("G1", "missed_production"),
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
        env_class=ElMarketEnv,
        env_config={
            'num_steps': NUM_EPISODE_STEPS,
            'network': network,
            'CUST_MAX_DEMAND': CUST_MAX_DEMAND,
            'MAX_BID_PRICE': MAX_BID_PRICE
        },
        iterations=100,
        checkpoint_freq=2,
        policies={"generator_policy": ["SG"]},
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

    centralpower_production = []
    customer_demand = []
    customer_satisfied_demand = []
    customer_missed_demand = []
    real_production = dm.prod_df['CentralPower']
    system_prices = []
    demand = []

    for rollout in results:
        # centralpower_production += list(
        #     CENTRALPOWER_CAPACITY*x[0] for x in rollout.actions_for_agent("SG")
        # )
        centralpower_production += list(rollout.metrics["SG/current_production"])
        # customer_satisfied_demand += list(rollout.metrics["SG/satisfied_demand"])
        # customer_missed_demand += list(rollout.metrics["SG/missed_demand"])
        system_prices += list(rollout.metrics["ENV/current_price"])
        demand += list(rollout.metrics["ENV/current_demand"])

    # Plot agent acions for each step

    plt.figure(figsize=(18, 6))

    plt.subplot(3, 1, 1)
    plt.plot(centralpower_production, label='Central Power Production (Agent)')
    plt.plot(real_production, label='Real Central Power Production')
    plt.title("Generator Production at Each Hour")
    plt.xlabel("Hour")
    plt.ylabel("Production (MWh)")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(system_prices, label='System price')
    plt.title("Spot Price at Each Hour")
    plt.xlabel("Hour")
    plt.ylabel("Price (EUR/MWh)")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(demand, label='Demand')
    plt.title("Demand at Each Hour")
    plt.xlabel("Hour")
    plt.ylabel("Demand")
    plt.legend()

    plt.tight_layout()
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
