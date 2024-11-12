import sys
import matplotlib.pyplot as plt
import phantom as ph
import pandas as pd
import numpy as np
from agents import DummyAgent, StrategicProsumerAgent, SimpleCommunityMediator
from environment import CommunityEnv
from datamanager import DataManager

# Params
NUM_EPISODE_STEPS = 48
eta = 0.9
battery = {
    'cap': 10.0,
    'charge_rate': 2.5,
}

dm = DataManager()

house1 = StrategicProsumerAgent('H1', 'CM', dm, battery, eta)
house2 = StrategicProsumerAgent('H2', 'CM', dm, battery, eta)
house3 = StrategicProsumerAgent('H3', 'CM', dm, battery, eta)
house4 = StrategicProsumerAgent('H4', 'CM', dm, battery, eta)
house5 = StrategicProsumerAgent('H5', 'CM', dm, battery, eta)
mediator = SimpleCommunityMediator('CM', grid_price=1.8, local_price=1.05, feedin_price=0.3)

#dummy_agent = DummyAgent("DD")

prosumer_agents = [
    house1, house2, house3, house4, house5
]

# Define Network and create connections between Actors
agents = [mediator] + prosumer_agents 
network = ph.Network(agents)

# Connect the agents to the mediator
for agent in prosumer_agents:
    network.add_connection("CM", agent.id)

leader_agents = ['CM']
follower_agents = [agent.id for agent in prosumer_agents]

env = CommunityEnv(
    num_steps=NUM_EPISODE_STEPS, 
    network=network, 
    leader_agents=leader_agents,
    follower_agents=follower_agents,
    )

##############################################################
# RUN VARIABLES
##############################################################
observations = env.reset()
rewards = {}
infos = {}
metrics = {}

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
        env_class=CommunityEnv,
        env_config={
            'num_steps': NUM_EPISODE_STEPS,
            'network': network,
            'leader_agents': leader_agents,
            'follower_agents': follower_agents
        },
        iterations=100,
        checkpoint_freq=2,
        policies={"prosumer_policy": follower_agents},
        metrics=metrics,
        results_dir="~/ray_results/community_market",
        num_workers=1
    )

elif sys.argv[1] == "rollout":
    results = ph.utils.rllib.rollout(
        directory="~/ray_results/community_market/LATEST",
        env_config={
            'num_steps': NUM_EPISODE_STEPS,
            'network': network,
            'leader_agents': leader_agents,
            'follower_agents': follower_agents
        },
        num_repeats=1,
        num_workers=1,
        metrics=metrics,
    )

    results = list(results)

elif sys.argv[1] == "test":
    while env.current_step < env.num_steps:
        actions = {
            agent.id: agent.action_space.sample()
            for agent in env.strategic_agents
        }

        step = env.step(actions)
        observations = step.observations
        rewards = step.rewards
        infos = step.infos

# elif sys.argv[1] == "test":
#     while env.current_step < env.num_steps:
#         actions = {}
#         for aid, obs in observations.items():
#             agent = env.agents[aid]
#             if isinstance(agent, DummyAgent):
#                 actions[aid] = 0.9

#         #print("\nactions:")
#         #print(actions)

#         step = env.step(actions)
#         observations = step.observations
#         rewards = step.rewards
#         infos = step.infos
