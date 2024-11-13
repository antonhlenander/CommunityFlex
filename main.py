import sys
import matplotlib.pyplot as plt
import phantom as ph
import pandas as pd
import numpy as np
from agents import DummyAgent, StrategicProsumerAgent, SimpleCommunityMediator
from environment import CommunityEnv
from datamanager import DataManager

# Params
NUM_EPISODE_STEPS = 2400
eta = 0.23 # From AI economist paper
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
# METRICS
##############################################################

for aid in (follower_agents):
    metrics[f"{aid}/current_charge"] = ph.metrics.SimpleAgentMetric(aid, "current_charge")
    metrics[f"{aid}/max_batt_charge"] = ph.metrics.SimpleAgentMetric(aid, "max_batt_charge")
    metrics[f"{aid}/max_batt_discharge"] = ph.metrics.SimpleAgentMetric(aid, "max_batt_discharge")
    metrics[f"{aid}/current_supply"] = ph.metrics.SimpleAgentMetric(aid, "current_supply")


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

    
    agent_charge = []
    agent_supply = []

    for rollout in results:
        for aid in follower_agents:
            agent_actions = []
            agent_actions += list(rollout.actions_for_agent(aid))
            agent_charge += list(rollout.metrics[f"{aid}/current_charge"])
            agent_supply += list(rollout.metrics[f"{aid}/current_supply"])

            # Remove None values from agent_actions
            agent_actions = [action for action in agent_actions if action is not None]
            # Plot distribution of agent action per step for all rollouts
            plt.hist(agent_actions, bins=4)
            plt.title("Distribution of action values")
            plt.xlabel("Agent action")
            plt.ylabel("Frequency")
            plt.show()


            # plt.figure(figsize=(12, 6))
            # plt.plot(agent_actions, label='Action')
            # plt.plot(agent_charge, label='Charge')
            # plt.plot(agent_supply, label='Supply')
            # plt.xlabel('Time Step')
            # plt.ylabel('Value')
            # plt.title(f'Agent {aid} Charge and Supply Over Time')
            # plt.legend()
            # plt.show()
        






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



