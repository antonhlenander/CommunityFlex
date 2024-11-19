import sys
import matplotlib.pyplot as plt
import phantom as ph
import pandas as pd
import numpy as np
from agents import SimpleProsumerAgent, SimpleCommunityMediator, StrategicProsumerAgent
from environment import CommunityEnv
from datamanager import DataManager


# Params
NUM_EPISODE_STEPS = 48*182
eta = 0.23 # From AI economist paper
greed = 0.8


dm = DataManager()

# Case of strategic Agents
# house1 = SimpleProsumerAgent('H1', 'CM', dm, battery, eta)
# house2 = SimpleProsumerAgent('H2', 'CM', dm, battery, eta)
# house3 = SimpleProsumerAgent('H3', 'CM', dm, battery, eta)
# house4 = SimpleProsumerAgent('H4', 'CM', dm, battery, eta)
# house5 = SimpleProsumerAgent('H5', 'CM', dm, battery, eta)

#Simple Agents case
house1 = SimpleProsumerAgent('H1', 'CM', dm, greed)
house2 = SimpleProsumerAgent('H2', 'CM', dm, greed)
house3 = SimpleProsumerAgent('H3', 'CM', dm, greed)
house4 = SimpleProsumerAgent('H4', 'CM', dm, greed)
house5 = SimpleProsumerAgent('H5', 'CM', dm, greed)

# Mediator 
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
# METRICS
##############################################################
metrics = {}
for aid in (follower_agents):
    metrics[f"{aid}/current_load"] = ph.metrics.SimpleAgentMetric(aid, "current_load")
    metrics[f"{aid}/current_prod"] = ph.metrics.SimpleAgentMetric(aid, "current_prod")
    metrics[f"{aid}/current_supply"] = ph.metrics.SimpleAgentMetric(aid, "current_supply")
    metrics[f"{aid}/current_charge"] = ph.metrics.SimpleAgentMetric(aid, "current_charge")
    metrics[f"{aid}/net_loss"] = ph.metrics.SimpleAgentMetric(aid, "net_loss")
    metrics["env/total_load"] = ph.metrics.AggregatedAgentMetric(follower_agents, "current_load", group_reduce_action="sum")
    metrics["env/total_prod"] = ph.metrics.AggregatedAgentMetric(follower_agents, "current_prod", group_reduce_action="sum")
    metrics["env/total_supply"] = ph.metrics.AggregatedAgentMetric(follower_agents, "current_supply", group_reduce_action="sum")
    #metrics["env/total_loss"] = ph.metrics.AggregatedAgentMetric(follower_agents, "net_loss", group_reduce_action="sum")
    

##############################################################
# LOGGING
##############################################################
#ph.telemetry.logger.configure_print_logging(print_messages=True, metrics=metrics, enable=True)
ph.telemetry.logger.configure_file_logging(file_path="log.json", human_readable=False, metrics=metrics, append=False)


##############################################################
# RUN VARIABLES
##############################################################
observations = env.reset()
rewards = {}
infos = {}

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
        iterations=2,
        checkpoint_freq=2,
        policies={"prosumer_policy": ["H1"]},
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
    terminate = False
    while env.current_step < env.num_steps:
        actions = {
            agent.id: agent.action_space.sample()
            for agent in env.strategic_agents
        }
        # log simple agent actions?
        # log messages?

        # Manually pass termination bool
        if env.current_step+1 == env.num_steps:
            terminate = True

        step = env.step(actions, terminate)
        observations = step.observations
        rewards = step.rewards
        infos = step.infos
