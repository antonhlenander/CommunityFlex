import sys
import matplotlib.pyplot as plt
import phantom as ph
import pandas as pd
import numpy as np
from agents import SimpleProsumerAgent, SimpleCommunityMediator, StrategicProsumerAgent
import stackelberg_custom
from datamanager import DataManager
from setup import Setup
from phantom.utils.samplers import UniformFloatSampler, UniformIntSampler

import os


# Params
NUM_EPISODE_STEPS = 48*365
eta = 0.1 # should this be trainable?
greed = 0.75
rotate = False
no_agents = 14
setup_type = sys.argv[2]

dm = DataManager(prod_path='data/eval/pv.csv', demand_path='data/eval/demandprofiles.csv')
mediator = SimpleCommunityMediator('CM', grid_price=1.8, feedin_price=0.3, local_price=1)

prosumer_agents = Setup.get_agents(setup_type, dm, no_agents)

# Define Network and create connections between Actors
agents = [mediator] + prosumer_agents
network = ph.Network(agents)

# Connect the agents to the mediator
for agent in prosumer_agents:
    network.add_connection("CM", agent.id)

leader_agents = ['CM']
follower_agents = [agent.id for agent in prosumer_agents]

##############################################################
# METRICS
##############################################################
metrics = {}

metrics["env/current_price"] = ph.metrics.SimpleAgentMetric("CM", "current_local_price")
metrics["env/total_load"] = ph.metrics.AggregatedAgentMetric(follower_agents, "current_load", group_reduce_action="sum")
metrics["env/total_prod"] = ph.metrics.AggregatedAgentMetric(follower_agents, "current_prod", group_reduce_action="sum")
metrics["env/total_charge"] = ph.metrics.AggregatedAgentMetric(follower_agents, "current_charge", group_reduce_action="sum")
metrics["env/total_supply"] = ph.metrics.AggregatedAgentMetric(follower_agents, "current_supply", group_reduce_action="sum")
metrics["env/self_consumption"] = ph.metrics.AggregatedAgentMetric(follower_agents, "self_consumption", group_reduce_action="sum")
metrics["env/current_local_bought"] = ph.metrics.AggregatedAgentMetric(follower_agents, "current_local_bought", group_reduce_action="sum")
metrics["env/total_avail_energy"] = ph.metrics.AggregatedAgentMetric(follower_agents, "avail_energy", group_reduce_action="sum")
metrics["env/total_surplus_energy"] = ph.metrics.AggregatedAgentMetric(follower_agents, "surplus_energy", group_reduce_action="sum")
metrics["env/total_loss"] = ph.metrics.AggregatedAgentMetric(follower_agents, "net_loss", group_reduce_action="sum")
metrics["env/current_price"] = ph.metrics.SimpleAgentMetric("CM", "current_local_price")
metrics["env/min_load"] = ph.metrics.AggregatedAgentMetric(follower_agents, "current_load", group_reduce_action="min")
metrics["env/max_load"] = ph.metrics.AggregatedAgentMetric(follower_agents, "current_load", group_reduce_action="max")

for aid in (follower_agents):
    if aid != 'H1':
        continue
    metrics[f"{aid}/current_load"] = ph.metrics.SimpleAgentMetric(aid, "current_load")
    metrics[f"{aid}/current_prod"] = ph.metrics.SimpleAgentMetric(aid, "current_prod")
    metrics[f"{aid}/current_supply"] = ph.metrics.SimpleAgentMetric(aid, "current_supply")
    metrics[f"{aid}/current_charge"] = ph.metrics.SimpleAgentMetric(aid, "current_charge")
    metrics[f"{aid}/self_consumption"] = ph.metrics.SimpleAgentMetric(aid, "self_consumption")
    metrics[f"{aid}/current_local_bought"] = ph.metrics.SimpleAgentMetric(aid, "current_local_bought")
    metrics[f"{aid}/net_loss"] = ph.metrics.SimpleAgentMetric(aid, "net_loss")
    metrics[f"{aid}/acc_local_market_coin"] = ph.metrics.SimpleAgentMetric(aid, "acc_local_market_coin")
    metrics[f"{aid}/acc_feedin_coin"] = ph.metrics.SimpleAgentMetric(aid, "acc_feedin_coin")
    metrics[f"{aid}/utility_prev"] = ph.metrics.SimpleAgentMetric(aid, "utility_prev")
    metrics[f"{aid}/reward"] = ph.metrics.SimpleAgentMetric(aid, "reward")
    metrics[f"{aid}/type.capacity"] = ph.metrics.SimpleAgentMetric(aid, "type.capacity")
    

##############################################################
# RUN
##############################################################

        

if sys.argv[1] == "simple":

    # Enable file logging
    ph.telemetry.logger.configure_file_logging(file_path="log.json", human_readable=False, metrics=metrics, append=False)
    # Define agent supertypes


    # sampler = ph.utils.ranges.UniformRange(1, 10)
    # range = sampler.values()
    # print(range)

    # Define run params
    env = stackelberg_custom.StackelbergEnvCustom(
        num_steps=NUM_EPISODE_STEPS, 
        network=network,
        leader_agents=leader_agents,
        follower_agents=follower_agents,
        #agent_supertypes=agent_supertypes
    )
    rewards = {}
    infos = {}
    episodes = 0

    while episodes < 1:
        terminate = False
        observations = env.reset()
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
        episodes += 1


elif sys.argv[1] == "rollout":

    rolloutnumber=0
    rollouts=10
    single_cap = [1, 2, 3]
    
    agent_supertypes = Setup.get_supertypes_eval('single', eta, greed, no_agents, 10, 3)

    if setup_type == 'single':
        agent_supertypes.update(
            {
                f"H1": StrategicProsumerAgent.Supertype( # 3 locked for this run
                    capacity=ph.utils.ranges.UniformRange(1, 10),
                    eta=eta
                )
            }
        )
        capsample = UniformIntSampler(1, 4)
        greedsample = UniformFloatSampler(0.5, 1.0)
        agent_supertypes.update(
            {
                f"H{i}": SimpleProsumerAgent.Supertype(
                    capacity=capsample.sample(),
                    greed=greedsample.sample(),
                    eta=eta
                )    
                for i in range(2, 15)
            }
        )

    if setup_type == 'single':
        network.agents[f"H{no_agents}"].rotate = False


    results = ph.utils.rllib.rollout(
        directory="~/ray_results/community_market/LATEST",
        env_class=ph.StackelbergEnv,
        env_config={
            'num_steps': NUM_EPISODE_STEPS,
            'network': network,
            'leader_agents': leader_agents,
            'follower_agents': follower_agents,
            'agent_supertypes': agent_supertypes,
        },
        num_repeats=1,
        metrics=metrics,
    )

    print("finished rollout")
    results = list(results)

#     # Plotting
#     for rollout in results:
#         for aid in follower_agents:
#             if sys.argv[2]=='single':
#                 aid = "H1"
#             agent_actions = []
#             agent_charge = []
#             agent_supply = []
#             agent_net_loss = []
#             invalid_actions = []
#             agent_actions += list(rollout.actions_for_agent(aid))
#             agent_charge += list(rollout.metrics[f"{aid}/current_charge"])
#             agent_supply += list(rollout.metrics[f"{aid}/current_supply"])
#             agent_prod = list(rollout.metrics[f"{aid}/current_prod"])
#             agent_load = list(rollout.metrics[f"{aid}/current_load"])
#             agent_net_loss += list(rollout.metrics[f"{aid}/net_loss"])
#             #invalid_actions += list(rollout.metrics[f"{aid}/acc_invalid_actions"])

#             # Remove None values from agent_actions
#             agent_actions = [action for action in agent_actions if action is not None]
#             # Plot distribution of agent action per step for all rollouts
#             folder = f"output/"

#             if not os.path.exists(folder):
#                 os.makedirs(folder)

#             print(agent_actions)
#             plt.hist(agent_actions, bins=6)
#             plt.title("Distribution of action values")
#             plt.xlabel("Agent action")
#             plt.ylabel("Frequency")
#             plt.savefig(f"{folder}action_dist.png")
#             plt.close()

#             plt.plot(agent_net_loss, label='Net Loss')
#             plt.savefig(f"{folder}agent_net_loss.png")
#             plt.close()

#             plt.plot(agent_charge, label='battery charge')
#             plt.savefig(f"{folder}batterycharge.png")
#             plt.close()

#             plt.plot(invalid_actions, label='Invalid Actions')
#             plt.savefig(f"{folder}invalid_actions.png")
#             plt.close()

#             plt.plot(agent_prod, label='Production')
#             plt.savefig(f"{folder}production.png")
#             plt.close()

#             plt.plot(agent_load, label='Load')
#             plt.savefig(f"{folder}load.png")
#             plt.close()

#             plt.hist(agent_load, bins=20)
#             plt.savefig(f"{folder}load_dist.png")
#             plt.close()

#             plt.hist(agent_prod, bins=20)
#             plt.savefig(f"{folder}prod_dist.png")
#             plt.close()

#             plt.plot(agent_supply, label='Supply')
#             plt.savefig(f"{folder}supply.png")
#             plt.close()

#             plt.hist(agent_supply, bins=20)
#             plt.savefig(f"{folder}supply_dist.png")
#             plt.close()