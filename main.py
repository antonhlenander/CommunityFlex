import sys
import matplotlib.pyplot as plt
import phantom as ph
import pandas as pd
import numpy as np
from agents import SimpleProsumerAgent, SimpleCommunityMediator, StrategicProsumerAgent
from environment import CommunityEnv, SimpleCommunityEnv
from datamanager import DataManager
import ray

from ray.rllib.examples.models.action_mask_model import TorchActionMaskModel
from ray.rllib.models import ModelCatalog

# Register the model
ModelCatalog.register_custom_model("torch_action_mask_model", TorchActionMaskModel)




# Params
NUM_EPISODE_STEPS = 48*365
eta = 0.0 # From AI economist paper
greed = 0.8


dm = DataManager()

# Case of strategic Agents
house1 = StrategicProsumerAgent('H1', 'CM', dm, eta)
house2 = StrategicProsumerAgent('H2', 'CM', dm, eta)
house3 = StrategicProsumerAgent('H3', 'CM', dm, eta)
house4 = StrategicProsumerAgent('H4', 'CM', dm, eta)
house5 = StrategicProsumerAgent('H5', 'CM', dm, eta)
house6 = StrategicProsumerAgent('H6', 'CM', dm, eta)
house7 = StrategicProsumerAgent('H7', 'CM', dm, eta)
house8 = StrategicProsumerAgent('H8', 'CM', dm, eta)
house9 = StrategicProsumerAgent('H9', 'CM', dm, eta)
house10 = StrategicProsumerAgent('H10', 'CM', dm, eta)
house11 = StrategicProsumerAgent('H11', 'CM', dm, eta)
house12 = StrategicProsumerAgent('H12', 'CM', dm, eta)
house13 = StrategicProsumerAgent('H13', 'CM', dm, eta)
house14 = StrategicProsumerAgent('H14', 'CM', dm, eta)


#Simple Agents case
# house1 = SimpleProsumerAgent('H1', 'CM', dm, greed)
# house2 = SimpleProsumerAgent('H2', 'CM', dm, greed)
# house3 = SimpleProsumerAgent('H3', 'CM', dm, greed)
# house4 = SimpleProsumerAgent('H4', 'CM', dm, greed)
# house5 = SimpleProsumerAgent('H5', 'CM', dm, greed)
# house6 = SimpleProsumerAgent('H6', 'CM', dm, greed)
# house7 = SimpleProsumerAgent('H7', 'CM', dm, greed)
# house8 = SimpleProsumerAgent('H8', 'CM', dm, greed)
# house9 = SimpleProsumerAgent('H9', 'CM', dm, greed)
# house10 = SimpleProsumerAgent('H10', 'CM', dm, greed)
# house11 = SimpleProsumerAgent('H11', 'CM', dm, greed)
# house12 = SimpleProsumerAgent('H12', 'CM', dm, greed)
# house13 = SimpleProsumerAgent('H13', 'CM', dm, greed)
# house14 = SimpleProsumerAgent('H14', 'CM', dm, greed)

# Mediator 
mediator = SimpleCommunityMediator('CM', grid_price=1.8, feedin_price=0.3)

#dummy_agent = DummyAgent("DD")
prosumer_agents = [
    house1, house2, house3, house4, house5, house6, house7, house8, house9, house10, house11, house12, house13, house14
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

for aid in (follower_agents):
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
    metrics[f"{aid}/type"] = ph.metrics.SimpleAgentMetric(aid, "type")
    
##############################################################
# LOGGING
##############################################################
#ph.telemetry.logger.configure_print_logging(print_messages=True, metrics=metrics, enable=True)
#ph.telemetry.logger.configure_file_logging(file_path="log.json", human_readable=False, metrics=metrics, append=False)


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
        rllib_config={"model": {"custom_model": "torch_action_mask_model"}},
        iterations=1000,
        checkpoint_freq=1,
        policies={"prosumer_policy": follower_agents},
        metrics=metrics,
        results_dir="~/ray_results/community_market_multi2",
        num_workers=1
    )

elif sys.argv[1] == "rollout":
    results = ph.utils.rllib.rollout(
        directory="~/ray_results/community_market_multi/LATEST",
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

    for rollout in results:
        for aid in follower_agents:
            agent_actions = []
            agent_charge = []
            agent_supply = []
            agent_net_loss = []
            invalid_actions = []
            agent_actions += list(rollout.actions_for_agent(aid))
            agent_charge += list(rollout.metrics[f"{aid}/current_charge"])
            agent_supply += list(rollout.metrics[f"{aid}/current_supply"])
            agent_prod = list(rollout.metrics[f"{aid}/current_prod"])
            agent_load = list(rollout.metrics[f"{aid}/current_load"])
            agent_net_loss += list(rollout.metrics[f"{aid}/net_loss"])
            #invalid_actions += list(rollout.metrics[f"{aid}/acc_invalid_actions"])

            # Remove None values from agent_actions
            agent_actions = [action for action in agent_actions if action is not None]
            # Plot distribution of agent action per step for all rollouts
            folder = f"output/{aid}/"

            print(agent_actions)
            plt.hist(agent_actions, bins=6)
            plt.title("Distribution of action values")
            plt.xlabel("Agent action")
            plt.ylabel("Frequency")
            plt.savefig(f"{folder}action_dist.png")
            plt.close()

            plt.plot(agent_net_loss, label='Net Loss')
            plt.savefig(f"{folder}agent_net_loss.png")
            plt.close()

            plt.plot(agent_charge, label='battery charge')
            plt.savefig(f"{folder}batterycharge.png")
            plt.close()

            plt.plot(invalid_actions, label='Invalid Actions')
            plt.savefig(f"{folder}invalid_actions.png")
            plt.close()

            plt.plot(agent_prod, label='Production')
            plt.savefig(f"{folder}production.png")
            plt.close()

            plt.plot(agent_load, label='Load')
            plt.savefig(f"{folder}load.png")
            plt.close()

            plt.hist(agent_load, bins=20)
            plt.savefig(f"{folder}load_dist.png")
            plt.close()

            plt.hist(agent_prod, bins=20)
            plt.savefig(f"{folder}prod_dist.png")
            plt.close()

            plt.plot(agent_supply, label='Supply')
            plt.savefig(f"{folder}supply.png")
            plt.close()

            plt.hist(agent_supply, bins=20)
            plt.savefig(f"{folder}supply_dist.png")
            plt.close()

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
