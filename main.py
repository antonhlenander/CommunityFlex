import sys
import matplotlib.pyplot as plt
import phantom as ph
import pandas as pd
import numpy as np
import ray

from trained_policy import TrainedPolicy
from agents import SimpleProsumerAgent, SimpleCommunityMediator, StrategicProsumerAgent, StrategicCommunityMediator
import stackelberg_custom
from datamanager import DataManager
from setup import Setup
from phantom.utils.samplers import UniformFloatSampler, UniformIntSampler


from ray.rllib.examples.models.action_mask_model import TorchActionMaskModel
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOTorchPolicy
import os

# Register the model
ModelCatalog.register_custom_model("torch_action_mask_model", TorchActionMaskModel)

# Params
NUM_EPISODE_STEPS = 8735*2
eta = 0.1 # should this be trainable?
greed = 0.8
rotate = False
no_agents = 14
discount = 0.5 # possibly supertype?
setup_type = sys.argv[2]

dm = DataManager(demand_path="data/fullyearPV_singleDemand/demandprofiles.csv", cap_path="data/eval/caps.csv")
mediator = SimpleCommunityMediator('CM', dm=dm)

prosumer_agents = Setup.get_agents(setup_type, dm, no_agents)

# Define Network and create connections between Actors
agents = prosumer_agents + [mediator]
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
metrics["env/total_charge"] = ph.metrics.AggregatedAgentMetric(follower_agents, "current_charge", group_reduce_action="sum")
metrics["env/total_supply"] = ph.metrics.AggregatedAgentMetric(follower_agents, "current_supply", group_reduce_action="sum")
metrics["env/self_consumption"] = ph.metrics.AggregatedAgentMetric(follower_agents, "self_consumption", group_reduce_action="sum")
metrics["env/current_local_bought"] = ph.metrics.AggregatedAgentMetric(follower_agents, "current_local_bought", group_reduce_action="sum")
metrics["env/total_loss"] = ph.metrics.AggregatedAgentMetric(follower_agents, "net_loss", group_reduce_action="sum")

for aid in (follower_agents):
    metrics[f"{aid}/net_loss"] = ph.metrics.SimpleAgentMetric(aid, "net_loss")
    metrics[f"{aid}/acc_local_market_coin"] = ph.metrics.SimpleAgentMetric(aid, "acc_local_market_coin")
    metrics[f"{aid}/acc_feedin_coin"] = ph.metrics.SimpleAgentMetric(aid, "acc_feedin_coin")
    metrics[f"{aid}/acc_local_market_cost"] = ph.metrics.SimpleAgentMetric(aid, "acc_local_market_cost")
    metrics[f"{aid}/acc_grid_cost"] = ph.metrics.SimpleAgentMetric(aid, "acc_grid_market_cost")
    metrics[f"{aid}/utility"] = ph.metrics.SimpleAgentMetric(aid, "utility_prev")

    #metrics[f"{aid}/utility_prev"] = ph.metrics.SimpleAgentMetric(aid, "utility_prev")
    #metrics[f"{aid}/reward"] = ph.metrics.SimpleAgentMetric(aid, "reward")
    #metrics[f"{aid}/type.capacity"] = ph.metrics.SimpleAgentMetric(aid, "type.capacity")
    
##############################################################
# LOGGING
##############################################################
#ph.telemetry.logger.configure_print_logging(print_messages=True, metrics=metrics, enable=True)
#ph.telemetry.logger.configure_file_logging(file_path="log.json", human_readable=False, metrics=metrics, append=False)

##############################################################
# RUN VARIABLES
##############################################################
rewards = {}
infos = {}

# I think this should be the same for training 1 agent and all agents?


##############################################################
# EXECUTE
# TODO: Entropy schedule?
##############################################################

if sys.argv[1] == "train":
    agent_supertypes = {}
    if setup_type == 'multi':
        agent_supertypes.update(
            {
                f"H{i}": StrategicProsumerAgent.Supertype(
                    capacity=UniformIntSampler(1, 4),
                    eta=UniformFloatSampler(eta, eta)
                )    
                for i in range(1, 15)
            }
        )

        policies = {
            "prosumer_policy": (
                TrainedPolicy, 
                follower_agents
            ),
            "mediator_policy": ["CM"]
        }
    
    if setup_type == 'simple':
        agent_supertypes.update(
            {
                f"H{i}": SimpleProsumerAgent.Supertype(
                    capacity=UniformIntSampler(1, 4),
                    greed=UniformFloatSampler(0.5, 1),
                    eta=UniformFloatSampler(eta, eta)
                )    
                for i in range(1, 15)
            }
        )

        policies = {"mediator_policy": ["CM"]}

    if setup_type == 'multsing':
        agent_supertypes.update(
            {
                f"H{i}": StrategicProsumerAgent.Supertype(
                    capacity=UniformIntSampler(1, 4),
                    eta=UniformFloatSampler(eta, eta)
                )    
                for i in range(1, 15)
            }
        )
        agent_supertypes.update(
            {
                "CM": SimpleCommunityMediator.Supertype(
                    discount=UniformFloatSampler(0.5, 0.5)
                )    
            }
        )
        policies = {"prosumer_policy": follower_agents}

    ph.utils.rllib.train(
        algorithm="PPO",
        env_class=ph.StackelbergEnv,
        env_config={
            'num_steps': NUM_EPISODE_STEPS,
            'network': network,
            'leader_agents': leader_agents,
            'follower_agents': follower_agents,
            'agent_supertypes': agent_supertypes,
        },
        rllib_config={
            "model": {"custom_model": "torch_action_mask_model"},
            "lr": 0.0003,
            "entropy_coeff": 0.002,
            "lambda": 0.95,
        },
        iterations=200,
        checkpoint_freq=1,
        policies=policies,
        metrics=metrics,
        #num_workers=1,
        results_dir="~/ray_results/community_flex",
    )

elif sys.argv[1] == "rollout":

    agent_supertypes = {}
    if setup_type == 'multi':
        agent_supertypes.update(
            {
                f"H{i}": StrategicProsumerAgent.Supertype(
                    capacity=UniformIntSampler(1, 4),
                    eta=UniformFloatSampler(eta, eta)
                )    
                for i in range(1, 15)
            }
        )
    
    if setup_type == 'single':
        agent_supertypes.update(
            {
                f"H1": StrategicProsumerAgent.Supertype( # 3 locked for this run
                    capacity=1,
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

    results = list(results)

    for rollout in results:
        for aid in follower_agents:
            aid = "H1"
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
            folder = f"output/H1_3/"

            if not os.path.exists(folder):
                os.makedirs(folder)

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
    # Define agent supertypes
    agent_supertypes = {}
    agent_supertypes.update(
        {
            f"H{i}": SimpleProsumerAgent.Supertype(
                capacity=UniformIntSampler(1, 4),
                greed=UniformFloatSampler(0.5, 1.0),
                eta=UniformFloatSampler(eta, eta)

            )    
            for i in range(1, 15)
        },
    )

    # Define environment
    env = stackelberg_custom.StackelbergEnvCustom(
        num_steps=NUM_EPISODE_STEPS, 
        network=network,
        leader_agents=leader_agents,
        follower_agents=follower_agents,
        agent_supertypes=agent_supertypes
    )
    
    terminate = False
    episodes = 0

    while episodes < 10:
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
