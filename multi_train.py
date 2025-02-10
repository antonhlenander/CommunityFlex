import sys
import matplotlib.pyplot as plt
import phantom as ph
import pandas as pd
import numpy as np
import ray

import train_from_checkpoint
from trained_policy import TrainedPolicy
from agents import SimpleProsumerAgent, SimpleCommunityMediator, StrategicProsumerAgent, StrategicCommunityMediator
import stackelberg_custom
from stackelberg_reward import StackelbergRewardDelayEnv
from datamanager import DataManager
from setup import Setup
from phantom.utils.samplers import UniformFloatSampler, UniformIntSampler

from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.examples.models.action_mask_model import TorchActionMaskModel
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOTorchPolicy
import os

# Register the model
ModelCatalog.register_custom_model("torch_action_mask_model", TorchActionMaskModel)
# Params
NUM_EPISODE_STEPS = 48*30
eta = 0.1 # should this be trainable?
greed = 0.8
rotate = False
no_agents = 5
setup_type = sys.argv[2]

dm = DataManager(demand_path="data/fullyearPV_singleDemand/demandprofiles.csv", cap_path="data/eval/caps.csv")
if setup_type == 'simple' or setup_type == 'multsing':
    mediator = SimpleCommunityMediator('CM', dm=dm)
else:
    mediator = StrategicCommunityMediator('CM', dm=dm)

prosumer_agents = Setup.get_agents(setup_type, dm, no_agents)

simple_agents = [agent.id for agent in prosumer_agents if isinstance(agent, SimpleProsumerAgent)]
strategic_prosumers = [agent.id for agent in prosumer_agents if agent.id not in simple_agents]

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
metrics["CM/budget_balance"] = ph.metrics.SimpleAgentMetric("CM", "budget_balance")
metrics["CM/penalized_amount"] = ph.metrics.SimpleAgentMetric("CM", "penalized_amount")
metrics["CM/no_of_diff_actions"] = ph.metrics.SimpleAgentMetric("CM", "no_different_prices")
metrics["CM/mediator_netloss"] = ph.metrics.SimpleAgentMetric("CM", "mediator_netloss")
metrics["CM/mediator_payments"] = ph.metrics.SimpleAgentMetric("CM", "alltime_mediator_payments")
metrics["CM/prosumers_netloss"] = ph.metrics.SimpleAgentMetric("CM", "prosumers_netloss")
metrics["CM/prosumers_payments"] = ph.metrics.SimpleAgentMetric("CM", "alltime_prosumers_payments")
metrics["CM/normed_balance"] = ph.metrics.SimpleAgentMetric("CM", "normed_balance")
metrics["CM/max_reward"] = ph.metrics.SimpleAgentMetric("CM", "max_reward")
metrics["CM/min_reward"] = ph.metrics.SimpleAgentMetric("CM", "min_reward")
metrics["CM/max_marg_netloss"] = ph.metrics.SimpleAgentMetric("CM", "max_marg_netloss")
metrics["CM/min_marg_netloss"] = ph.metrics.SimpleAgentMetric("CM", "min_marg_netloss")
metrics["CM/current_cap_limit"] = ph.metrics.SimpleAgentMetric("CM", "current_cap_limit")

for aid in (follower_agents):
    metrics[f"{aid}/net_loss"] = ph.metrics.SimpleAgentMetric(aid, "net_loss")
    metrics[f"{aid}/acc_local_market_coin"] = ph.metrics.SimpleAgentMetric(aid, "acc_local_market_coin")
    metrics[f"{aid}/acc_feedin_coin"] = ph.metrics.SimpleAgentMetric(aid, "acc_feedin_coin")
    metrics[f"{aid}/acc_local_market_cost"] = ph.metrics.SimpleAgentMetric(aid, "acc_local_market_cost")
    metrics[f"{aid}/acc_grid_cost"] = ph.metrics.SimpleAgentMetric(aid, "acc_grid_market_cost")
    #metrics[f"{aid}/utility"] = ph.metrics.SimpleAgentMetric(aid, "utility_prev")

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
# TODO: LR schedule?
##############################################################

if sys.argv[1] == "train":
    agent_supertypes = {}
    
    if setup_type == 'multi':
        rollout_length=5
        agent_supertypes.update(
            {
                f"H{i}": StrategicProsumerAgent.Supertype(
                    capacity = UniformIntSampler(1, 4),
                    #capacity = 2,
                    eta=UniformFloatSampler(eta, eta),
                    rollout=0
                )    
                for i in range(1, no_agents+1)
            }
        )
        agent_supertypes.update(
            {
                f"CM": StrategicCommunityMediator.Supertype(
                    discount=1,
                    cap_var=1,
                    dso_penalty=15,
                    lagrange_mult=0, # 0 for penalty objective, 1 for budget balance objective
                    lagrange_lr=0,
                    rollout_length=rollout_length,
                    rollout=1, # 1 to deactivate resets of netloss
                    # range for langrange multiplier to update through training?
                )    
            }
        )

        policies = {
            # "prosumer_policy": (
            #     TrainedPolicy,
            #     follower_agents
            # ),
            "prosumer_policy": strategic_prosumers,
            "mediator_policy": ["CM"]
        }

    ##############
    # Copy setup
    ##############
    if setup_type == 'copy':
        rollout_length=4
        agent_supertypes.update(
            {
                aid : SimpleProsumerAgent.Supertype(
                    capacity=0,
                    rollout=0
                )    
                for aid in simple_agents
            }
        ) 
        agent_supertypes.update(
            {
                aid : StrategicProsumerAgent.Supertype(
                    capacity = UniformIntSampler(1, 4),
                    eta=0.05,
                    price_multiplier=2,
                    rollout=0,
                    maxbuy=0.5,
                    maxsell=1
                )    
                for aid in strategic_prosumers
            }
        ) 
        agent_supertypes.update(
            {
                f"CM": StrategicCommunityMediator.Supertype(
                    discount=1,
                    cap_var=1,
                    dso_penalty=75,
                    lagrange_mult=0.0, # 0 for penalty objective, 1 for budget balance objective
                    lagrange_lr=0,
                    rollout_length=rollout_length,
                    rollout=1, # 1 to deactivate resets of netloss
                    reward_scale=1000,
                    no_agents=no_agents
                )    
            }
        )

        policies = {
        #     "prosumer_policy": (
        #         TrainedPolicy,
        #         follower_agents
        #     ),
            "prosumer_policy": strategic_prosumers,
            "mediator_policy": ["CM"]
        }

    num_workers = int(sys.argv[3])

    train_from_checkpoint.train(
        policy_checkpoint="/Users/antonlenander/ray_results/single_policy_new/eta0.1_new_allfixed/checkpoint_000034/policies/prosumer_policy",
        algorithm="PPO",
        env_class=StackelbergRewardDelayEnv,
        env_config={
            'num_steps': NUM_EPISODE_STEPS,
            'network': network,
            'leader_agents': leader_agents,
            'follower_agents': follower_agents,
            'agent_supertypes': agent_supertypes,
        },
        rllib_config={
            "multiagent": {
                "policies": {
                    "mediator_policy": PolicySpec(
                        policy_class=None,
                        action_space=mediator.action_space,
                        observation_space=mediator.observation_space,
                        config={
                            "lr": 0.0001,
                            "num_sgd_iter": 5,
                            "entropy_coeff_schedule": [[0, 1], [1e+5, 1], [1e+6, 0.1]],
                        },
                    ),
                    "prosumer_policy": PolicySpec(
                        policy_class=None,
                        action_space=prosumer_agents[0].action_space,
                        observation_space=prosumer_agents[0].observation_space,
                        config={
                            "lr": 0.0003,
                            "num_sgd_iter": 100,
                            "entropy_coeff": 0.025,
                        },
                    ),
                },
            },
            "model": {"custom_model": "torch_action_mask_model"},
            "lambda": 0.98,
            "gamma": 0.998,
            "grad_clip": 10,
            "vf_loss_coeff": 0.05,
            "rollout_fragment_length": 48*4,
            "train_batch_size": NUM_EPISODE_STEPS*num_workers,
            "sgd_minibatch_size": int(NUM_EPISODE_STEPS),
        },
        iterations=300,
        checkpoint_freq=1,
        policies=policies,
        metrics=metrics,
        num_workers=num_workers,
        results_dir="~/ray_results/new_multi_2",
    )