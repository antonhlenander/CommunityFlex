import sys
import matplotlib.pyplot as plt
import phantom as ph
import pandas as pd
import numpy as np
import ray

from trained_policy import TrainedPolicy
from agents import SimpleProsumerAgent, SimpleCommunityMediator, StrategicProsumerAgent, StrategicCommunityMediator
import stackelberg_custom
from stackelberg_reward import StackelbergRewardDelayEnv
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
no_agents = 5
setup_type = sys.argv[2]

dm = DataManager(demand_path="data/fullyearPV_singleDemand/demandprofiles.csv", cap_path="data/eval/caps.csv")
mediator = StrategicCommunityMediator('CM', dm=dm)

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
    
    if setup_type == 'simple':
        agent_supertypes.update(
            {
                f"H{i}": SimpleProsumerAgent.Supertype(
                    capacity=0,
                    greed=UniformFloatSampler(0.5, 1),
                    eta=UniformFloatSampler(eta, eta)
                )    
                for i in range(1, no_agents+1)
            }
        )
        agent_supertypes.update(
            {
                f"CM": StrategicCommunityMediator.Supertype(
                    discount=0.8,
                    cap_var=0.8
                )    
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
                    #discount=UniformFloatSampler(0.2, 1),
                    std_dev=UniformFloatSampler(0.015, 0.1),
                    #std_dev=UniformFloatSampler(0.0, 0.0)
                )    
            }
        )
        policies = {"prosumer_policy": follower_agents}


    if setup_type == 'multi':
        rollout_length=5
        agent_supertypes.update(
            {
                f"H{i}": StrategicProsumerAgent.Supertype(
                    capacity = UniformIntSampler(1, 3),
                    #capacity = 1,
                    eta=UniformFloatSampler(eta, eta),
                    rollout=0
                )    
                for i in range(1, no_agents+1)
            }
        )
        agent_supertypes.update(
            {
                f"CM": StrategicCommunityMediator.Supertype(
                    discount=0.8,
                    cap_var=0.5,
                    dso_penalty=75,
                    lagrange_mult=0, # 0 for penalty objective, 1 for budget balance objective
                    lagrange_lr=0,
                    rollout_length=rollout_length,
                    # range for langrange multiplier to update through training?
                )    
            }
        )

        policies = {
            "prosumer_policy": (
                TrainedPolicy,
                follower_agents
            ),
            #"prosumer_policy": follower_agents,
            "mediator_policy": ["CM"]
        }

    ph.utils.rllib.train(
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
            "model": {"custom_model": "torch_action_mask_model"},
            "lr": 0.00001,
            "entropy_coeff": 0.125,
            "lambda": 0.98,
            "gamma": 0.998,
            #"grad_clip": 7.6,
            #"value_loss_coeff": 0.24,
            "rollout_fragment_length": 48*rollout_length,
            "num_sgd_iter": 10,
            "train_batch_size": NUM_EPISODE_STEPS*4,
            "sgd_minibatch_size": int(NUM_EPISODE_STEPS/10),
        },
        iterations=300,
        checkpoint_freq=1,
        policies=policies,
        metrics=metrics,
        num_workers=4,
        results_dir="~/ray_results/community_multi_combined",
    )

# This is used for simple runs, fx debugging locked states.
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
            for i in range(1, no_agents+1)
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
