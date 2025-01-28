import sys
import matplotlib.pyplot as plt
import phantom as ph
import pandas as pd
import numpy as np
import random
import os

from trained_policy import TrainedPolicy
from agents import SimpleProsumerAgent, SimpleCommunityMediator, StrategicProsumerAgent, StrategicCommunityMediator
import stackelberg_custom
from stackelberg_reward import StackelbergRewardDelayEnv
from datamanager import DataManager
from setup import Setup
from phantom.utils.samplers import UniformFloatSampler, UniformIntSampler
from utils import custom_train

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
discount = 0.5 # possibly supertype?
setup_type = "multi"

dm = DataManager(demand_path="data/fullyearPV_singleDemand/demandprofiles.csv", cap_path="data/eval/caps.csv")
mediator = StrategicCommunityMediator('CM', dm=dm, no_agents=no_agents, lagrange_mult=1, lagrange_lr=0.001)

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

metrics["CM/normed_balance"] = ph.metrics.SimpleAgentMetric("CM", "normed_balance")

agent_supertypes = {}
if setup_type == 'multi':
    agent_supertypes.update(
        {
            f"H{i}": StrategicProsumerAgent.Supertype(
                #capacity = UniformIntSampler(1, 1),
                capacity = 1,
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
                cap_var=0.8
            )    
        }
    )
    policies = {
        "prosumer_policy": (
            TrainedPolicy,
            follower_agents
        ),
        "mediator_policy": ["CM"]
    }
    
np.random.seed(321)
random.seed(321)

for i in range(25, 50):
    config = {
        "lr": 10 ** np.random.choice([0.0001, 0.0002, 0.0003, 0.0004, 0.0005],),
        "entropy_coeff": random.choice([0.01, 0.2, 0.3, 0.04, 0.05]),
        "lambda": np.random.uniform(0.9, 0.99),
        "gamma": np.random.uniform(0.8, 0.99),
        "grad_clip": np.random.uniform(6, 8),
        "value_loss_coeff": np.random.uniform(0.2, 0.4),
        "rollout_fragment_length": random.choice([48*3, 48*5, 48*7]),
        "num_sgd_iter": 100,
        "train_batch_size": 4000*2,
        "sgd_minibatch_size": 1000*2,
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
        rllib_config=config,
        iterations=200,
        checkpoint_freq=0,
        policies=policies,
        metrics=metrics,
        num_workers=4,
        results_dir=f"~/ray_results/param_search/config{i}",
    )
    os.makedirs(os.path.expanduser(f"~/ray_results/param_search/config{i}"), exist_ok=True)
    with open(os.path.expanduser(f"~/ray_results/param_search/config{i}/config.txt"), "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
