import sys
import matplotlib.pyplot as plt
import phantom as ph
import cloudpickle
import os
from agents import SimpleProsumerAgent, SimpleCommunityMediator, StrategicProsumerAgent, StrategicCommunityMediator
import stackelberg_custom
from stackelberg_reward import StackelbergRewardDelayEnv
from datamanager import DataManager
from setup import Setup
from phantom.utils.samplers import UniformFloatSampler, UniformIntSampler
from trained_policy import TrainedPolicy

from ray.rllib.examples.models.action_mask_model import TorchActionMaskModel
from ray.rllib.models import ModelCatalog

# Register the model
ModelCatalog.register_custom_model("torch_action_mask_model", TorchActionMaskModel)


# Params
NUM_EPISODE_STEPS = 48*30
eta = 0.1 # should this be trainable?
greed = 0.8
rotate = False
no_agents = 5
setup_type = sys.argv[2]

dm = DataManager(prod_path='data/eval/pv.csv', demand_path='data/fullyearPV_singleDemand/demandprofiles.csv', cap_path='data/eval/caps.csv')
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
metrics["cm/budget_balance"] = ph.metrics.SimpleAgentMetric("CM", "budget_balance")
metrics["cm/normed_balance"] = ph.metrics.SimpleAgentMetric("CM", "normed_balance")
metrics["cm/mediator_netloss"] = ph.metrics.SimpleAgentMetric("CM", "mediator_netloss")
metrics["cm/capacity_balance"] = ph.metrics.SimpleAgentMetric("CM", "capacity_balance")
metrics["cm/capacity_limit"] = ph.metrics.SimpleAgentMetric("CM", "current_cap_limit")
metrics["cm/total_import"] = ph.metrics.SimpleAgentMetric("CM", "current_total_import")
metrics["cm/total_export"] = ph.metrics.SimpleAgentMetric("CM", "current_total_export")
metrics["cm/current_grid_price"] = ph.metrics.SimpleAgentMetric("CM", "current_grid_price")
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
# metrics["cm/rewards"] = ph.metrics.SimpleAgentMetric("CM", "acc_reward")


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

    #metrics[f"{aid}/reward"] = ph.metrics.SimpleAgentMetric(aid, "reward")
    #metrics[f"{aid}/type.capacity"] = ph.metrics.SimpleAgentMetric(aid, "type.capacity")
    

##############################################################
# RUN

if sys.argv[1] == "rollout":

    agent_supertypes = {}
    custom_policy_mapping = {}

    if setup_type == 'multi':
        #directory = "~/ray_results/community_flex_balance_multi/LATEST"
        directory = "~/ray_results/param_search/config0/LATEST"
        #directory = "/Users/antonlenander/ray_results/community_flex_balance2/entropy0_onlynetlossobservation"
        agent_supertypes.update(
            {
                f"H{i}": StrategicProsumerAgent.Supertype(
                    #capacity=2,
                    eta=eta,
                    rollout=1
                )    
                for i in range(1, no_agents+1)
            }
        )
        agent_supertypes.update(
            {
                "CM": StrategicCommunityMediator.Supertype(
                    cap_var=0.5,
                    discount=0.8,
                    rollout=1,
                    dso_penalty=15,
                    lagrange_mult=0,
                    lagrange_lr=0
                )    
            }
        )
        custom_policy_mapping.update(
            {
                f"H{i}": TrainedPolicy for i in range(1, no_agents+1)
            }
        )


    ##############
    # Copy setup
    ##############
    if setup_type == 'copy':
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
                    eta=0.05,
                    price_multiplier=2,
                    rollout=1,
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
                    lagrange_mult=0, # 0 for penalty objective, 1 for budget balance objective
                    lagrange_lr=0,
                    rollout=1,
                    reward_scale=1000,
                    no_agents=no_agents # 1 to deactivate resets of netloss
                )    
            }
        )

    results = ph.utils.rllib.rollout(
        directory="~/ray_results/new_multi_2/LATEST",
        env_class=StackelbergRewardDelayEnv,
        env_config={
            'num_steps': NUM_EPISODE_STEPS,
            'network': network,
            'leader_agents': leader_agents,
            'follower_agents': follower_agents,
            'agent_supertypes': agent_supertypes,
        },
        explore=False,
        num_repeats=1,
        num_workers=1,
        metrics=metrics,
        #custom_policy_mapping=custom_policy_mapping
    )

    results = list(results)

    path = f"output/new_multi_3/"
    if not os.path.exists(path):
        os.makedirs(path)

    cloudpickle.dump(results, open(os.path.join(path, "results.pkl"), "wb"))
