import sys
import matplotlib.pyplot as plt
import phantom as ph
import cloudpickle
import os
from agents import SimpleProsumerAgent, SimpleCommunityMediator, StrategicProsumerAgent, StrategicCommunityMediator
import stackelberg_custom
from datamanager import DataManager
from setup import Setup
from phantom.utils.samplers import UniformFloatSampler, UniformIntSampler

from ray.rllib.examples.models.action_mask_model import TorchActionMaskModel
from ray.rllib.models import ModelCatalog

# Register the model
ModelCatalog.register_custom_model("torch_action_mask_model", TorchActionMaskModel)


# Params
NUM_EPISODE_STEPS = 8735*2
eta = 0.1 # should this be trainable?
greed = 0.75
rotate = False
no_agents = 14
setup_type = sys.argv[2]

dm = DataManager(prod_path='data/eval/pv.csv', demand_path='data/eval/demandprofiles.csv', cap_path='data/eval/caps.csv')
mediator = SimpleCommunityMediator('CM', dm=dm,)

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
    metrics[f"{aid}/current_load"] = ph.metrics.SimpleAgentMetric(aid, "current_load")
    metrics[f"{aid}/current_prod"] = ph.metrics.SimpleAgentMetric(aid, "current_prod")
    metrics[f"{aid}/current_supply"] = ph.metrics.SimpleAgentMetric(aid, "current_supply")
    metrics[f"{aid}/current_charge"] = ph.metrics.SimpleAgentMetric(aid, "current_charge")
    metrics[f"{aid}/self_consumption"] = ph.metrics.SimpleAgentMetric(aid, "self_consumption")
    metrics[f"{aid}/current_local_bought"] = ph.metrics.SimpleAgentMetric(aid, "current_local_bought")
    metrics[f"{aid}/net_loss"] = ph.metrics.SimpleAgentMetric(aid, "net_loss")
    metrics[f"{aid}/acc_local_market_coin"] = ph.metrics.SimpleAgentMetric(aid, "acc_local_market_coin")
    metrics[f"{aid}/acc_feedin_coin"] = ph.metrics.SimpleAgentMetric(aid, "acc_feedin_coin")
    #metrics[f"{aid}/utility_prev"] = ph.metrics.SimpleAgentMetric(aid, "utility_prev")
    #metrics[f"{aid}/reward"] = ph.metrics.SimpleAgentMetric(aid, "reward")
    #metrics[f"{aid}/type.capacity"] = ph.metrics.SimpleAgentMetric(aid, "type.capacity")
    

##############################################################
# RUN
##############################################################

if sys.argv[1] == "simple":

    # Enable file logging
    ph.telemetry.logger.configure_file_logging(file_path="log.json", human_readable=False, metrics=metrics, append=True)
    # Define agent supertypes
    agent_supertypes = {}

    if setup_type == 'simple':
        agent_supertypes.update(
            {
                f"H{i}": SimpleProsumerAgent.Supertype(
                    #capacity=2,
                    greed=0.75,
                    eta=eta,
                    rollout=1
                )    
                for i in range(1, 15)
            }
        )

    # Define run params
    env = stackelberg_custom.StackelbergEnvCustom(
        num_steps=NUM_EPISODE_STEPS, 
        network=network,
        leader_agents=leader_agents,
        follower_agents=follower_agents,
        agent_supertypes=agent_supertypes
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

    agent_supertypes = {}

    if setup_type == 'single':
        directory = "~/ray_results/community_market/LATEST/"
        agent_supertypes.update(
            {
                f"H1": StrategicProsumerAgent.Supertype( 
                    capacity=2, 
                    eta=eta
                )
            }
        )
        #sample = UniformIntSampler(1,4).sample()
        agent_supertypes.update(
            {
                f"H{i}": SimpleProsumerAgent.Supertype(
                    #capacity=2,
                    greed=0.75,
                    eta=eta,
                    rollout=1
                )    
                for i in range(2, 15)
            }
        )

    if setup_type == 'multi':
        directory = "~/ray_results/community_flex/LATEST/"
        agent_supertypes.update(
            {
                f"H{i}": StrategicProsumerAgent.Supertype(
                    #capacity=2,
                    eta=eta,
                    rollout=1
                )    
                for i in range(1, 15)
            }
        )
        agent_supertypes.update(
            {
                "CM": SimpleCommunityMediator.Supertype(
                    discount=0.5
                )    
            }
        )


    results = ph.utils.rllib.rollout(
        directory=directory,
        env_class=ph.StackelbergEnv,
        env_config={
            'num_steps': NUM_EPISODE_STEPS,
            'network': network,
            'leader_agents': leader_agents,
            'follower_agents': follower_agents,
            'agent_supertypes': agent_supertypes,
        },
        num_repeats=1,
        num_workers=1,
        metrics=metrics,
    )

    results = list(results)

    path = f"output/flex"
    if not os.path.exists(path):
        os.makedirs(path)

    cloudpickle.dump(results, open(os.path.join(path, "results.pkl"), "wb"))
