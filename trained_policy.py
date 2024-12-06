import phantom
import ray
import torch

import gymnasium as gym


# class TrainedPolicy(phantom.Policy):
#     def __init__(self, observation_space, action_space, **kwargs):
#         # Load the policy from the checkpoint once
#         print("init")
#         self.policy = ray.rllib.policy.Policy.from_checkpoint(
#             "/Users/antonlenander/ray_results/community_market_multi/PPO_CommunityEnv_2024-11-30_11-09-586iub4p4v/checkpoint_000179/policies/prosumer_policy"
#             )
#         self.model = self.policy.model
#         self.action_fn = self.policy.compute_single_action
#         super().__init__(self.model.obs_space, self.model.action_space, **kwargs)

#     def compute_action(self, observation):
#         # Directly call the cached action function
#         return self.action_fn(observation)[0]

class TrainedPolicy(phantom.Policy):
    def __init__(self, observation_space, action_space, **kwargs):
        self.policy = ray.rllib.policy.Policy.from_checkpoint(
            "/Users/antonlenander/ray_results/community_market_multi/PPO_CommunityEnv_2024-11-30_11-09-586iub4p4v/checkpoint_000179/policies/prosumer_policy"
            )
        super().__init__(self.policy.model.obs_space, self.policy.model.action_space, **kwargs)

    def compute_action(self, observation):
        return self.policy.compute_single_action(observation)[0]
        