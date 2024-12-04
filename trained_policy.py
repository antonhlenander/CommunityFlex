import phantom
import ray

import gymnasium as gym


class TrainedPolicy(phantom.Policy):
    def __init__(self, observation_space, action_space, **kwargs):
        self.policy = ray.rllib.policy.Policy.from_checkpoint(
            "/Users/antonlenander/ray_results/community_market_multi/PPO_CommunityEnv_2024-11-30_11-09-586iub4p4v/checkpoint_000179/policies/prosumer_policy"
            )
        super().__init__(self.policy.model.obs_space, self.policy.model.action_space, **kwargs)

    def compute_action(self, observation):
        return self.policy.compute_single_action(observation)[0]
        