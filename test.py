import ray.rllib

policy =  ray.rllib.policy.Policy.from_checkpoint(
            "/Users/antonlenander/ray_results/community_market_multi/PPO_CommunityEnv_2024-11-30_11-09-586iub4p4v/checkpoint_000179/policies/prosumer_policy"
            )

print(policy.model.observation_space)