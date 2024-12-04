import pickle

# Load the policy state
with open("/Users/antonlenander/ray_results/community_market_multi/PPO_CommunityEnv_2024-11-30_11-09-586iub4p4v/checkpoint_000179/policies/prosumer_policy/policy_state.pkl", 'rb') as f:
    policy_state = pickle.load(f)

# Inspect keys or class information
print(policy_state.keys())
print(policy_state['policy_spec'])  # If available