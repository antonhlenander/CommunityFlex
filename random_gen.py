import numpy as np
import json

# for i in range(50):
#     max_cap = np.random.randint(1, 4, size=14)
#     with open('capacities.json', 'a') as f:
#         json.dump(max_cap.tolist(), f)
#         f.write('\n')

all_capacities = []

with open('capacities.json', 'r') as f:
    for line in f:
        capacities = json.loads(line)
        all_capacities.append(capacities)
    
#print(all_capacities)
unique_capacities = set(tuple(capacity) for capacity in all_capacities)

if len(unique_capacities) < len(all_capacities) / 14:
    print("There are duplicate lists.")
else:
    print("All lists are unique.")