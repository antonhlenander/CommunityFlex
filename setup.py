from agents import SimpleProsumerAgent, StrategicProsumerAgent
import capacities as cp
import numpy as np
import phantom as ph

class Setup():


    def get_agents(setup_type, dm, no_agents):
        agents = []
        if setup_type == 'simple':
            for i in range(no_agents):
                agent = SimpleProsumerAgent(f'H{i+1}', 'CM', dm)
                agents.append(agent)
            return agents

        elif setup_type == 'single':
            house1 = StrategicProsumerAgent('H1', 'CM', dm)
            agents.append(house1)
            for i in range(no_agents-1):
                agent = SimpleProsumerAgent(f'H{i+2}', 'CM', dm)
                agents.append(agent)
            return agents

        elif setup_type == 'multi':
            for i in range(no_agents):
                agent = StrategicProsumerAgent(f'H{i+1}', 'CM', dm)
                agents.append(agent)
            return agents

        elif setup_type == 'multsing':
            for i in range(no_agents):
                agent = StrategicProsumerAgent(f'H{i+1}', 'CM', dm)
                agents.append(agent)
            return agents
    
    def get_supertypes_eval(setup_type, eta, greed, no_agents, no_episodes, hardcap=1):

        agent_supertypes = {}
        agent_caps = {}
        caps = cp.caps

        if setup_type == 'multi':
            for i in range(no_episodes):
                for j in range(no_agents):
                    agent_caps.setdefault(f"H{j+1}", []).append(caps[i][j])
            # Iterate over every agent in the dict
            for key, value in agent_caps.items():
                agent_supertypes.update(
                    {
                        key: StrategicProsumerAgent.Supertype(
                            capacity=value,
                            eta=eta
                        )    
                    }
                )

        elif setup_type == 'single':
            for i in range(no_episodes):
                for j in range(no_agents):
                    agent_caps.setdefault(f"H{j+1}", []).append(caps[i][j])

            # Assign the new capacity lists to agents
            for key, value in agent_caps.items():
                if key == 'H1':
                    agent_supertypes.update(
                        {
                            key: StrategicProsumerAgent.Supertype( 
                                capacity=hardcap,
                                eta=eta
                            )
                        }
                    )
                else:
                    agent_supertypes.update(
                        {
                            key: SimpleProsumerAgent.Supertype( 
                                #capacity=3,
                                capacity=ph.utils.ranges.UniformRange(1,10),
                                greed=greed,
                                eta=eta
                            )
                        }
                    )
        return agent_supertypes
        

