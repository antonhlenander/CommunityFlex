from agents import SimpleProsumerAgent, StrategicProsumerAgent

class Setup():
    def get_agents(setup_type, dm, greed, eta, no_agents):
        agents = []
        if setup_type == 'simple':
            for i in range(no_agents):
                agent = SimpleProsumerAgent(f'H{i}+1', 'CM', dm, greed)
                agents.append(agent)
            return agents

        elif setup_type == 'single':
            house1 = StrategicProsumerAgent('H1', 'CM', dm, eta)
            for i in range(no_agents-1):
                agent = SimpleProsumerAgent(f'H{i}+2', 'CM', dm, greed)
                agents.append(agent)
            return agents

        elif setup_type == 'multi':
            for i in range(no_agents):
                agent = StrategicProsumerAgent(f'H{i}+1', 'CM', dm, eta)
                agents.append(agent)
            return agents
    