import numpy as np
import copy
import settings
from utils import normalize
from developer import DEVELOPER
from speculator import SPECULATOR
from homeowner import HOMEOWNER


class POPULATION():

    def __init__(self):
        self.agents = []
        self.developers = []
        self.speculators = []
        self.homeowners = []
        self._init_agents()

    def _init_agents(self):
        timestep = 0
        for i in range(settings.N_INIT_AGENTS):
            # roll a dice to decide agent type
            dice = np.random.randint(1, sum(settings.AG_TYPE_WT) + 1)
            if dice <= settings.AG_TYPE_WT[0]:
                type = 'developer'
                agent = DEVELOPER(type, timestep=timestep)
                self.developers.append(agent)
            elif dice <= (settings.AG_TYPE_WT[1] + settings.AG_TYPE_WT[0]):
                type = 'homeowner'
                agent = HOMEOWNER(type, timestep=timestep)
                self.homeowners.append(agent)
            else:
                type = 'speculator'
                agent = SPECULATOR(type, timestep=timestep)
                self.speculators.append(agent)
            self.agents.append(agent)

    def get_agent_with_id(self, id):
        for agent in self.agents:
            if agent.id == id:
                return agent
        return None

    def __len__(self):
        return len(self.agents)

    def __getitem__(self, key):
        return self.agents[key]

    def __iter__(self):
        return iter(self.agents)

    def append(self, agent):
        assert isinstance(agent, (DEVELOPER, SPECULATOR, HOMEOWNER))
        self.agents.append(agent)
        if agent.type == 'developer':
            self.developers.append(agent)
        elif agent.type == 'speculator':
            self.speculators.append(agent)
        elif agent.type == 'homeowner':
            self.homeowners.append(agent)

    def get_history(self):
        return copy.deepcopy(self)

    # def get_reg_prefs(self):
    #     nb = []
    #     cbd = []
    #     density = []
    #     lotsize = []
    #     ideal_density = []
    #     budget_ls = []
    #     for agent in self.agents:
    #         if agent.type == 'homeowner' and len(agent.ownedbuilding) == 0:
    #             nb.append(agent.nb_pref)
    #             cbd.append(agent.cbd_pref)
    #             density.append(agent.density_pref)
    #             lotsize.append(agent.lotsize_pref)
    #             ideal_density.append(agent.ideal_density)
    #             budget_ls.append(agent.budget)
    #     reg_nb_pref = sum(nb) / len(nb) # TODO: should we worried about division by zero?
    #     reg_cbd_pref = sum(cbd) / len(cbd)
    #     reg_density_pref = sum(density) / len(density)
    #     reg_lotsize_pref = sum(lotsize) / len(lotsize)
    #     reg_ideal_density = sum(ideal_density) / len(ideal_density)
    #     budget_ls = np.array(budget_ls)
    #     budget_75 = np.percentile(budget_ls, 75)
    #     return reg_nb_pref, reg_cbd_pref, reg_density_pref, reg_lotsize_pref, reg_ideal_density, budget_75

        

if __name__ == '__main__':
    pop = POPULATION()
    print(pop)
    print(pop[0])
    print(pop[0].type)
    print( len(pop) )
    for agent in pop:
        print(agent.type)
        break
    pop.append(DEVELOPER('developer', timestep=0))
    print(pop[-1].type)



