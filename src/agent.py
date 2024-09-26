import numpy as np
import string
from datetime import datetime
import random

import settings

class AGENT():
    '''
    Base class for agents
    attributes:
        type (string): agent type
        starting_wealth (float): agent wealth
        timestep (int): current timestep
        wealth (float): agent wealth, updated at each timestep
        wealth_t (1d np array): agent wealth at each timestep
        income (float): agent income at each timestep
        ownedland (list): list of land owned by agent
        ownedbuilding (list): list of buildings owned by agent
        restored (float)
    '''
    def __init__(self, type, timestep):
        assert type in ['speculator', 'developer', 'homeowner']
        # general attributes
        self.type = type
        self.timestep = timestep # current timestep
        # monetary attributes
        self.wealth_t = np.full(settings.MAX_TIMESTEP+1, np.nan) # tracks wealth at each timstep 
        self.income = 0 # income at each timestep
        self.tax_bill = 0
        self.unrealized_wealth = 0
        # property attributes
        self.ownedland = [] # keep track of all undeveloped land they own
        self.ownedbuilding = [] # keep track of all developed land they own
        self.restored = 0
        self.total_restored = 0
        # create a unique id for each agent
        self.id = ''.join(random.sample(string.ascii_uppercase, k=5))
        self.id += '_' + datetime.now().strftime('%H%M%S%f')
        self.id += '_' + self.type

    def act(self, world):
        '''
        Agent action
        '''
        option = self.assess_options(world)
        assert option in ['buy_land', 'buy_land_or_hu', 'develop', 'sell', 'buy_home', 'restore', 'do_nothing']
        if option == 'buy_land':
            self.buy_land(world)
        elif option == 'buy_land_or_hu':
            self.buy_land_or_hu(world)
        elif option == 'develop':
            self.develop(world)
        elif option == 'sell':
            self.sell(world)
        elif option == 'buy_home':
            self.buy_home(world)
        elif option == 'restore':
            self.restore(world)
        self.money_flow(world)
        if len(self.ownedland) > 0 or len(self.ownedbuilding) > 0:
            self.get_unrealized_wealth(world)
        self.timestep += 1
        return 

    def get_unrealized_wealth(self, world):
        land_wealth = []
        capital_wealth = []
        for idx in self.ownedland:
            land_wealth.append(world.LV[idx[0],idx[1]] * world.lotsize_mx[idx[0],idx[1]])
        for idx in self.ownedbuilding:
            if self.type == 'developer':
                capital_wealth.append(world.hu_price[idx[0],idx[1]] * (world.on_market[idx[0],idx[1]])) #temp fix but doesn't account for homeowners selling unit or developer not listing dev for sale yet
            else:
                capital_wealth.append(world.hu_price[idx[0],idx[1]])
        self.unrealized_wealth = sum(land_wealth) + sum(capital_wealth)

    
    def assess_options(self):
        '''
        Assess options available to agent
        '''
        raise NotImplementedError

    def buy_land(self):
        '''
        Buy land
        '''
        raise NotImplementedError

    def develop(self):
        '''
        Develop land
        '''
        raise NotImplementedError

    def sell(self):
        '''
        Sell land
        '''
        raise NotImplementedError

    def buy_home(self):
        '''
        Buy home
        '''
        raise NotImplementedError

    def restore(self):
        '''
        Restore land
        '''
        raise NotImplementedError

    def money_flow(self, world):
        '''
        Money flow
        '''
        raise NotImplementedError





