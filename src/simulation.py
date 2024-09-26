import numpy as np
from copy import deepcopy

from world import WORLD
from agent import AGENT #?
from utils import get_world_history_from


class SIMULATION():
    '''
    Base class for simulation
    attributes:
        world (WORLD): world object
        agents (list): list of agents
        timestep (int): current timestep
    '''
    def __init__(self, world, agents):
        # general attributes
        self.world = world
        self.agents = agents
        self.timestep = 0 # current timestep
        # record keeping. history is a dictionary. keys are timestaps, values are dictionaries with keys 'world' and 'population'
        self.history = {}

    def update(self, agents):
        # prepare for update
        self.world.get_land_wtp() # TODO: do we need this here? or should we do this in self.world.update()?
        # agents act and update world
        for agent in agents: # TODO: they act in same order every timestep. randomize
            agent.act(self.world)
        self.world.update(world_history=get_world_history_from(self.history)) 
        # record keeping
        self.history[self.timestep] = {'world': self.world.get_history(), 'population': self.agents.get_history()}
        self.timestep += 1
        return
