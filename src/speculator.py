import random
import numpy as np

from agent import AGENT
import settings
import copy



class SPECULATOR(AGENT):
    '''
    Speculator class
    attributes (besides those inherited from AGENT):
        boughtland (int): flag to indicate whether land was bought in a particular action step
        soldland (int): flag to indicate whether land was sold in a particular action step
    '''
    def __init__(self, type, timestep):
        AGENT.__init__(self, type, timestep)
        self.wealth = np.exp(np.random.normal(settings.SPECULATOR_STARTING_WEALTH_LOG, scale=1))
        self.init_wealth = copy.deepcopy(self.wealth)
        self.wealth_t[self.timestep] = self.wealth
        self.boughtland = 0
        self.boughthome = 0
        self.soldland = 0
        self.soldhome = 0
        self.tax_bill = 0
        self.altruism = 0.01
        self.ownedbuilding_to_sell = []

    def assess_options(self, world):
        '''
        assesses options for buying and selling land
        '''
        if self.wealth > 0:
            if ((len(self.ownedland) >= 1 or len(self.ownedbuilding_to_sell) >=1) and random.random() < 0.1): # TODO: magic number
                option = 'sell'
            elif len(self.ownedland) > 0:
                option = self.consider_restoration(world)
            else:
                option = 'buy_land_or_hu' 
        else:
            option = 'sell' #currently only selling if agent needs money. Should add option to sell if the land has appreciated alot
        return option

    def buy_land_or_hu(self, world):
        '''
        buys land or one housing unit
        '''
        off_lim_types = world.state_mx != 0
        off_lim_types[world.on_market > 0] = False
        off_limits = (off_lim_types == 1) | (world.LV >= self.wealth) | (world.rnd_off_limits == 1)
        MBbuy = world.dLV * world.LV * world.lotsize_mx #dLV is percent change, so this turns it back into estimated earnings in one timestep
        MCbuy = world.LV * world.lotsize_mx * world.landtax_mx #TODO: does not account for increasing taxes for LV appreciation
        MBbuy[world.on_market > 0] = world.dLV[world.on_market > 0] * (world.LV[world.on_market > 0] * world.lotsize_mx[world.on_market > 0] / world.housing_mx[world.on_market > 0])
        MCbuy[world.on_market > 0] = ((world.LV[world.on_market > 0] * world.lotsize_mx[world.on_market > 0] / world.housing_mx[world.on_market > 0] * world.landtax_mx[world.on_market > 0]) + 
                                        world.IV[world.on_market > 0] * world.improvtax_mx[world.on_market > 0])
        self.BC_ratio = np.ma.masked_array(MBbuy / MCbuy, mask = off_limits)
        # assert np.all(self.BC_ratio >= 0), 'MBbuy is negative or nan' #TODO: figure out why BC_ratio is negative or nan
        if self.BC_ratio.max() > 0.2:
            choices = np.argwhere(self.BC_ratio > self.BC_ratio.max()-self.BC_ratio.max() * settings.CHOICE_PCT) #choose from top ~12.5% of highest MBbuy from 25% of all available
            if len(choices) >= 1:
                plot = random.choice(choices)
                if world.state_mx[plot[0],plot[1]] == 0: #buying land
                    world.state_mx[plot[0],plot[1]] = 1
                    self.ownedland.append([plot[0],plot[1]])
                    self.boughtland = 1
                elif world.state_mx[plot[0],plot[1]] == 2: #buying home
                    world.on_market[plot[0],plot[1]] -= 1 
                    assert world.on_market[plot[0],plot[1]] >= 0 #TODO: find out where they're choosing housing not on market
                    self.ownedbuilding.append([plot[0],plot[1]])
                    self.boughthome = 1
                    self.ownedbuilding_to_sell.append((plot[0],plot[1]))
                    # first find the potential seller or sellers
                    if (plot[0],plot[1]) in world.marketplace.keys(): #TODO: this might not work because removing seller keeps the key (val just contains an empty list)
                        sellers = world.marketplace[(plot[0],plot[1])]
                    else:
                        return # plot belongs to nobody, randomly initiliazed as available to buy
                    if len(sellers) > 1:
                        seller = random.choice(sellers)
                    else:
                        seller = sellers[0]
                    # then find the price
                    price_plot = world.hu_price[plot[0],plot[1]]
                    # handle the money payment to the seller
                    seller_id = seller['seller_id']
                    # decrease the number of available units in the marketplace
                    last_unit = False
                    seller['nr_units'] -= 1
                    if seller['nr_units'] == 0:
                        sellers.remove(seller)
                        last_unit = True
                    world.agents.get_agent_with_id(seller_id).complete_transaction_for_sold_home(price_plot, last_unit, [plot[0],plot[1]])
                    if len(world.marketplace[(plot[0],plot[1])]) == 0:
                        world.marketplace.pop((plot[0],plot[1])) 
        return 
    
    def sell(self, world):
            '''
            sells property
            '''
            if len(self.ownedland) >= 1:
                idx = self.ownedland[0] #assumes they sell first bought land
                world.state_mx[idx[0], idx[1]] = 0
                self.soldland = 1
            elif len(self.ownedland) == 0 and len(self.ownedbuilding_to_sell) >= 1:
                idx = self.ownedbuilding_to_sell[0] # put this on world's marketplace dictionary
                # first check whether someone else is selling on this plot
                if (idx[0],idx[1]) in world.marketplace.keys():
                    world.on_market[idx[0], idx[1]] += 1
                    world.marketplace[(idx[0],idx[1])].append( {'seller_id': self.id, 'nr_units': 1.0} )
                # if not, create a new entry
                else:
                    world.on_market[idx[0], idx[1]] += 1
                    world.marketplace[(idx[0],idx[1])] = [ {'seller_id': self.id, 'nr_units': 1.0} ]
                # remove it from ownedbuilding_to_sell
                self.ownedbuilding_to_sell.pop(0)
            return 
    
    def complete_transaction_for_sold_home(self, price, last_unit, plot):
        '''
        gets money from selling home
        '''
        self.wealth += price
        self.wealth_t[self.timestep] = self.wealth
        # remove from ownedbuilding
        if last_unit == True:
            self.ownedbuilding.remove(plot)
            self.hometype = None #TODO: hometype should just be for homeowners because the assumption is speculators dont live at their owned homes
        return
    
    def consider_restoration(self, world):
        option = 'do_nothing'
        profit_potential = []
        for idx in self.ownedland:
            if world.restored_mx[idx[0],idx[1]] == 0 and world.eco_current_mx[idx[0],idx[1]] < 0.5:
                profit_potential.append( ((world.LV[idx[0], idx[1]] * world.lotsize_mx[idx[0],idx[1]]) * (world.landtax_mx[idx[0], idx[1]] - world.ideal_landtax_mx[idx[0], idx[1]]) / settings.SPECULATOR_FUTURE_DISCOUNT_RT - (world.rc * world.lotsize_mx[idx[0], idx[1]]))
                                / (world.rc * world.lotsize_mx[idx[0], idx[1]]) )
        if len(profit_potential) > 0:
            profit_potential = np.array(profit_potential)
            if profit_potential.max() > 0.15:
                die = random.random()
                if die < profit_potential.max():
                    option = 'restore'
        return option
    
    def restore(self, world):
        profit_potential = {}
        for idx in self.ownedland:
            if world.restored_mx[idx[0],idx[1]] == 0 and world.eco_current_mx[idx[0],idx[1]] < 0.5:
                profit_potential[tuple(idx)] = ( ((world.LV[idx[0], idx[1]] * world.lotsize_mx[idx[0],idx[1]]) * (world.landtax_mx[idx[0], idx[1]] - world.ideal_landtax_mx[idx[0], idx[1]]) / settings.SPECULATOR_FUTURE_DISCOUNT_RT - (world.rc * world.lotsize_mx[idx[0], idx[1]]))
                                / (world.rc * world.lotsize_mx[idx[0], idx[1]]) )
        assert len(profit_potential) > 0, 'told to restore, but nothing to restore'
        assert max(profit_potential.values()) > 0
        plot = list(max(profit_potential, key = profit_potential.get))
        world.restored_mx[plot[0], plot[1]] = 1
        self.restored = 1
        self.total_restored += 1
        return
    
    def money_flow(self, world):
        #MONEY OUT
        if self.boughtland == 1:
            plot = self.ownedland[-1]
            cost = world.LV[plot[0],plot[1]] * world.lotsize_mx[plot[0],plot[1]]
            self.wealth -= cost
            self.boughtland = 0 
        if self.boughthome == 1:
            plot = self.ownedbuilding[-1]
            cost = world.hu_price[plot[0],plot[1]]
            self.wealth -= cost
            self.boughthome = 0
        if self.restored > 0: 
            plot = self.ownedland[0]
            self.wealth -= world.rc * self.restored * world.lotsize_mx[plot[0],plot[1]]
            self.restored = 0
        #tax bill
        landtax_bill = []
        improvtax_bill = [] 
        for idx in self.ownedland:
            landtax_bill.append(world.landtax_mx[idx[0],idx[1]] * (world.LV[idx[0],idx[1]]  * world.lotsize_mx[idx[0],idx[1]]))
        for idx in self.ownedbuilding:
            improvtax_bill.append(world.improvtax_mx[idx[0],idx[1]] * (world.IV[idx[0],idx[1]] / world.housing_mx[idx[0],idx[1]])) #TODO: check this, seems to append(0)
            landtax_bill.append(world.landtax_mx[idx[0],idx[1]] * (world.LV[idx[0],idx[1]]  * world.lotsize_mx[idx[0],idx[1]] / world.housing_mx[idx[0],idx[1]]))
        landtax_bill = sum(landtax_bill)
        improvtax_bill = sum(improvtax_bill)
        self.tax_bill = landtax_bill + improvtax_bill
        self.wealth -= self.tax_bill
        #MONEY IN
        #one time payments
        if self.soldland == 1:
            plot = self.ownedland[0] #assumes they sell first bought land
            price = world.LV[plot[0],plot[1]]  * world.lotsize_mx[plot[0],plot[1]]
            self.wealth += price #assumes immediate sale of land
            self.ownedland.pop(0)
            self.soldland = 0
        #yearly income or revenue
        self.wealth_t[self.timestep] = self.wealth


if __name__=='__main__':
    speculator = SPECULATOR(type='speculator', timestep=0)
    print(speculator.type)
    print(speculator.wealth)
    print(speculator.wealth_t)
