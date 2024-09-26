import random
import numpy as np
import copy

from agent import AGENT
from utils import normalize
import settings

class HOMEOWNER(AGENT):
    '''
    Homeowner class
    attributes (besides those inherited from AGENT):
        nb_pref_range ([float,float]): environmental preference range
        cbd_pref_range ([float,float]): central business district preference range
        density_pref_range ([float,float]): density preference range
        nb_pref (float): environmental preference
        cbd_pref (float): central business district preference
        density_pref (float): density preference
        budget (float): landowner budget
        boughthome (int): flag to indicate whether a landowner buys a home at a particular timestep
        soldhome (int): flag to indicate whether a landowner sells their home at a particular timestep
        income (float): landowner income
    '''
    def __init__(self, type, timestep):
        AGENT.__init__(self, type, timestep)
        # initialize landowner preferences
        nb_pref = random.uniform(settings.NB_PREF_RANGE[0], settings.NB_PREF_RANGE[1])
        cbd_pref = random.uniform(settings.CBD_PREF_RANGE[0], settings.CBD_PREF_RANGE[1])
        density_pref = random.uniform(settings.DENSITY_PREF_RANGE[0], settings.DENSITY_PREF_RANGE[1])
        lotsize_pref = random.uniform(settings.LOTSIZE_PREF_RANGE[0], settings.LOTSIZE_PREF_RANGE[1])
        #normalized land preferences (sum to one)
        total_land_prefs = nb_pref + cbd_pref + density_pref
        self.nb_land_pref = nb_pref/total_land_prefs
        self.cbd_land_pref = cbd_pref/total_land_prefs
        self.density_land_pref = density_pref/total_land_prefs
        #normalized hu preferences (sum to one)
        total_hu_prefs = nb_pref + cbd_pref + density_pref + lotsize_pref
        self.nb_hu_pref = nb_pref/total_hu_prefs 
        self.cbd_hu_pref = cbd_pref/total_hu_prefs
        self.density_hu_pref = density_pref/total_hu_prefs
        self.lotsize_hu_pref = lotsize_pref/total_hu_prefs
        self.ideal_density = random.uniform(settings.IDEAL_DENSITY_RANGE[0], settings.IDEAL_DENSITY_RANGE[1])
        self.ideal_lotsize = random.uniform(settings.IDEAL_LOTSIZE_RANGE[0], settings.IDEAL_LOTSIZE_RANGE[1])
        self.altruism = random.uniform(settings.HOMEOWNER_ALTRUISM[0], settings.HOMEOWNER_ALTRUISM[1])
        self.income = random.randint(settings.HOMEOWNER_INCOME_RANGE[0], settings.HOMEOWNER_INCOME_RANGE[1])
        # init other landowner attributes
        self.wealth = np.exp(np.random.normal(settings.HOMEOWNER_STARTING_WEALTH_LOG, scale=0.25))
        self.init_wealth = copy.deepcopy(self.wealth)
        self.wealth_t[self.timestep] = self.wealth
        self.budget = self.wealth #* 0.75
        self.boughthome = 0 #keep track of whether a landowner buys a home at a particular timestep 
        self.price_paid_for_home = None
        self.soldhome = 0
        self.hometype = None
        self.building_on_market = False


    def assess_options(self, world):
        '''
        assesses options for buying and selling land
        '''
        self.price_paid_for_home = None
        self.get_pref_mx(world)
        if len(self.ownedbuilding) == 0:
            option = 'buy_home'
        elif self.building_on_market == False:
            move_options = copy.deepcopy(self.hu_pref_mx)
            move_options[world.hu_price > self.budget] = 0
            best_option_pref = move_options.max()
            if self.wealth <= 0 or (best_option_pref - self.current_home_pref) > settings.SELL_MIN_PREF_DIF:
                option = 'sell'
            elif (settings.TAX_SCHEME == 'ELVT' and (self.wealth/5 >= world.rc * world.lotsize_mx[self.ownedbuilding[0][0],self.ownedbuilding[0][1]] * 0.5) 
                                                    and (world.lotsize_mx[self.ownedbuilding[0][0],self.ownedbuilding[0][1]] / world.housing_mx[self.ownedbuilding[0][0],self.ownedbuilding[0][1]] > 0.2)
                                                         and (world.eco_current_mx[self.ownedbuilding[0][0],self.ownedbuilding[0][1]] < 0.5)):
                option = self.consider_restoration(world)
            else:
                option = 'do_nothing' 
        else:
            option = 'do_nothing'
        return option

    def get_pref_mx(self, world):
        #land preferences
        density_match = 1 - np.abs(world.density_mx - self.ideal_density)
        density_match[density_match <= 0] = 0.05
        if settings.PREF_WEIGHTING == 'additive':
            self.land_pref_mx = (self.nb_land_pref * world.NB + 
                                    self.cbd_land_pref * world.cbd_dist_mx + 
                                    self.density_land_pref * density_match) 
        elif settings.PREF_WEIGHTING == 'multiplicative':
            self.land_pref_mx = (world.NB ** self.nb_land_pref * 
                            world.cbd_dist_mx ** self.cbd_land_pref * 
                            density_match ** self.density_land_pref)
        else:
            raise ValueError('PREF_WEIGHTING must be additive or multiplicative')
        #housing preferences
        hu_lotsize = copy.deepcopy(world.lotsize_mx)
        not_available = world.on_market < 1
        hu_lotsize = np.ma.masked_array(hu_lotsize, mask = not_available)
        hu_lotsize /= world.housing_mx
        hu_lotsize_match = 1 - np.abs(hu_lotsize - self.ideal_lotsize)
        hu_lotsize_match = normalize(hu_lotsize_match)
        hu_lotsize_match[hu_lotsize_match <= 0.05] = 0.05
        if settings.PREF_WEIGHTING == 'additive':
            self.hu_pref_mx = (self.nb_hu_pref * world.NB + 
                                self.cbd_hu_pref * world.cbd_dist_mx + 
                                self.density_hu_pref * density_match + self.lotsize_hu_pref * hu_lotsize_match)
        elif settings.PREF_WEIGHTING == 'multiplicative':
            self.hu_pref_mx = (world.NB ** self.nb_hu_pref * 
                                world.cbd_dist_mx ** self.cbd_hu_pref * 
                                density_match ** self.density_hu_pref * hu_lotsize_match ** self.lotsize_hu_pref)
        #current home preference relative to others on market
        if len(self.ownedbuilding) == 1:
            plot = self.ownedbuilding[0]
            housing = copy.deepcopy(world.housing_mx)
            hu_lotsize = copy.deepcopy(world.lotsize_mx)
            mask = world.on_market < 1
            mask[plot[0],plot[1]] = False
            hu_lotsize = np.ma.masked_array(hu_lotsize, mask = mask)
            housing = np.ma.masked_array(housing, mask = mask)
            hu_lotsize = world.lotsize_mx / housing
            hu_lotsize_match = 1 - np.abs(hu_lotsize - self.ideal_lotsize)
            hu_lotsize_match = normalize(hu_lotsize_match)
            hu_lotsize_match[hu_lotsize_match <= 0] = 0.05
            hu_lotsize_match = hu_lotsize_match[plot[0],plot[1]]
            density_match = (1 - np.abs(world.density_mx - self.ideal_density))[plot[0],plot[1]]
            nb = world.NB[plot[0],plot[1]]
            cbd =  world.cbd_dist_mx[plot[0],plot[1]]
            if settings.PREF_WEIGHTING == 'additive':
                self.current_home_pref = (self.nb_hu_pref * nb + 
                                    self.cbd_hu_pref * cbd + 
                                    self.density_hu_pref * density_match + self.lotsize_hu_pref * hu_lotsize_match) 
            elif settings.PREF_WEIGHTING == 'multiplicative':
                self.current_home_pref = (nb ** self.nb_hu_pref * 
                                    cbd ** self.cbd_hu_pref * 
                                    density_match ** self.density_hu_pref * hu_lotsize_match ** self.lotsize_hu_pref)

    def buy_home(self, world):
        '''
        buy home
        '''
        housing = copy.deepcopy(world.housing_mx)
        mask = world.on_market == 0
        housing = np.ma.masked_array(housing, mask=mask)
        price = copy.deepcopy(world.hu_price)
        self.hu_wtp = self.hu_pref_mx * self.budget
        pres_val_tax_bill = ((world.landtax_mx * world.LV * world.lotsize_mx + world.improvtax_mx * world.IV) / housing) / settings.HOMEOWNER_FUTURE_DISCOUNT_RT
        self.hu_wtp -= pres_val_tax_bill
        off_limits = (world.on_market == 0) | (price > self.hu_wtp)
        msk_pref_mx = np.ma.masked_array(self.hu_pref_mx, mask = off_limits) #apply mask so as not to count developed plots
        msk_pref_mx = normalize(msk_pref_mx) # TODO: decide whether to normalize or not
        choices = np.argwhere(msk_pref_mx > (1 - settings.CHOICE_PCT))
        if len(choices) >= 1:
            plot = random.choice(choices) # choose plot based on random choice based on preferences only
            world.on_market[plot[0],plot[1]] -= 1 
            assert world.on_market[plot[0],plot[1]] >= 0
            self.ownedbuilding.append([plot[0],plot[1]])
            self.boughthome = 1
            if world.lotsize_mx[plot[0],plot[1]] / world.housing_mx[plot[0],plot[1]] < 0.125:
                self.hometype = 'high_density'
            elif world.lotsize_mx[plot[0],plot[1]] / world.housing_mx[plot[0],plot[1]] < 0.25:
                self.hometype = 'medium_density'
            else:
                self.hometype = 'single_family'
            # first find the potential seller or sellers
            if (plot[0],plot[1]) in world.marketplace.keys():
                sellers = world.marketplace[(plot[0],plot[1])]
            else:
                return # plot belongs to nobody, randomly initiliazed as available to buy
            if len(sellers) > 1:
                seller = random.choice(sellers) #TODO: change to make first unit put on market be the one that sells
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
            

    def sell(self, world):
        '''
        sells a home
        '''
        assert len(self.ownedbuilding) == 1
        idx = self.ownedbuilding[0]
        world.on_market[idx[0], idx[1]] += 1 
        # put this on world's marketplace dictionary
        # first check whether someone else is selling on this plot
        if (idx[0],idx[1]) in world.marketplace.keys():
            world.marketplace[(idx[0],idx[1])].append( {'seller_id': self.id, 'nr_units': 1} )
        # if not, create a new entry
        else:
            world.marketplace[(idx[0],idx[1])] = [ {'seller_id': self.id, 'nr_units': 1} ]
        self.building_on_market = True

    def complete_transaction_for_sold_home(self, price, last_unit, plot):
        '''
        gets money from selling home
        '''
        self.wealth += price
        self.wealth_t[self.timestep] = self.wealth
        # remove the plot from the agent's ownedbuilding list
        if last_unit == True:
            self.ownedbuilding = []
            self.hometype = None
            self.building_on_market = False
        return

    def consider_restoration(self, world):
        plot = self.ownedbuilding[0]
        savings_to_income = ((world.LV[plot[0], plot[1]] * world.lotsize_mx[plot[0],plot[1]]) 
                             / world.housing_mx[plot[0],plot[1]] * (world.landtax_mx[plot[0], plot[1]] - world.ideal_landtax_mx[plot[0], plot[1]]) / settings.HOMEOWNER_FUTURE_DISCOUNT_RT
                             / self.income)
        nbr_adopt = world.nbr_restored_mx[plot[0],plot[1]]
        p_restore = (self.altruism + nbr_adopt) * savings_to_income
        die = random.random()
        if die < p_restore:
            option = 'restore'
        else:
            option = 'do_nothing' 
        return option
    
    def restore(self, world):
        assert len(self.ownedbuilding) == 1
        plot = self.ownedbuilding[0]
        world.restored_mx[plot[0], plot[1]] = 0.5
        self.restored = 0.5
        self.total_restored += 1

    def money_flow(self, world):
        #MONEY OUT
        if self.boughthome == 1:
            assert len(self.ownedbuilding) == 1, 'no home owned'
            plot = self.ownedbuilding[0]
            self.price_paid_for_home = world.hu_price[plot[0],plot[1]]
            self.wealth -= self.price_paid_for_home
            self.boughthome = 0 #home paid for, so setting this back to zero
        elif self.restored > 0: 
            plot = self.ownedbuilding[0]
            self.wealth -= world.rc * self.restored * world.lotsize_mx[plot[0],plot[1]]
            self.restored = 0
        #tax bill
        landtax_bill = []
        improvtax_bill = []
        if len(self.ownedbuilding) > 0:
            assert len(self.ownedbuilding) == 1
            idx = self.ownedbuilding[0]
            improvtax_bill = world.improvtax_mx[idx[0],idx[1]] * (world.IV[idx[0],idx[1]] / world.housing_mx[idx[0],idx[1]]) 
            landtax_bill = world.landtax_mx[idx[0],idx[1]] * (world.LV[idx[0],idx[1]]  * world.lotsize_mx[idx[0],idx[1]] / world.housing_mx[idx[0],idx[1]])
            self.tax_bill = landtax_bill + improvtax_bill
            self.wealth -= self.tax_bill
        #MONEY IN
        self.wealth += self.income
        self.wealth_t[self.timestep] = self.wealth
        self._update_budget()

    def _update_budget(self): # TODO: add in other factors that affect budget
        self.budget = self.wealth * 0.75
    
        

# if __name__=='__main__':
    # landowner = LANDOWNER(type='landowner', timestep=0)
    # print(landowner.type)
