import random
import numpy as np

import settings
import copy
from agent import AGENT
from utils import normalize


class DEVELOPER(AGENT):
    '''
    Developer class
    attributes (besides those inherited from AGENT):
        boughtland (int): flag to indicate whether land was bought in a particular action step
        developedland (int): flag to indicate whether the agent should pay construction costs in a particular action step
        soldhome (int): flag to indicate whether a home was sold in a particular action step
    '''
    def __init__(self, type, timestep):
        AGENT.__init__(self, type, timestep)
        self.wealth = np.exp(np.random.normal(settings.DEVELOPER_STARTING_WEALTH_LOG, scale=0.5))
        self.init_wealth = copy.deepcopy(self.wealth)
        self.wealth_t[self.timestep] = self.wealth
        self.altruism = random.uniform(settings.DEV_ALTRUISM[0],settings.DEV_ALTRUISM[1])
        self.build_cost_per_unit = random.uniform(settings.UNIT_SF[0],settings.UNIT_SF[1]) * random.uniform(settings.BUILD_COSTS_PSF[0],settings.BUILD_COSTS_PSF[1])
        self.dev_iv_lv_ratio = random.uniform(settings.DEV_IV_LV_RATIO[0], settings.DEV_IV_LV_RATIO[1])
        self.boughtland = 0 #keep track of whether land was bought in a particular action step
        self.developedland = 0 #keep track of whether the agent should pay construction costs in a particular action step
        self.soldhome = 0 #keep track of whether a developer sells their home at a particular timestep
        self.building_on_market = False

    def assess_options(self, world):
            '''
            assesses options for buying and selling land
            '''
            die = random.random()
            if len(self.ownedbuilding) == 0 and len(self.ownedland) == 0:# and vacancy_norm <= die:
                option = 'buy_land'
            elif len(self.ownedland) > 0:
                # plot = self.ownedland[0]
                # vacancy_rt = world.vacancy_rt_mx[plot[0], plot[1]]
                # hold_profit = (world.dLV[plot[0], plot[1]] * world.LV[plot[0], plot[1]]) - (world.landtax_mx[plot[0], plot[1]] * world.LV[plot[0], plot[1]])
                # if (hold_profit > 0):
                #     option = 'do_nothing'
                # else: 
                option = 'develop'
            elif len(self.ownedbuilding) > 0 and self.building_on_market == False:
                option = 'sell'
            else:
                option = 'do_nothing'
            return option

    def buy_land(self, world):
        '''
        buys land
        '''
        # estimate costs by calculating the optimal dev ratio for each plot
        LC = world.LV * world.lotsize_mx * (1 + world.dev_landtax_mx/settings.DEVELOPER_FUTURE_DISCOUNT_RT)
        n_units = np.round(LC/self.build_cost_per_unit * self.dev_iv_lv_ratio, 0)
        n_units[n_units == 0] = 1
        COST = LC + self.build_cost_per_unit * n_units
        self.hyp_project_wtp = world.all_hyp_hu_wtp * n_units
        self.profit_buy = (self.hyp_project_wtp - COST)/COST
        #mask off limits plots by availability, cost, random, and high vacancy rt
        self.off_limits = (world.state_mx != 0) | (COST >= self.wealth) | (world.vacancy_rt_mx > 0.25)| (world.rnd_off_limits == 1) | (self.profit_buy <= 0) #TODO: magic number
        self.profit_buy = np.ma.masked_array(self.profit_buy, mask = self.off_limits) #apply mask so as not to count developed plots
        self.choices = np.argwhere(self.profit_buy >= self.profit_buy.max() - np.abs(self.profit_buy.max() * settings.CHOICE_PCT))
        if len(self.choices) >= 1:
            plot = random.choice(self.choices)
            world.state_mx[plot[0],plot[1]] = 1
            self.ownedland.append([plot[0],plot[1]]) #log all parcels owned by agent
            self.boughtland = 1
    
    def develop(self, world):
        '''
        develops land
        '''
        choices = self.ownedland #options are only those that are currently owned
        assert len(choices) == 1, 'Developer has no land to develop'
        plot = choices[0]
        n_units, cost_per_unit = self._get_optimal_dev(plot, world)
        die = random.random()
        if (n_units > 1 and world.tax_scheme == 'ELVT' and die < self.altruism):
            n_units, cost_per_unit, eco_current = self._consider_half_restoration(plot, n_units, cost_per_unit, world) #returns half-units and new cost if restoration if feasible, else variables unchanged
            if eco_current == 0.5:
                world.restored_mx[plot[0],plot[1]] = eco_current
                self.restored = 0.5
                self.total_restored += 1
        if n_units > 0:
            hu_wtp = world.get_new_hu_wtp(plot, n_units, cost_per_unit)
            if cost_per_unit * 1.15 <= hu_wtp:
                self.dev_cost = n_units * cost_per_unit - world.LV[plot[0],plot[1]] * world.lotsize_mx[plot[0],plot[1]]
                vac = copy.deepcopy(world.vacancy_rt_mx)
                vac[vac >= 0.5] = 0.5
                vac_multiplier = (1 - vac[plot[0],plot[1]] - 0.75) * 4 #normalize to between -1 and 1
                price = cost_per_unit + ((hu_wtp - cost_per_unit)/2 * (1 + vac_multiplier)) #vac_multiplier (between 0 and 1), gives what percent of dif btwn wtp and cost to add
                world.state_mx[plot[0],plot[1]] = 2
                world.housing_mx[plot[0],plot[1]] = n_units
                world.hu_price[plot[0],plot[1]] = price
                world.IV[plot[0],plot[1]] = price*n_units - world.LV[plot[0],plot[1]]
                self.ownedland.pop(0)
                self.ownedbuilding.append([plot[0],plot[1]])
                self.developedland = 1

  
    
    def sell(self, world):
        '''
        sells a home
        '''
        assert len(self.ownedbuilding) == 1
        idx = self.ownedbuilding[0]
        world.on_market[idx[0], idx[1]] = (world.housing_mx[idx[0], idx[1]]) #including number of units of dev on the market
        # put this on world's marketplace dictionary
        # first check whether someone else is selling on this plot
        if (idx[0],idx[1]) in world.marketplace.keys():
            world.marketplace[(idx[0],idx[1])].append( {'seller_id': self.id, 'nr_units': world.housing_mx[idx[0], idx[1]]} )
        # if not, create a new entry
        else:
            world.marketplace[(idx[0],idx[1])] = [ {'seller_id': self.id, 'nr_units': world.housing_mx[idx[0], idx[1]]} ]
        self.building_on_market = True

    def complete_transaction_for_sold_home(self, price, last_unit, plot):
        '''
        gets money from selling home
        '''
        self.wealth += price
        self.wealth_t[self.timestep] = self.wealth
        # remove from ownedbuilding
        if last_unit == True:
            self.ownedbuilding.pop(0)
            self.hometype = None
            self.building_on_market = False
        return
    
    def money_flow(self, world):
        # MONEY OUT
        if self.boughtland == 1:
            plot = self.ownedland[-1]
            price = world.LV[plot[0],plot[1]] * world.lotsize_mx[plot[0],plot[1]]
            self.wealth -= price
            self.boughtland = 0
        if self.developedland == 1:
            assert self.dev_cost > 0, 'Developer has no development cost'
            plot = self.ownedbuilding[0]
            self.wealth -= self.dev_cost
            self.developedland = 0 
            if self.restored == 0.5: 
                self.wealth -= world.rc * 0.5 * world.lotsize_mx[plot[0],plot[1]]
                self.restored = 0
            self.dev_cost = 0
        #tax bill
        landtax_bill = []
        improvtax_bill = []
        for idx in self.ownedland:
            landtax_bill.append(world.LV[idx[0],idx[1]] * world.landtax_mx[idx[0],idx[1]]  * world.lotsize_mx[idx[0],idx[1]])
        for idx in self.ownedbuilding:
            if world.on_market[idx[0],idx[1]] > 0:
                improvtax_bill.append( (world.IV[idx[0],idx[1]] * world.improvtax_mx[idx[0],idx[1]]) * (world.on_market[idx[0],idx[1]] / world.housing_mx[idx[0],idx[1]])) #approximation given multiple agents could be listing a unit in that plot
                landtax_bill.append( (world.LV[idx[0],idx[1]] * world.landtax_mx[idx[0],idx[1]]  * world.lotsize_mx[idx[0],idx[1]]) * (world.on_market[idx[0],idx[1]] / world.housing_mx[idx[0],idx[1]]) )  #approximation given multiple agents could be listing a unit in that plot
            else:
                improvtax_bill.append( world.IV[idx[0],idx[1]] * world.improvtax_mx[idx[0],idx[1]] )
                landtax_bill.append(world.LV[idx[0],idx[1]] * world.landtax_mx[idx[0],idx[1]]  * world.lotsize_mx[idx[0],idx[1]])
        landtax_bill = sum(landtax_bill)
        improvtax_bill = sum(improvtax_bill)
        self.tax_bill = landtax_bill + improvtax_bill
        self.wealth -= self.tax_bill
        self.wealth_t[self.timestep] = self.wealth

    ## INTERNAL FUNCTIONS ##
    def _get_optimal_dev(self, plot, world): 
        lc = (world.LV[plot[0],plot[1]] * world.lotsize_mx[plot[0],plot[1]]) * (1 + world.dev_landtax_mx[plot[0],plot[1]] / settings.DEVELOPER_FUTURE_DISCOUNT_RT)
        unit_hard_cost = self.build_cost_per_unit * (1 + world.improvtax_mx[plot[0],plot[1]] / settings.DEVELOPER_FUTURE_DISCOUNT_RT)
        n_units = np.round(lc / unit_hard_cost * self.dev_iv_lv_ratio, 0)
        if n_units > 0:
            cost_per_unit = lc/n_units + unit_hard_cost
        else:
            n_units = 1
            cost_per_unit = lc/n_units + unit_hard_cost
        return n_units, cost_per_unit
    
    def _consider_half_restoration(self, plot, n_units, cost_per_unit, world): #TODO: make wtp affected by lower tax rate if restoration is done
        assert n_units > 1
        hu_wtp = world.get_new_hu_wtp(plot, n_units, cost_per_unit)
        lv = world.LV[plot[0],plot[1]] * world.lotsize_mx[plot[0],plot[1]]
        rc_plot = world.rc * world.lotsize_mx[plot[0],plot[1]] * 0.5
        eco_current = -1
        improvtax = world.improvtax_mx[plot[0],plot[1]]
        unit_hard_cost = self.build_cost_per_unit * (1 + improvtax/settings.DEVELOPER_FUTURE_DISCOUNT_RT)
        #recalculate costs assuming restoration takes place and half units are developed
        n_units_restore = np.round(n_units/2)
        eco_current_restore = 0.5
        eco_potent = world.eco_potent_mx[plot[0],plot[1]]
        eco_burden = (eco_potent * (-eco_current_restore))/settings.ECO_BURDEN_DENOM
        landtax_restore = world.lvt_rate + eco_burden
        lc_restore = lv * (1 + 3 * landtax_restore)
        cost_per_unit_restore = (lc_restore + rc_plot)/n_units_restore + unit_hard_cost
        if (eco_potent > 1 and cost_per_unit_restore*1.15 <= hu_wtp): #if restoration decreases tax rate and new cost per unit is affordable
            n_units = n_units_restore
            cost_per_unit = cost_per_unit_restore
            eco_current = 0.5
        return n_units, cost_per_unit, eco_current

if __name__=='__main__':
    developer = DEVELOPER(type='developer', timestep=0)
    print(developer.type)
