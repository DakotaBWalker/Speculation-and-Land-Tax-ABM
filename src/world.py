import numpy as np
import copy
from scipy.ndimage import gaussian_filter # TODO: find a filter that works better for our purposes
from scipy.signal import convolve2d
from skimage import draw
import random
import rasterio

from utils import normalize, get_patches, get_dist_decay
import settings

class WORLD():
    def __init__(self, agents):
        self.agents = agents
        self.size = settings.SIZE
        self.pr_init_development = settings.PR_INIT_DEV
        self.max_steps = settings.MAX_TIMESTEP
        self.nr_environmental_attractor = settings.NR_ENVR_ATTR
        self.nr_eco_potent_seeds = settings.NR_ECO_POTENT_SEEDS
        self.neighborhood_size = settings.NBR_SIZE
        self.sigma = settings.SIGMA
        self.tax_scheme = settings.TAX_SCHEME
        assert self.tax_scheme in ['SQ', 'LVT', 'ELVT']
        self.lvt_rate = settings.LVT_RATE
        self.ivt_rate = settings.IVT_RATE
        self.rc = settings.RESTORATION_COST
        self.initialize()
        self.marketplace = {}

    def initialize(self):
        #create CBD dist matrix
        x, y = np.meshgrid(np.arange(settings.SIZE), np.arange(settings.SIZE))
        dist = np.sqrt(( (x-settings.SIZE//2)**2+(y-settings.SIZE//2)**2 )) 
        self.cbd_dist_mx = 1 - (dist / np.sqrt((settings.SIZE/2)**2+(settings.SIZE/2)**2) )
        self.cbd_dist_mx += 0.001
        # lotsizes
        self.lotsize_mx = 0.075 * (dist + 1) + 0.25#np.full((self.size,self.size), 1).astype(float)#0.0003*dist**2.25 + 0.5 #0.25 * (dist + 1)**0.65 + 0.25 #
        e = np.random.normal(loc=0, scale=self.lotsize_mx/4, size=self.lotsize_mx.shape) #gives variability in lotsizes
        self.lotsize_mx += e
        self.lotsize_mx[self.lotsize_mx < 0.1] = 0.1

        #initial states, housing, and on_market
        self.state_mx = np.zeros((self.size,self.size))
        self.housing_mx = np.zeros((self.size,self.size))
        self.on_market = np.zeros((self.size,self.size))
        self.housing_mx = self._get_init_housing()
        self.state_mx[self.housing_mx > 0] = 2
        rnd_on_market = np.random.choice([1, 0], size=(self.size,self.size), p=[0.20, .80])
        self.on_market[self.state_mx == 2] = rnd_on_market[self.state_mx == 2]
        conserved = self._get_init_conserved()
        self.state_mx[conserved == 1] = -1
        self.housing_mx[self.state_mx == -1] = 0
        self.on_market[self.state_mx == -1] = 0
        self.vacancy_rt_mx = self._get_vacancy_mx()
        self.restored_mx = self._init_restored_mx()
        self.nbr_restored_mx = convolve2d(self.restored_mx, np.ones((settings.NBR_SIZE,settings.NBR_SIZE)), mode='same', boundary='wrap') / (settings.NBR_SIZE*settings.NBR_SIZE) * 2 #because restored = 0.5
        self._init_eco_current_mx()
        #create environmental value mx
        self.NB = self._init_natural_beauty_mx() 
        #create density mx
        self.density_mx = self._get_density_mx()
        #create ecological potential matrix
        self.eco_potent_mx = self._init_eco_potent_mx()
        self.landtax_mx, self.improvtax_mx = self._get_tax_rate()
        self.ideal_landtax_mx = self._get_hyp_restored_tax_rate() #tax rate if restoration occurs
        self.dev_landtax_mx = self._get_hyp_developed_tax_rate() #tax rate if dev occurs with no restoration
        self.rnd_off_limits = np.random.choice([0, 1], size=(self.size,self.size), p=[1-settings.RND_OFF_LIMITS, settings.RND_OFF_LIMITS])

        # get land values
        self.initialize_property_values()
        self.get_land_wtp()
        self.get_all_hyp_hu_wtp()

    def update(self, world_history):
        '''
        updates the world after agents acted on it
        '''
        
        self.nbr_restored_mx = convolve2d(self.restored_mx, np.ones((settings.NBR_SIZE,settings.NBR_SIZE)), mode='same', boundary='wrap') / (settings.NBR_SIZE*settings.NBR_SIZE) * 2 #because restored = 0.5
        # update density
        self.density_mx = self._get_density_mx()
        # calculate LVs
        self.calculate_landvalues(world_history)
        #update vacancy mx
        self.vacancy_rt_mx = self._get_vacancy_mx()
        # update IV
        self.IV *= 0.98
        vac_multiplier = copy.deepcopy(self.vacancy_rt_mx)
        vac_multiplier = (1 - vac_multiplier - 0.75) * 0.25 #normalizes numbers to between -0.0625 and 0.0625 (with zero change at vac_rt = 0.25)
        self.IV = self.IV * (1 + vac_multiplier)
        self.IV[self.IV < 0] = 0
        #update housing prices
        self.hu_price[self.state_mx == 2] = ((self.LV[self.state_mx == 2] * self.lotsize_mx[self.state_mx == 2]) + self.IV[self.state_mx == 2]) / self.housing_mx[self.state_mx == 2]
        # update eco current
        self._update_eco_current_mx()
        # update natural beauty
        self.update_natural_beauty_mx()
        # update eco potential
        # self._update_eco_potent_mx()
        # update tax rate
        self.landtax_mx, self.improvtax_mx = self._get_tax_rate()
        # update ideal tax rate
        self.ideal_landtax_mx = self._get_hyp_restored_tax_rate()
        self.dev_landtax_mx = self._get_hyp_developed_tax_rate()
        #update random subset of land available to buy
        self.rnd_off_limits = np.random.choice([0, 1], size=(self.size,self.size), p=[1-settings.RND_OFF_LIMITS, settings.RND_OFF_LIMITS])
        self.get_land_wtp()
        self.get_all_hyp_hu_wtp()

    def _init_natural_beauty_mx(self):
        NB = copy.deepcopy(self.eco_current_mx == 1).astype(float)
        NB = get_dist_decay(NB, exponent = 1, dist_wt=0.6, gaus_wt=0.4)
        NB[self.state_mx == -1] = 0
        NB = normalize(NB)
        NB[NB < 0.2] = 0.2
        return NB

    def update_natural_beauty_mx(self):
        forest = copy.deepcopy(self.eco_current_mx == 1).astype(float)
        forest = get_dist_decay(forest, exponent = 1, dist_wt=0.6, gaus_wt=0.4)
        forest[self.state_mx == -1] = 0
        forest = normalize(forest)
        gardens = copy.deepcopy(self.eco_current_mx == 0.5).astype(float)
        if gardens.max() == 1:
            gardens = get_dist_decay(gardens, exponent = 1, dist_wt=0.5, gaus_wt=0.5, high=0.5)
            gardens = normalize(gardens, high=0.75)
            self.NB = np.maximum(forest, gardens)
        self.NB -= self.density_mx/6
        self.NB[self.NB < 0.3] = 0.3
        conserved = copy.deepcopy(self.state_mx == -1).astype(float)
        conserved = get_dist_decay(conserved, exponent = 1, dist_wt=0.8, gaus_wt=0.2)
        conserved[self.state_mx == -1] = 0
        conserved = normalize(conserved, high=0.75)
        self.NB = np.maximum(self.NB, conserved)

    def _get_density_mx(self):
        '''
        returns a mx of values from 0 to 1 based on the density 
        of the surrounding nbr_size x nbr_size zone of each development
        currently weighted by units per cell (so all cells with sf houing = 0.5 all housing with 2units per cell = 1)
        '''
        tmp = copy.deepcopy(self.housing_mx)
        density_mx = convolve2d(tmp, np.ones((self.neighborhood_size,self.neighborhood_size)), mode='same', boundary='wrap')/(self.neighborhood_size*2)**2
        density_mx[density_mx < 0.02] = 0.02 #multiplicative prefs with zero equal zero
        return density_mx

    # TODO: check the intended behavior
    def _init_restored_mx(self): #need to change so it actually updates
        restored_mx = np.zeros((self.size,self.size))
        return restored_mx

    def _init_eco_current_mx(self):
        random_numpy = np.random.RandomState(0)
        self.eco_current_mx = copy.deepcopy(self.state_mx == -1).astype(float)
        dist_decay = get_dist_decay(self.eco_current_mx, exponent=0.1, dist_wt=0.5, gaus_wt=0.5)
        die = random_numpy.rand(100,100)
        self.eco_current_mx[die<dist_decay*2] = 1
        self.eco_current_mx[self.housing_mx >= 1] = -1
        condition_housing = np.all([self.housing_mx/self.lotsize_mx <= 1, self.housing_mx/self.lotsize_mx > 0], axis=0)
        self.eco_current_mx[condition_housing == True] = -0.5
        self.eco_current_mx[self.state_mx == -1] = 1
        self.eco_current_mx[self.restored_mx > 0] = self.restored_mx[self.restored_mx > 0]
    
    def _update_eco_current_mx(self):
        self.eco_current_mx[self.housing_mx >= 1] = -1
        condition_housing = np.all([self.housing_mx/self.lotsize_mx <= 1, self.housing_mx/self.lotsize_mx > 0], axis=0)
        self.eco_current_mx[condition_housing == True] = -0.5
        self.eco_current_mx[self.state_mx == -1] = 1
        self.eco_current_mx[self.restored_mx > 0] = self.restored_mx[self.restored_mx > 0]

    def _init_eco_potent_mx(self):
        eco_potent_mx = np.zeros((self.size,self.size))
        #areas next to large patches of existing ecol land get high scores
        eco_patch = copy.deepcopy(self.state_mx)
        eco_patch[eco_patch>=0] = 0
        eco_patch[eco_patch<0] = 1
        eco_patch = get_dist_decay(eco_patch, exponent = 0.75) #gaussian_filter(eco_patch, sigma=self.sigma, mode='wrap')
        eco_patch[self.state_mx == -1] = 0
        eco_patch = normalize(eco_patch, high=3)

        #predefined areas of high importance - current just a circle around center
        predef = np.zeros((self.size,self.size))
        for _ in range(self.nr_eco_potent_seeds):
            predef[ np.random.randint(30,70),np.random.randint(30,70) ] = 1 # environmental amenities
        if predef.max() == 1:
            predef = get_dist_decay(predef, exponent = 0.75) #gaussian_filter(predef, sigma=4, mode='wrap')
            predef = normalize(predef, high=3)
        eco_potent_mx = np.maximum(eco_patch, predef)
        return eco_potent_mx
    
    # def _update_eco_potent_mx(self):
    #     condition = np.all([self.eco_potent_mx < 2, self.density_mx > 0.7], axis=0)
    #     self.eco_potent_mx[condition == True] = 2


    def _get_LV(self):
        '''
        get the land value matrix given regional preferences and related cell characteristics
        '''
        self.avg_pref_mx = np.zeros((self.size,self.size))
        count = 0
        agent_sample = random.choices(self.agents.homeowners, k=50)
        for agent in agent_sample:
            if not hasattr(agent, 'land_pref_mx'):
                agent.get_pref_mx(self)
            self.avg_pref_mx += agent.land_pref_mx
            count += 1
        self.avg_pref_mx /= count
        housing = copy.deepcopy(self.housing_mx)
        on_market = copy.deepcopy(self.on_market)
        total_occ_housing = housing.sum() - on_market.sum()
        nbr_housing = convolve2d(housing, np.ones((13,13)), mode='same', boundary='wrap')
        nbr_on_market = convolve2d(on_market, np.ones((13,13)), mode='same', boundary='wrap')
        nbr_occ_housing = nbr_housing - nbr_on_market
        LV = self.avg_pref_mx * (nbr_occ_housing + 1)**settings.LV_DENSITY_EXP * total_occ_housing**settings.LV_HOUSING_EXP + settings.LV_MIN
        assert np.isnan(LV).any() == False, 'LV contains NaN values'
        return LV
    
    def update_LV(self):
        self.avg_pref_mx = np.zeros((self.size,self.size))
        count = 0
        agent_sample = random.choices(self.agents.homeowners, k=50)
        for agent in agent_sample:
            self.avg_pref_mx += agent.land_pref_mx
            count += 1
        self.avg_pref_mx /= count
        housing = copy.deepcopy(self.housing_mx)
        on_market = copy.deepcopy(self.on_market)
        total_occ_housing = housing.sum() - on_market.sum()
        nbr_housing = convolve2d(housing, np.ones((13,13)), mode='same', boundary='wrap')
        nbr_on_market = convolve2d(on_market, np.ones((13,13)), mode='same', boundary='wrap')
        nbr_occ_housing = nbr_housing - nbr_on_market
        self.LV = self.avg_pref_mx * (nbr_occ_housing + 1)**settings.LV_DENSITY_EXP * total_occ_housing**settings.LV_HOUSING_EXP + settings.LV_MIN
        assert np.isnan(self.LV).any() == False, 'LV contains NaN values'

    def _get_dLV(self, world_history):
        if world_history is None:
            dLV = np.zeros((self.size,self.size))
            return dLV
        else:
            t = world_history.shape[0]
        if t>=4:
            dLV = (self.LV - world_history[t-4,2,:,:]) / world_history[t-4,2,:,:] / 3 # percent change from 3yrs ago to current lv
        elif t==0:
            dLV = (self.LV - world_history[0,2,:,:]) / world_history[0,2,:,:] #percent change from year 0 to current
        else:
            dLV = (self.LV - world_history[0,2,:,:]) / world_history[0,2,:,:] / t #percent change from year 0 to current
        dLV = np.nan_to_num(dLV, nan=0.0) #change all nan to 0
        return dLV

    def initialize_property_values(self):
        ''' 
        initialize land values
        '''
        self.LV = self._get_LV()
        self.dLV = np.zeros_like(self.LV)
        #initialize improvement value per unit
        self.IV = np.zeros((self.size,self.size))
        self.IV[self.state_mx == 2] = settings.AVG_HU_HARD_COST * self.housing_mx[self.state_mx == 2]
        #initialize empty "housing-unit price" matrix
        self.hu_price = np.zeros((self.size,self.size))
        self.hu_price[self.state_mx == 2] = ((self.LV[self.state_mx == 2] * self.lotsize_mx[self.state_mx == 2]) + self.IV[self.state_mx == 2]) / self.housing_mx[self.state_mx == 2]

    def calculate_landvalues(self, world_history):
        '''
        calculates land values based on the current state of the world and agents preferences
        '''
        self.update_LV()
        self.dLV = self._get_dLV(world_history)

    def _get_tax_rate(self):
        landtax_mx = np.zeros((self.size,self.size))
        improvtax_mx = np.zeros((self.size,self.size))
        if self.tax_scheme == 'SQ':
            landtax_rate = self.lvt_rate
            landtax_mx.fill(landtax_rate)
            improvtax_rate = self.ivt_rate
            improvtax_mx.fill(improvtax_rate)
        elif self.tax_scheme == 'LVT':
            landtax_rate = self.lvt_rate
            landtax_mx.fill(landtax_rate)
        elif self.tax_scheme == 'ELVT':
            eco_burden = (self.eco_potent_mx * (-self.eco_current_mx))/settings.ECO_BURDEN_DENOM
            landtax_mx = self.lvt_rate + eco_burden
        return landtax_mx, improvtax_mx

    def _get_hyp_restored_tax_rate(self):
        landtax_mx = np.zeros((self.size,self.size))
        if self.tax_scheme == 'ELVT':
            eco_current_mx = np.ones((self.size,self.size))
            eco_burden = (self.eco_potent_mx * (-eco_current_mx))/settings.ECO_BURDEN_DENOM 
            landtax_mx = self.lvt_rate + eco_burden
        else:
            landtax_rate = self.lvt_rate
            landtax_mx.fill(landtax_rate)
        return landtax_mx
    
    def _get_hyp_developed_tax_rate(self):
        landtax_mx = np.zeros((self.size,self.size))
        if self.tax_scheme == 'ELVT':
            eco_current_mx = np.full((self.size,self.size), -1)
            eco_burden = (self.eco_potent_mx * (-eco_current_mx))/settings.ECO_BURDEN_DENOM 
            landtax_mx = self.lvt_rate + eco_burden
        else:
            landtax_rate = self.lvt_rate
            landtax_mx.fill(landtax_rate)
        return landtax_mx

    def get_land_wtp(self): #agregating all homeowner agent's land_wtp
        self.land_wtp = np.zeros((self.size,self.size))
        agent_sample = random.choices(self.agents.homeowners, k=30)
        count = 0
        agWTP = []
        for agent in agent_sample:
            if len(agent.ownedbuilding) == 0:
                count += 1
                if not hasattr(agent, 'land_pref_mx'):
                    agent.get_pref_mx(self)
                land_wtp = agent.land_pref_mx * agent.budget
                if count == 1:
                    agWTP = np.expand_dims(land_wtp, axis=0)
                else:
                    agWTP = np.concatenate((agWTP,np.expand_dims(land_wtp, axis=0)), axis=0)
        if len(agWTP) > 0:
            self.land_wtp = np.percentile(agWTP,75, axis=0)
        else:
            self.land_wtp = self.LV
        vac_multiplier = (0.25 - self.vacancy_rt_mx) * 0.25 #normalizes numbers to between -0.0625 and 0.0625
        self.land_wtp = self.land_wtp * (1 + vac_multiplier)
        pres_val_tax_bill = self.landtax_mx * self.LV * self.lotsize_mx / settings.HOMEOWNER_FUTURE_DISCOUNT_RT
        self.land_wtp -= pres_val_tax_bill

    def get_all_hyp_hu_wtp(self):
        self.all_hyp_hu_wtp = np.zeros((self.size, self.size))
        hyp_housing = copy.deepcopy(self.state_mx == 0).astype(float)
        non_housing = copy.deepcopy(self.state_mx != 0)
        hyp_housing = np.ma.masked_array(hyp_housing, mask=non_housing)
        LC = self.LV * self.lotsize_mx * (1 + self.dev_landtax_mx/settings.DEVELOPER_FUTURE_DISCOUNT_RT)
        n_units = np.round(LC/settings.AVG_HU_HARD_COST * settings.AVG_DEV_IV_LV_RATIO, 0)
        n_units[n_units == 0] = 1
        hyp_housing *= n_units
        lotsize = copy.deepcopy(self.lotsize_mx)
        hu_lotsize = lotsize / hyp_housing
        hu_wtp = []
        count = 0
        agent_sample = random.choices(self.agents.homeowners, k=30)
        for agent in agent_sample:
            # if len(agent.ownedbuilding) == 0:
            count += 1
            density_match = 1 - np.abs(self.density_mx - agent.ideal_density)
            density_match[density_match <= 0.05] = 0.05
            lotsize_match = 1 - np.abs(hu_lotsize - agent.ideal_lotsize)
            lotsize_match = normalize(lotsize_match)
            lotsize_match[lotsize_match <= 0] = 0.05
            if settings.PREF_WEIGHTING == 'additive':
                hu_pref = (agent.nb_hu_pref * self.NB + 
                                        agent.cbd_hu_pref * self.cbd_dist_mx + 
                                        agent.density_hu_pref * density_match + agent.lotsize_hu_pref * lotsize_match)
            elif settings.PREF_WEIGHTING == 'multiplicative':
                hu_pref = (self.NB ** agent.nb_hu_pref * 
                                self.cbd_dist_mx ** agent.cbd_hu_pref * 
                                density_match ** agent.density_hu_pref * lotsize_match ** agent.lotsize_hu_pref)
            agent_hu_wtp = hu_pref * agent.budget
            if count == 1:
                hu_wtp = np.ma.expand_dims(agent_hu_wtp, axis=0)
            else:
                hu_wtp = np.ma.concatenate((hu_wtp, np.expand_dims(agent_hu_wtp, axis=0)), axis=0)
        if len(hu_wtp) > 0:
            self.all_hyp_hu_wtp = np.ma.filled(self.all_hyp_hu_wtp, np.nan)
            self.all_hyp_hu_wtp = np.percentile(hu_wtp, 75, axis=0)
            self.all_hyp_hu_wtp = np.ma.masked_array(self.all_hyp_hu_wtp, mask=np.isnan(self.all_hyp_hu_wtp))
        vac_multiplier = (0.25 - self.vacancy_rt_mx) * 0.25 #normalizes numbers to between -0.0625 and 0.0625
        self.all_hyp_hu_wtp *= (1 + vac_multiplier)
        pres_val_tax_bill = ((self.dev_landtax_mx * self.LV * hu_lotsize / hyp_housing) + self.improvtax_mx * settings.AVG_HU_HARD_COST) / settings.HOMEOWNER_FUTURE_DISCOUNT_RT
        self.all_hyp_hu_wtp -= pres_val_tax_bill

    def get_new_hu_wtp(self, plot_idx, n_units, cost_per_unit):
        hyp_housing = copy.deepcopy(self.housing_mx)
        hyp_housing[plot_idx[0],plot_idx[1]] = n_units
        lotsize = copy.deepcopy(self.lotsize_mx)
        non_housing = hyp_housing < 1
        lotsize = np.ma.masked_array(lotsize, mask = non_housing)
        hu_lotsize = lotsize / hyp_housing
        hu_wtp = []
        agent_sample = random.choices(self.agents.homeowners, k=20)
        for agent in agent_sample:
            # if len(agent.ownedbuilding) == 0:
            density_match = 1 - np.abs(self.density_mx - agent.ideal_density)
            density_match[density_match <= 0.05] = 0.05
            lotsize_match = 1 - np.abs(hu_lotsize - agent.ideal_lotsize)
            lotsize_match = normalize(lotsize_match)
            lotsize_match[lotsize_match <= 0] = 0.05
            if settings.PREF_WEIGHTING == 'additive':
                hu_pref = (agent.nb_hu_pref * self.NB + 
                                        agent.cbd_hu_pref * self.cbd_dist_mx + 
                                        agent.density_hu_pref * density_match + agent.lotsize_hu_pref * lotsize_match)
            elif settings.PREF_WEIGHTING == 'multiplicative':
                hu_pref = (self.NB ** agent.nb_hu_pref * 
                                self.cbd_dist_mx ** agent.cbd_hu_pref * 
                                density_match ** agent.density_hu_pref * lotsize_match ** agent.lotsize_hu_pref)
            vac_multiplier = (0.25 - self.vacancy_rt_mx[plot_idx[0],plot_idx[1]]) * 0.25 #normalizes numbers to between -0.0625 and 0.0625
            agent_hu_wtp = hu_pref[plot_idx[0],plot_idx[1]] * agent.budget * (1 + vac_multiplier)
            pres_val_tax_bill = ((self.dev_landtax_mx[plot_idx[0],plot_idx[1]] * self.LV[plot_idx[0],plot_idx[1]] * hu_lotsize[plot_idx[0],plot_idx[1]] / hyp_housing[plot_idx[0],plot_idx[1]]) + self.improvtax_mx[plot_idx[0],plot_idx[1]] * cost_per_unit) / settings.HOMEOWNER_FUTURE_DISCOUNT_RT
            agent_hu_wtp -= pres_val_tax_bill
            hu_wtp.append(agent_hu_wtp)
        if len(hu_wtp) > 0: #TODO: make list of on_market homeowners in population so sample contains only on_market homeowners
            hu_wtp_75 = np.percentile(hu_wtp, 75)
        else:
            hu_wtp_75 = 0
        return hu_wtp_75
    
    def _get_init_conserved(self):
        random_numpy = np.random.RandomState(20)
        x, y = np.meshgrid(np.arange(settings.SIZE), np.arange(settings.SIZE))
        p = 1 - normalize(np.sqrt(( (x-settings.SIZE//2)**2+(y-settings.SIZE//2)**2 )))
        die = random_numpy.rand(100,100)
        die *= p
        conserved = np.zeros((settings.SIZE,settings.SIZE))
        conserved[die < 0.002] = 1
        dist_decay = get_dist_decay(conserved, exponent=0.2, dist_wt=0.8, gaus_wt=0.2)
        die = random_numpy.rand(100,100)
        conserved[die<dist_decay/6.5] = 1
        dist_decay = get_dist_decay(conserved, exponent=0.2, dist_wt=0.7, gaus_wt=0.3)
        conserved[dist_decay>0.75] = 1
        dist_decay = get_dist_decay(conserved, exponent=0.2, dist_wt=0.6, gaus_wt=0.4)
        conserved[dist_decay<0.7] = 0
        dist_decay = get_dist_decay(conserved, exponent=0.2, dist_wt=0.6, gaus_wt=0.4)
        conserved[dist_decay>0.7] = 1
        return conserved
    
    def _get_init_housing(self):
        random_numpy = np.random.RandomState(4)
        x, y = np.meshgrid(np.arange(settings.SIZE), np.arange(settings.SIZE))
        dist = np.sqrt(( (x-settings.SIZE//2)**2+(y-settings.SIZE//2)**2 )) 
        decay = 0.05*dist**1.7
        die = random_numpy.rand(100,100)
        die *= decay #weighting by distance to center (more dev near center)
        housing_mx = np.zeros((self.size,self.size))
        housing_mx[die < settings.PR_INIT_DEV] = 1
        housing_mx[die < settings.PR_INIT_DEV / 4] = 3
        housing_mx[die < settings.PR_INIT_DEV / 16] = 5
        return housing_mx

    def get_history(self):
        return np.ma.stack((self.state_mx, self.housing_mx, self.LV, self.dLV, self.NB, self.density_mx, self.eco_potent_mx, self.on_market, self.restored_mx, self.hu_price, self.vacancy_rt_mx, self.eco_current_mx, self.avg_pref_mx, self.lotsize_mx, self.IV, self.all_hyp_hu_wtp))

    def _get_vacancy_mx(self):
        vacancy = copy.deepcopy(self.on_market)
        housing = copy.deepcopy(self.housing_mx)
        vacancy = convolve2d(vacancy, np.ones((13,13)), mode='same', boundary='wrap')
        housing = convolve2d(housing, np.ones((13,13)), mode='same', boundary='wrap')
        vacancy_rt_mx = np.full((self.size,self.size), 0.1) #set medium vacancy rate for areas without housing
        vacancy_rt_mx[housing > 0] = (vacancy[housing > 0])/(housing[housing > 0])
        vacancy_rt_mx[vacancy_rt_mx < 0.05] = 0.05
        vacancy_rt_mx[vacancy_rt_mx > 0.5] = 0.5
        return vacancy_rt_mx








