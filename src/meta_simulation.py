import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt 
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import scipy.stats as st

import settings
import utils
from world import WORLD
from population import POPULATION
from developer import DEVELOPER
from homeowner import HOMEOWNER
from speculator import SPECULATOR
from simulation import SIMULATION



def run_sim():
    # create agents
    print('creating population')
    agents = POPULATION()
    # create world
    print('creating world')
    world = WORLD(agents)
    # create simulation
    print('creating simulation')
    sim = SIMULATION(world, agents)
    # run simulation
    for i in range(settings.MAX_TIMESTEP):
        print(f'{i} -> {settings.MAX_TIMESTEP}')
        for j in range(settings.N_NEW_AGENTS):
            # roll a dice to decide agent type
            dice = np.random.randint(1, sum(settings.AG_TYPE_WT) + 1)
            if dice <= settings.AG_TYPE_WT[0]:
                type = 'developer'
                agents.append(DEVELOPER(type, timestep=i))
            elif dice <= (settings.AG_TYPE_WT[1] + settings.AG_TYPE_WT[0]):
                type = 'homeowner'
                agents.append(HOMEOWNER(type, timestep=i))
            else:
                type = 'speculator'
                agents.append(SPECULATOR(type, timestep=i))
        sim.update(agents)
    simulation_history = sim.history
    return simulation_history, agents #adding this to be able to see agent attributes for single-run

#TODO: make run_experiment able to change parameter settings between runs
def run_experiment(n_runs):
    '''
    runs full simulation multiple times
    outputs multi_world:
        5d array: [run, time, cell characteristic, row, column]
    '''
    multi_sim_history = []
    for run in range(n_runs):
        simulation_history, agents = run_sim()
        multi_sim_history.append(simulation_history)
        world_run = utils.get_world_history_from(simulation_history)
        agents_run = simulation_history[ list(simulation_history.keys())[-1] ]['population'] # TODO: just getting the last agents rn
        world_run = np.expand_dims(world_run, axis=0)
        if run == 0:
            multi_world = world_run
        else:
            multi_world = np.concatenate((multi_world, world_run))
    return multi_world, multi_sim_history

def monte_carlo_spatial(multi_world, attribute_idx):
    '''
    for averaging a particular cell characteristic at final timestep across runs
    returns 2d array representing avg vals across space
    '''
    spatial_outcome = multi_world[:,settings.MAX_TIMESTEP - 1,attribute_idx,:,:]
    mean_spatial_outcome = np.mean(spatial_outcome, axis=0) #sums the 2d cell characteristic array for each run, starting with rows, then sums columns of row sums
    return mean_spatial_outcome #returns 2d arrays of cell values averaged across all runs

def monte_carlo_time(multi_world, attribute_idx, spatial_agg_type='sum'):
    '''
    for averaging a particular cell characteristic at each timestep across space and runs
    returns 1d arrays of mean, low and high confidence interval for each timstep
    '''
    #TODO: try to combine mean into one function, not rows, then columns
    space_time_outcome = multi_world[:,:,attribute_idx,:,:]
    if spatial_agg_type == 'sum':
        temp = np.sum(space_time_outcome, axis=2) #sums the 2d cell characteristic array for each timestep for each run, starting with rows, then sums columns of row sums
        time_outcome = np.sum(temp, axis=2)
    if spatial_agg_type == 'mean':
        temp = np.mean(space_time_outcome, axis=2) #sums the 2d cell characteristic array for each timestep for each run, starting with rows, then sums columns of row sums
        time_outcome = np.mean(temp, axis=2)
    mean = np.mean(time_outcome, axis=0)
    low_ci = [] #low 95% confidence interval band
    high_ci = [] #high 95% confidence interval band
    for t in range(settings.MAX_TIMESTEP):
        runs_t = time_outcome[:,t]
        low_ci_t, high_ci_t = st.t.interval(alpha=0.95, df=len(runs_t)-1, loc=mean[t], scale=st.sem(runs_t))
        low_ci.append(low_ci_t)
        high_ci.append(high_ci_t)
    low_ci = np.array(low_ci)
    high_ci = np.array(high_ci)
    return mean, low_ci, high_ci

def monte_carlo_metrics(multi_world, multi_sim_history): #TODO: check 95ci calc -- currently returns values lower than or higher than list min or max respectively
    ##WORLD METRICS##
    df = pd.DataFrame(index=['mean', 'se'])
    #total housing
    housing = copy.deepcopy(multi_world[:,settings.MAX_TIMESTEP - 1,1,:,:])
    housing_sum = np.sum(housing, axis=(1,2)) #calcs total housing for each run
    df['total_housing'] = [np.mean(housing_sum), st.sem(housing_sum)]
    #average acres per housing unit
    lotsize = copy.deepcopy(multi_world[:,settings.MAX_TIMESTEP - 1,13,:,:])
    lotsize_per_hu = lotsize
    lotsize_per_hu[housing > 1] = lotsize[housing > 1] / housing[housing > 1]
    occupied_housing = housing - copy.copy(multi_world[:,settings.MAX_TIMESTEP - 1,7,:,:])
    mask = occupied_housing == 0
    masked_lotsize_per_acre = np.ma.masked_array(lotsize_per_hu, mask=mask)
    masked_lotsize_per_acre *= occupied_housing
    avg_hu_lotsize = np.sum(masked_lotsize_per_acre, axis=(1,2)) / np.sum(occupied_housing, axis=(1,2))
    df['avg_acres_per_hu'] = [np.mean(avg_hu_lotsize), st.sem(avg_hu_lotsize)]
    #high density housing
    high_density_housing = copy.deepcopy(housing)
    density = copy.deepcopy(multi_world[:,settings.MAX_TIMESTEP - 1,5,:,:])
    high_density_housing[density < 0.4] = 0
    high_density_housing_rt = np.sum(high_density_housing, axis=(1,2)) / housing_sum * 100
    df['high_density_housing_rt'] = [np.mean(high_density_housing_rt), st.sem(high_density_housing_rt)]
    #urban boundary housing
    x, y = np.meshgrid(np.arange(settings.SIZE), np.arange(settings.SIZE))
    dist = np.sqrt(( (x-settings.SIZE//2)**2+(y-settings.SIZE//2)**2 )) 
    mask = dist > 20
    mask = mask[np.newaxis]
    mask_list = [mask for i in range(multi_world.shape[0])]
    mask = np.concatenate((mask_list))
    housing = copy.deepcopy(multi_world[:,settings.MAX_TIMESTEP - 1,1,:,:])
    housing = np.ma.masked_array(housing, mask=mask)
    housing_within_boundary = np.sum(housing, axis=(1,2)) / housing_sum
    df['housing_within_boundary'] = [np.mean(housing_within_boundary), st.sem(housing_within_boundary)]
    #urban boundary ecological value
    ec = copy.deepcopy(multi_world[:,settings.MAX_TIMESTEP - 1,11,:,:])
    ec[ec<0] = 0
    ec_urban = np.ma.masked_array(ec, mask=mask)
    ec_urban = np.sum(ec_urban, axis=(1,2))
    df['urban_ecological_value'] = [np.mean(ec_urban), st.sem(ec_urban)]
    # development patch metrics
    multi_developed = copy.deepcopy(multi_world[:,settings.MAX_TIMESTEP - 1,0,:,:] == 2)
    multi_developed = np.split(multi_developed, multi_developed.shape[0])
    largest_patch = []
    largest_patch_hu_per_acre = []
    mean_patch = []
    for count, run in enumerate(multi_developed):
        run = np.squeeze(run)
        patches = utils.get_patches(run)
        largest_patch.append(patches.max())
        housing = copy.deepcopy(multi_world[count,settings.MAX_TIMESTEP - 1,1,:,:])
        lotsize = copy.deepcopy(multi_world[count,settings.MAX_TIMESTEP - 1,13,:,:])
        mask = copy.deepcopy(patches != patches.max())
        patch_housing = np.ma.masked_array(housing, mask=mask)
        patch_lotsize = np.ma.masked_array(lotsize, mask=mask)
        largest_patch_hu_per_acre.append(patch_housing.sum() / patch_lotsize.sum())
        sizes = utils.get_patch_sizes(run)
        mean_patch.append(np.mean(sizes))
    df['largest_patch'] = [np.mean(np.array(largest_patch)), st.sem(np.array(largest_patch))]
    df['hu_density_in_largest_patch'] = [np.mean(np.array(largest_patch_hu_per_acre)), st.sem(np.array(largest_patch_hu_per_acre))]
    df['mean_patch'] = [np.mean(np.array(mean_patch)), st.sem(np.array(mean_patch))]
    #avg vacancy for all timesteps
    vacancy = copy.deepcopy(multi_world[:,:,10,:,:]) * 100
    avg_vacancy = np.mean(vacancy, axis=(1,2,3)) #averages over time and space
    df['vacancy_rt'] = [np.mean(avg_vacancy), st.sem(avg_vacancy)]
    #avg max dLV for all timesteps
    dLV = copy.deepcopy(multi_world[:,:,3,:,:])
    max_dLV = np.max(dLV, axis=(2,3)) #max dLV for each timestep for each run
    avg_max_dLV = np.mean(max_dLV, axis=1) #average max dLV across time for each run
    df['avg_max_dLV'] = [np.mean(avg_max_dLV), st.sem(avg_max_dLV)]
    # avg housing price
    hu_price = copy.deepcopy(multi_world[:,settings.MAX_TIMESTEP - 1,9,:,:])
    # non_housing = copy.deepcopy(multi_world[:,settings.MAX_TIMESTEP - 1,1,:,:] == 0)
    # hu_price = np.ma.masked_array(hu_price, mask=non_housing)
    # avg_hu_price = np.mean(hu_price, axis=(1,2)).data
    housing = copy.deepcopy(multi_world[:,settings.MAX_TIMESTEP - 1,1,:,:])
    n = np.sum(housing, axis=(1,2))
    unit_weighted_price = hu_price * housing
    sum_price = np.sum(unit_weighted_price, axis=(1,2))
    avg_hu_price = sum_price / n
    df['avg_hu_price'] = [np.mean(avg_hu_price), st.sem(avg_hu_price)]
    #change in ecological value
    ec_init = copy.deepcopy(multi_world[:,0,11,:,:])
    lotsize = copy.deepcopy(multi_world[:,0,13,:,:])
    ec_init_acres = ec_init * lotsize
    ec_init_acres_sum = np.sum(ec_init_acres, axis=(1,2))
    ec_final = copy.deepcopy(multi_world[:,settings.MAX_TIMESTEP - 1,11,:,:])
    lotsize = copy.deepcopy(multi_world[:,settings.MAX_TIMESTEP - 1,13,:,:])
    ec_final_acres = ec_final * lotsize
    ec_final_acres_sum = np.sum(ec_final_acres, axis=(1,2))
    ec_change = (ec_final_acres_sum - ec_init_acres_sum) / ec_init_acres_sum * 100
    df['ecological_value_change'] = [np.mean(ec_change), st.sem(ec_change)]
    #urban boundary ecological value
    
    #sum of restored cells
    restored = copy.deepcopy(multi_world[:,settings.MAX_TIMESTEP - 1,8,:,:])
    restored_sum = np.sum(restored, axis=(1,2))
    df['restored'] = [np.mean(restored_sum), st.sem(restored_sum)]

    ##AGENT METRICS##
    wealth_change_homeowner = np.full(len(multi_sim_history), np.nan)
    wealth_change_developer = np.full(len(multi_sim_history), np.nan)
    wealth_change_speculator = np.full(len(multi_sim_history), np.nan)
    tax_rev_final_t = np.full(len(multi_sim_history), np.nan)
    avg_tax_homeowner = np.full(len(multi_sim_history), np.nan)
    avg_tax_developer = np.full(len(multi_sim_history), np.nan)
    avg_tax_speculator = np.full(len(multi_sim_history), np.nan)
    avg_tax_high_density = np.full(len(multi_sim_history), np.nan)
    avg_tax_mid_density = np.full(len(multi_sim_history), np.nan)
    avg_tax_low_density = np.full(len(multi_sim_history), np.nan)
    min_home_purchase_price = np.full(len(multi_sim_history), np.nan)
    avg_relocation = np.full(len(multi_sim_history), np.nan)
    avg_homeless = np.full(len(multi_sim_history), np.nan)
    for run, sim_history in enumerate(multi_sim_history):
        avg_percent_change_in_wealth = utils.get_avg_percentage_change_in_wealth_from(sim_history)
        wealth_change_homeowner[run] = avg_percent_change_in_wealth['homeowner']
        wealth_change_developer[run] = avg_percent_change_in_wealth['developer']
        wealth_change_speculator[run] = avg_percent_change_in_wealth['speculator']
        tax_rev_final_t[run] = utils.get_tax_revenue_over_time_from(sim_history)[-1]
        avg_tax = utils.get_avg_tax_paid_from(sim_history)
        avg_tax_homeowner[run] = avg_tax['homeowner']
        avg_tax_developer[run] = avg_tax['developer']
        avg_tax_speculator[run] = avg_tax['speculator']
        avg_tax_hometype = utils.get_avg_tax_bill_per_home_type_from(sim_history)
        avg_tax_high_density[run] = avg_tax_hometype['high_density']
        avg_tax_mid_density[run] = avg_tax_hometype['medium_density']
        avg_tax_low_density[run] = avg_tax_hometype['single_family']
        min_home_purchase_price[run] = utils.get_min_home_purchase_price(sim_history)
        avg_relocation[run] = utils.get_avg_relocation_from(sim_history)
        homeless_over_time = np.array(utils.get_homelessness_over_time_from(sim_history)[5:]) #from timestep 5 to final timestep
        avg_homeless[run] = homeless_over_time.mean()
    df['wealth_change_homeowner'] = [np.mean(wealth_change_homeowner), st.sem(wealth_change_homeowner)]
    df['wealth_change_developer'] = [np.mean(wealth_change_developer), st.sem(wealth_change_developer)]
    df['wealth_change_speculator'] = [np.mean(wealth_change_speculator), st.sem(wealth_change_speculator)]
    df['tax_rev_final_t'] = [np.mean(tax_rev_final_t), st.sem(tax_rev_final_t)]
    df['avg_tax_homeowner'] = [np.mean(avg_tax_homeowner), st.sem(avg_tax_homeowner)]
    df['avg_tax_developer'] = [np.mean(avg_tax_developer), st.sem(avg_tax_developer)]
    df['avg_tax_speculator'] = [np.mean(avg_tax_speculator), st.sem(avg_tax_speculator)]
    df['avg_tax_high_density'] = [np.mean(avg_tax_high_density), st.sem(avg_tax_high_density)]
    df['avg_tax_mid_density'] = [np.mean(avg_tax_mid_density), st.sem(avg_tax_mid_density)]
    df['avg_tax_low_density'] = [np.mean(avg_tax_low_density), st.sem(avg_tax_low_density)]
    df['min_home_purchase_price'] = [np.mean(min_home_purchase_price), st.sem(min_home_purchase_price)]
    df['avg_relocation'] = [np.mean(avg_relocation), st.sem(avg_relocation)]
    df['avg_homeless'] = [np.mean(avg_homeless), st.sem(avg_homeless)]
    return df


    
