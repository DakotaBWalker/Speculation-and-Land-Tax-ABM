import numpy as np
from scipy import ndimage
from scipy.spatial import KDTree
from copy import deepcopy

import settings


def get_patches(mx, structure = [[0,1,0],[1,1,1],[0,1,0]]): #input mx must be comprised of zeros and ones
    ''' finds the connected parts in the mx and 
    returns a matrix where each element in each connected part are equal to the size of the connected part
    '''
    patch_mx, nb_labels = ndimage.label(mx,structure) #label clusters
    sizes = ndimage.sum(mx, patch_mx, range(nb_labels + 1))
    patches = deepcopy(patch_mx)
    for i in np.arange(len(sizes)):
        patches[patch_mx == i] = sizes[i]
    patches = patches.astype('float64')
    return patches

def get_patch_sizes(mx, structure = [[0,1,0],[1,1,1],[0,1,0]]):
    ''' finds the connected parts in the mx and 
    returns a list of the size of each patch
    '''
    patch_mx, nb_labels = ndimage.label(mx,structure) #label clusters
    sizes = ndimage.sum(mx, patch_mx, range(nb_labels + 1))
    return sizes

def normalize(arr, high=1): #might want to do z standardization instead
    if arr.sum() > 0:
        normalized = ((arr - np.min(arr)) * high) / (np.max(arr) - np.min(arr) + 0.0001)
    else: 
        normalized = arr
    return normalized

def get_dist_decay(arr, exponent=1, dist_wt=1, gaus_wt=0, high=1):
    arr = deepcopy(arr)
    assert arr.max() == 1
    #distance decay
    mx = np.zeros((100,100))
    mx[arr == 1] = 1
    indices = np.argwhere(mx == 1) # Find the indices of the cells equal to 1
    tree = KDTree(indices)
    distances, _ = tree.query(np.argwhere(mx == 0)) # Compute the distance from each cell to its nearest neighbor
    dist_decay = np.zeros_like(mx) # Create an array of zeros with the same shape as arr
    dist_decay[mx == 0] = 1 / (1 + distances) # Fill in the non-zero values using the indices of the cells equal to 0
    dist_decay[mx == 1] = dist_decay.max()
    dist_decay = dist_decay** exponent
    dist_decay = normalize(dist_decay, high=high)
    #gaussian filter
    gaus = ndimage.gaussian_filter(arr, sigma = 4)
    gaus = normalize(gaus, high=high)
    #combine with weight
    combined = dist_decay*dist_wt + gaus*gaus_wt
    # combined = dist_decay**dist_wt * gaus**gaus_wt
    return combined

def get_world_history_from(simulation_history):
    if len(simulation_history) == 0:
        return None
    for timestep in simulation_history:
        if timestep == 0:
            world_history = np.ma.expand_dims(simulation_history[timestep]['world'], axis=0)
        else:
            world_history = np.ma.concatenate((world_history, np.expand_dims(simulation_history[timestep]['world'], axis=0)), axis=0)
    return world_history

def get_homelessness_over_time_from(simulation_history):
    '''
    takes the simulation history
    returns
        percentage_homelessness (list of floats): percentage of people experiencing homelessness in each timestep of the simulation
    ''' 
    assert len(simulation_history) > 0, "simulation history is empty"
    percentage_homelessness = []
    for timestep in simulation_history:
        assert 'population' in simulation_history[timestep], "population not in simulation history"
        population = simulation_history[timestep]['population']
        count_experiencing_homelessness = 0.0
        assert len(population.homeowners) > 0, "no homeowners in population"
        for agent in population.homeowners:
            if len(agent.ownedbuilding) == 0:
                count_experiencing_homelessness += 1.0
        percentage_homelessness.append(count_experiencing_homelessness / len(population.homeowners) * 100)
    return percentage_homelessness

def get_tax_revenue_over_time_from(simulation_history):
    '''
    takes the simulation history
    returns
        tax_revenue_each_timestep (list of floats): total tax revenue in each timestep of the simulation
    '''
    assert len(simulation_history) > 0, "simulation history is empty"
    tax_revenue_each_timestep = []
    for timestep in simulation_history:
        assert 'population' in simulation_history[timestep], "population not in simulation history"
        population = simulation_history[timestep]['population']
        tax_revenue = 0.0
        for agent in population:
            tax_revenue += agent.tax_bill
        tax_revenue_each_timestep.append(tax_revenue)
    return tax_revenue_each_timestep

def get_avg_tax_paid_from(simulation_history):
    '''
    takes the simulation history
    returns
        avg_tax_paid (dictionary): keys are agent types, values are the average tax paid by that agent type over the course of the simulation
    ( assumes that agents don't leave the simulation )
    '''
    assert len(simulation_history) > 0, "simulation history is empty"
    avg_tax_paid = {}
    avg_tax_paid['homeowner'] = 0.0
    avg_tax_paid['developer'] = 0.0
    avg_tax_paid['speculator'] = 0.0
    for timestep in simulation_history:
        assert 'population' in simulation_history[timestep], "population not in simulation history"
        population = simulation_history[timestep]['population']
        for agent in population.homeowners:
            avg_tax_paid['homeowner'] += agent.tax_bill
        for agent in population.developers:
            avg_tax_paid['developer'] += agent.tax_bill
        for agent in population.speculators:
            avg_tax_paid['speculator'] += agent.tax_bill
    avg_tax_paid['homeowner'] /= len(population.homeowners) #* len(simulation_history)
    avg_tax_paid['developer'] /= len(population.developers) #* len(simulation_history)
    avg_tax_paid['speculator'] /= len(population.speculators) #* len(simulation_history)
    return avg_tax_paid

def get_min_home_purchase_price(simulation_history):
    '''
    takes the simulation history
    returns
        min_home_purchase_price (float): the minimum home purchase price over the course of the simulation
    '''
    assert len(simulation_history) > 0, "simulation history is empty"
    min_home_purchase_price = np.inf
    for timestep in simulation_history:
        assert 'population' in simulation_history[timestep], "population not in simulation history"
        population = simulation_history[timestep]['population']
        for agent in population.homeowners:
            if not agent.price_paid_for_home is None and agent.price_paid_for_home < min_home_purchase_price:
                min_home_purchase_price = agent.price_paid_for_home 
    return min_home_purchase_price

def get_avg_percentage_change_in_wealth_from(simulation_history):
    '''
    takes the simulation history
    returns
        avg_percentage_change_in_wealth (dictionary): keys are agent types, values are the average percentage change in wealth (realized and unrealized) for that agent type over the course of the simulation
    '''
    assert len(simulation_history) > 0, "simulation history is empty"
    avg_percentage_change_in_wealth = {}
    avg_percentage_change_in_wealth['homeowner'] = 0.0
    avg_percentage_change_in_wealth['developer'] = 0.0
    avg_percentage_change_in_wealth['speculator'] = 0.0
    # get the last population
    last_population = simulation_history[len(simulation_history) - 1]['population']
    # find the total wealth at the end 
    agent_wealth_change = 0.0
    for agent in last_population.homeowners:
        agent_wealth_change += (100 * ((agent.wealth + agent.unrealized_wealth - agent.init_wealth) / agent.init_wealth))
    avg_percentage_change_in_wealth['homeowner'] = agent_wealth_change / len(last_population.homeowners)
    agent_wealth_change = 0.0
    for agent in last_population.developers:
        agent_wealth_change += (100 * ((agent.wealth + agent.unrealized_wealth - agent.init_wealth) / agent.init_wealth))
    avg_percentage_change_in_wealth['developer'] = agent_wealth_change / len(last_population.developers)
    agent_wealth_change = 0.0
    for agent in last_population.speculators:
        agent_wealth_change += (100 * ((agent.wealth + agent.unrealized_wealth - agent.init_wealth) / agent.init_wealth))
    avg_percentage_change_in_wealth['speculator'] = agent_wealth_change / len(last_population.speculators)
    return avg_percentage_change_in_wealth

def get_avg_relocation_from(simulation_history): # TODO: probably counts the first time they buy house as relocation
    '''
    takes the simulation history
    returns
        avg_relocation (float): the average number of relocations per homeowner over the course of the simulation
    '''
    assert len(simulation_history) > 0, "simulation history is empty"
    avg_relocation = 0.0
    for timestep in simulation_history:
        assert 'population' in simulation_history[timestep], "population not in simulation history"
        population = simulation_history[timestep]['population']
        for agent in population.homeowners:
            if not agent.price_paid_for_home is None:
                avg_relocation += 1
    avg_relocation /= len(population.homeowners) #* len(simulation_history)
    return avg_relocation

def get_avg_tax_bill_per_home_type_from(simulation_history):
    '''
    takes the simulation history
    returns
        avg_tax_bill_per_home_type (dictionary): keys are home types, values are the average tax bill per home type over the course of the simulation
    '''
    assert len(simulation_history) > 0, "simulation history is empty"
    avg_tax_bill_per_home_type = {}
    avg_tax_bill_per_home_type['high_density'] = 0.0 # < 0.125
    avg_tax_bill_per_home_type['medium_density'] = 0.0 # 0.125< < 0.25
    avg_tax_bill_per_home_type['single_family'] = 0.0 #  0.25 < 
    high_density_count = 0
    medium_density_count = 0
    single_family_count = 0
    for timestep in simulation_history:
        assert 'population' in simulation_history[timestep], "population not in simulation history"
        population = simulation_history[timestep]['population']
        for agent in population.homeowners:
            if not agent.hometype is None:
                if agent.hometype == 'high_density':
                    avg_tax_bill_per_home_type['high_density'] += agent.tax_bill
                    high_density_count += 1
                elif agent.hometype == 'medium_density':
                    avg_tax_bill_per_home_type['medium_density'] += agent.tax_bill
                    medium_density_count += 1
                elif agent.hometype == 'single_family':
                    avg_tax_bill_per_home_type['single_family'] += agent.tax_bill
                    single_family_count += 1
    if high_density_count > 0:
        avg_tax_bill_per_home_type['high_density'] /= high_density_count 
    if medium_density_count > 0:
        avg_tax_bill_per_home_type['medium_density'] /= medium_density_count 
    if single_family_count > 0:
        avg_tax_bill_per_home_type['single_family'] /= single_family_count 
    return avg_tax_bill_per_home_type

def get_percentage_of_hometypes_overtime_from(simulation_history):
    '''
    takes the simulation history
    returns
        percentage_of_hometypes_overtime (dictionary): 
            keys are home types 
            values are lists of percentages of each home type over the course of the simulation
    '''
    assert len(simulation_history) > 0, "simulation history is empty"
    percentage_of_hometypes_overtime = {}
    percentage_of_hometypes_overtime['high_density'] = []
    percentage_of_hometypes_overtime['medium_density'] = []
    percentage_of_hometypes_overtime['single_family'] = []
    for timestep in simulation_history:
        assert 'population' in simulation_history[timestep], "population not in simulation history"
        population = simulation_history[timestep]['population']
        high_density_count = 0
        medium_density_count = 0
        single_family_count = 0
        for agent in population.homeowners:
            if agent.hometype == 'high_density':
                high_density_count += 1
            elif agent.hometype == 'medium_density':
                medium_density_count += 1
            elif agent.hometype == 'single_family':
                single_family_count += 1
        percentage_of_hometypes_overtime['high_density'].append(100 * high_density_count / len(population.homeowners))
        percentage_of_hometypes_overtime['medium_density'].append(100 * medium_density_count / len(population.homeowners))
        percentage_of_hometypes_overtime['single_family'].append(100 * single_family_count / len(population.homeowners))
    return percentage_of_hometypes_overtime

def get_total_restoration_from(simulation_history):
    '''
    takes the simulation history
    returns
        avg_tax_paid (dictionary): keys are agent types, values are the average tax paid by that agent type over the course of the simulation
    ( assumes that agents don't leave the simulation )
    '''
    assert len(simulation_history) > 0, "simulation history is empty"
    total_restoration = {}
    total_restoration['homeowner'] = 0.0
    total_restoration['developer'] = 0.0
    total_restoration['speculator'] = 0.0
    population = simulation_history[settings.MAX_TIMESTEP-1]['population']
    for agent in population.homeowners:
        total_restoration['homeowner'] += agent.total_restored
    for agent in population.developers:
        total_restoration['developer'] += agent.total_restored
    for agent in population.speculators:
        total_restoration['speculator'] += agent.total_restored
    return total_restoration

if __name__=='__main__':
    arr =  np.array([[0,0,1,1,0,0],
                      [0,0,0,1,0,0],
                      [1,1,0,0,1,0],
                      [0,0,0,1,0,0]])
    print( get_patches(arr) )

