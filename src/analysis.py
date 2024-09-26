import numpy as np
import pickle as pkl
import utils
import meta_simulation
import glob
import pandas as pd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str, default='LU_scenarios', choices=['LU_scenarios', 'speculator_sweep', 'eco_burden_sweep', 'lvt_rate_sweep'])

args = parser.parse_args()

scenarios = ['baseline', 'sprawl', 'density']
tax_schemes = ['SQ', 'LVT', 'ELVT']
all_files = glob.glob('outputs/*.pkl')
# lu scenarios
lu_scenarios_files = [f for f in all_files if not 'speculator_sweep' in f and not 'lvt_rate_sweep' in f and not 'eco_burden_sweep' in f and not 'results' in f]
# speculator sweep
speculator_sweep_files = [f for f in all_files if 'speculator_sweep' in f and not 'results' in f]
speculator_numbers = []
for file in speculator_sweep_files:
    # split the file name by '_' and save the results in a list
    file = file.split('_')
    # get the speculator nr
    speculator_nr = int(file[-2])
    speculator_numbers.append(speculator_nr)
# get unique speculator numbers
speculator_numbers = np.unique(speculator_numbers)
# eco burden sweep
eco_burden_sweep_files = [f for f in all_files if 'eco_burden_sweep' in f and not 'results' in f]
eco_burdens = []
for file in eco_burden_sweep_files:
    # split the file name by '_' and save the results in a list
    file = file.split('_')
    # get the speculator nr
    eco_burden = float(file[-2])
    eco_burdens.append(eco_burden)
# get unique speculator numbers
eco_burdens = np.flip( np.sort( np.unique(eco_burdens) ) )
# lvt rate sweep
lvt_rate_sweep_files = [f for f in all_files if 'lvt_rate_sweep' in f and not 'results' in f]
lvt_rates = []
for file in lvt_rate_sweep_files:
    # split the file name by '_' and save the results in a list
    file = file.split('_')
    # get the speculator nr
    lvt_rate = float(file[-2])
    lvt_rates.append(lvt_rate)
# get unique speculator numbers
lvt_rates = np.unique(lvt_rates)

def get_multi_world_multisim_history(files_to_load):
    # loop over the files
    current_results = []
    for file in files_to_load:
        print(f'Loading {file}')
        with open(file, 'rb') as f:
            history = pkl.load(f)
        # save the results
        current_results.append(history)
    # turn the results into multi_world, multi_sim_history format
    print('Processing results')
    multi_sim_history = []
    for i, history in enumerate(current_results):
        multi_sim_history.append(history)
        world_history = utils.get_world_history_from(history)
        world_history = np.expand_dims(world_history, axis=0)
        if i == 0:
            multi_world = world_history
        else:
            multi_world = np.concatenate((multi_world, world_history), axis=0)
    # delete the data to free memory
    del current_results
    # return the multi_world
    return multi_world, multi_sim_history

def stack_multi_worlds(results):
    # stack the multi_worlds so that the first index is the scenario, the second
    # index is the tax scheme
    multi_worlds = []
    for first_key in results.keys():
        multi_worlds.append([])
        for second_key in results[first_key].keys():
            multi_worlds[-1].append(results[first_key][second_key])
        multi_worlds[-1] = np.stack(multi_worlds[-1], axis=0)
    multi_worlds = np.stack(multi_worlds, axis=0)
    print(multi_worlds.shape)
    return multi_worlds

def run_LU_scenarios():
    # create a dictionary to store the results
    results = {}
    # loop over scenarios
    for scenario in scenarios:
        # loop over tax schemes
        for tax_scheme in tax_schemes:
            print(f'Processing {scenario} scenario with {tax_scheme} tax scheme')
            # find the file names that matches the scenario and tax scheme
            # and save them in a list (due to memory issues, we will only
            # load the first 10 files if there are more than 10 files)
            files_to_load = [f for f in lu_scenarios_files if f'{scenario}_{tax_scheme}' in f][:20]
            multi_world, multi_sim_history = get_multi_world_multisim_history(files_to_load)
            # calculate results
            df = meta_simulation.monte_carlo_metrics(multi_world, multi_sim_history)
            # delete the data to free memory
            del multi_world
            del multi_sim_history
            # create a list to store the results
            if scenario not in results.keys():
                results[scenario] = {}
            if tax_scheme not in results[scenario].keys():
                results[scenario][tax_scheme] = []
            # append the results
            results[scenario][tax_scheme].append(df)
    # save the results
    print('Saving results')
    with open(f'outputs/results_{args.experiment_name}.pkl', 'wb') as f:
        pkl.dump(results, f)

def run_speculator_sweep():
    # create a dictionary to store the results
    results = {}
    # loop over speculator numbers
    for speculator_nr in speculator_numbers:
        # loop over tax schemes
        for tax_scheme in tax_schemes:
            print(f'Processing {speculator_nr} speculators with {tax_scheme} tax scheme')
            # find the file names that matches the scenario and tax scheme
            # and save them in a list (due to memory issues, we will only
            # load the first 10 files if there are more than 10 files)
            files_to_load = [f for f in speculator_sweep_files if f'_{tax_scheme}_{speculator_nr}' in f][:20]
            multi_world, multi_sim_history = get_multi_world_multisim_history(files_to_load)
            # calculate results
            df = meta_simulation.monte_carlo_metrics(multi_world, multi_sim_history)
            # delete the data to free memory
            del multi_world
            del multi_sim_history
            # create a list to store the results
            if tax_scheme not in results.keys():
                results[tax_scheme] = {}
            if speculator_nr not in results[tax_scheme].keys():
                results[tax_scheme][speculator_nr] = []
            # append the results
            results[tax_scheme][speculator_nr].append(df)
    # save the results
    print('Saving results')
    with open(f'outputs/results_{args.experiment_name}.pkl', 'wb') as f:
        pkl.dump(results, f)

def run_lvt_rate_sweep():
    tax_schemes = ['LVT', 'ELVT']
    # create a dictionary to store the results
    results = {}
    # loop over speculator numbers
    for lvt_rate in lvt_rates:
        # loop over tax schemes
        for tax_scheme in tax_schemes:
            print(f'Processing {lvt_rate} LVT rate with {tax_scheme} tax scheme')
            # find the file names that matches the scenario and tax scheme
            # and save them in a list (due to memory issues, we will only
            # load the first 10 files if there are more than 10 files)
            files_to_load = [f for f in lvt_rate_sweep_files if f'_{tax_scheme}_{lvt_rate}' in f][:20]
            multi_world, multi_sim_history = get_multi_world_multisim_history(files_to_load)
            # calculate results
            df = meta_simulation.monte_carlo_metrics(multi_world, multi_sim_history)
            # delete the data to free
            del multi_world
            del multi_sim_history
            # create a list to store the results
            if tax_scheme not in results.keys():
                results[tax_scheme] = {}
            if lvt_rate not in results[tax_scheme].keys():
                results[tax_scheme][lvt_rate] = []
            # append the results
            results[tax_scheme][lvt_rate].append(df)
    # save the results
    print('Saving results')
    with open(f'outputs/results_{args.experiment_name}.pkl', 'wb') as f:
        pkl.dump(results, f)

def run_eco_burden_sweep():
    tax_schemes = ['LVT', 'ELVT']
    # create a dictionary to store the results
    results = {}
    # loop over speculator numbers
    for eco_burden in eco_burdens:
        # loop over tax schemes
        for tax_scheme in tax_schemes:
            print(f'Processing {eco_burden} eco burden with {tax_scheme} tax scheme')
            # find the file names that matches the scenario and tax scheme
            # and save them in a list (due to memory issues, we will only
            # load the first 10 files if there are more than 10 files)
            files_to_load = [f for f in eco_burden_sweep_files if f'_{tax_scheme}_{eco_burden}' in f][:20]
            multi_world, multi_sim_history = get_multi_world_multisim_history(files_to_load)
            # calculate results
            df = meta_simulation.monte_carlo_metrics(multi_world, multi_sim_history)
            # delete the data to free
            del multi_world
            del multi_sim_history
            # create a list to store the results
            if tax_scheme not in results.keys():
                results[tax_scheme] = {}
            if eco_burden not in results[tax_scheme].keys():
                results[tax_scheme][eco_burden] = []
            # append the results
            results[tax_scheme][eco_burden].append(df)
    # save the results
    print('Saving results')
    with open(f'outputs/results_{args.experiment_name}.pkl', 'wb') as f:
        pkl.dump(results, f)

def process_results_LU_scenarios():
    # load the results file
    with open(f'outputs/results_{args.experiment_name}.pkl', 'rb') as f:
        results = pkl.load(f)

    scenarios_dfs = []
    for scenario in scenarios:
        results_scenario = results[scenario]
        tax_schemes_dfs = []
        for ts in tax_schemes:
            tax_schemes_dfs.append(results_scenario[ts][0])
        # concatenate the results
        tax_schemes_dfs = pd.concat(tax_schemes_dfs, keys=tax_schemes)
        tax_schemes_dfs.index.names = ['tax_scheme', 'stat']
        # append the results
        scenarios_dfs.append(tax_schemes_dfs)
    # concatenate the results
    scenarios_dfs = pd.concat(scenarios_dfs, keys=scenarios)
    scenarios_dfs.index.names = ['scenario', 'tax_scheme', 'stat']
    print(scenarios_dfs)
    # save the results
    scenarios_dfs.to_csv(f'outputs/results_{args.experiment_name}.csv')

def process_results_speculator_sweep():
    # load the results file
    with open(f'outputs/results_{args.experiment_name}.pkl', 'rb') as f:
        results = pkl.load(f)

    speculators = list(results['LVT'].keys())

    tax_schemes_dfs = []
    for ts in tax_schemes:
        speculators_dfs = []
        for speculator in speculators:
            speculators_dfs.append(results[ts][speculator][0])
        # concatenate the results
        speculators_dfs = pd.concat(speculators_dfs, keys=speculators)
        speculators_dfs.index.names = ['speculator', 'stat']
        # append the results
        tax_schemes_dfs.append(speculators_dfs)
    # concatenate the results
    tax_schemes_dfs = pd.concat(tax_schemes_dfs, keys=tax_schemes)
    tax_schemes_dfs.index.names = ['tax_scheme', 'speculator', 'stat']
    print(tax_schemes_dfs)
    # save the results
    tax_schemes_dfs.to_csv(f'outputs/results_{args.experiment_name}.csv')

def process_results_eco_burden_sweep():
    # load the results file
    with open(f'outputs/results_{args.experiment_name}.pkl', 'rb') as f:
        results = pkl.load(f)

    tax_schemes = ['LVT', 'ELVT']

    tax_schemes_dfs = []
    for ts in tax_schemes:
        eco_burdens_dfs = []
        for eco_burden in eco_burdens:
            eco_burdens_dfs.append(results[ts][eco_burden][0])
        # concatenate the results
        eco_burdens_dfs = pd.concat(eco_burdens_dfs, keys=eco_burdens)
        eco_burdens_dfs.index.names = ['eco_burden', 'stat']
        # append the results
        tax_schemes_dfs.append(eco_burdens_dfs)
    # concatenate the results
    tax_schemes_dfs = pd.concat(tax_schemes_dfs, keys=tax_schemes)
    tax_schemes_dfs.index.names = ['tax_scheme', 'eco_burden', 'stat']
    print(tax_schemes_dfs)
    # save the results
    tax_schemes_dfs.to_csv(f'outputs/results_{args.experiment_name}.csv')

def process_results_lvt_rate_sweep():
    # load the results file
    with open(f'outputs/results_{args.experiment_name}.pkl', 'rb') as f:
        results = pkl.load(f)

    tax_schemes = ['LVT', 'ELVT']

    tax_schemes_dfs = []
    for ts in tax_schemes:
        lvt_rates_dfs = []
        for lvt_rate in lvt_rates:
            lvt_rates_dfs.append(results[ts][lvt_rate][0])
        # concatenate the results
        lvt_rates_dfs = pd.concat(lvt_rates_dfs, keys=lvt_rates)
        lvt_rates_dfs.index.names = ['lvt_rate', 'stat']
        # append the results
        tax_schemes_dfs.append(lvt_rates_dfs)
    # concatenate the results
    tax_schemes_dfs = pd.concat(tax_schemes_dfs, keys=tax_schemes)
    tax_schemes_dfs.index.names = ['tax_scheme', 'lvt_rate', 'stat']
    print(tax_schemes_dfs)  
    # save the results
    tax_schemes_dfs.to_csv(f'outputs/results_{args.experiment_name}.csv')


def multi_worlds_LU_scenarios():
    # create a dictionary to store the results
    results = {}

    # loop over scenarios
    for scenario in scenarios:
        # loop over tax schemes
        for tax_scheme in tax_schemes:
            print(f'Processing {scenario} scenario with {tax_scheme} tax scheme')
            # find the file names that matches the scenario and tax scheme
            # and save them in a list (due to memory issues, we will only
            # load the first 10 files if there are more than 10 files)
            files_to_load = [f for f in lu_scenarios_files if f'{scenario}_{tax_scheme}' in f][:20]
            multi_world, _ = get_multi_world_multisim_history(files_to_load)
            del _
            # create a list to store the results
            if scenario not in results.keys():
                results[scenario] = {}
            if tax_scheme not in results[scenario].keys():
                results[scenario][tax_scheme] = None
            # append the results
            results[scenario][tax_scheme] = multi_world
    multi_worlds = stack_multi_worlds(results)
    # save the results
    print('Saving results')
    with open(f'outputs/results_{args.experiment_name}_multi_worlds.pkl', 'wb') as f:
        pkl.dump(multi_worlds, f)

def multi_worlds_speculator_sweep():
    # create a dictionary to store the results
    results = {}
    # loop over scenarios
    for speculator_nr in speculator_numbers:
        # loop over tax schemes
        for tax_scheme in tax_schemes:
            print(f'Processing {speculator_nr} speculator with {tax_scheme} tax scheme')
            # find the file names that matches the scenario and tax scheme
            # and save them in a list (due to memory issues, we will only
            # load the first 10 files if there are more than 10 files)
            files_to_load = [f for f in speculator_sweep_files if f'_{tax_scheme}_{speculator_nr}' in f][:20]
            multi_world, _ = get_multi_world_multisim_history(files_to_load)
            # delete the data to free memory
            del _
            # create a list to store the results
            if speculator_nr not in results.keys():
                results[speculator_nr] = {}
            if tax_scheme not in results[speculator_nr].keys():
                results[speculator_nr][tax_scheme] = None
            # append the results
            results[speculator_nr][tax_scheme] = multi_world
    multi_worlds = stack_multi_worlds(results)
    # save the results
    print('Saving results')
    with open(f'outputs/results_{args.experiment_name}_multi_worlds.pkl', 'wb') as f:
        pkl.dump(multi_worlds, f)

def multi_worlds_eco_burden_sweep():
    tax_schemes = ['LVT', 'ELVT']
    # create a dictionary to store the results
    results = {}
    # loop over scenarios
    for eco_burden in eco_burdens:
        # loop over tax schemes
        for tax_scheme in tax_schemes:
            print(f'Processing {eco_burden} eco burden with {tax_scheme} tax scheme')
            # find the file names that matches the scenario and tax scheme
            # and save them in a list (due to memory issues, we will only
            # load the first 10 files if there are more than 10 files)
            files_to_load = [f for f in eco_burden_sweep_files if f'_{tax_scheme}_{eco_burden}' in f][:20]
            multi_world, _ = get_multi_world_multisim_history(files_to_load)
            # delete the data to free memory
            del _
            # create a list to store the results
            if eco_burden not in results.keys():
                results[eco_burden] = {}
            if tax_scheme not in results[eco_burden].keys():
                results[eco_burden][tax_scheme] = None
            # append the results
            results[eco_burden][tax_scheme] = multi_world
    multi_worlds = stack_multi_worlds(results)
    # save the results
    print('Saving results')
    with open(f'outputs/results_{args.experiment_name}_multi_worlds.pkl', 'wb') as f:
        pkl.dump(multi_worlds, f)

def multi_worlds_lvt_rate_sweep():
    tax_schemes = ['LVT', 'ELVT']
    # create a dictionary to store the results
    results = {}
    # loop over scenarios
    for lvt_rate in lvt_rates:
        # loop over tax schemes
        for tax_scheme in tax_schemes:
            print(f'Processing {lvt_rate} lvt rate with {tax_scheme} tax scheme')
            # find the file names that matches the scenario and tax scheme
            # and save them in a list (due to memory issues, we will only
            # load the first 10 files if there are more than 10 files)
            files_to_load = [f for f in lvt_rate_sweep_files if f'_{tax_scheme}_{lvt_rate}' in f][:20]
            multi_world, _ = get_multi_world_multisim_history(files_to_load)
            # delete the data to free memory
            del _
            # create a list to store the results
            if lvt_rate not in results.keys():
                results[lvt_rate] = {}
            if tax_scheme not in results[lvt_rate].keys():
                results[lvt_rate][tax_scheme] = None
            # append the results
            results[lvt_rate][tax_scheme] = multi_world
    multi_worlds = stack_multi_worlds(results)
    # save the results
    print('Saving results')
    with open(f'outputs/results_{args.experiment_name}_multi_worlds.pkl', 'wb') as f:
        pkl.dump(multi_worlds, f)

if __name__=='__main__':
    if args.experiment_name == 'LU_scenarios':
        run_LU_scenarios()
        process_results_LU_scenarios()
        multi_worlds_LU_scenarios()
    elif args.experiment_name == 'speculator_sweep':
        run_speculator_sweep()
        process_results_speculator_sweep()
        multi_worlds_speculator_sweep()
    elif args.experiment_name == 'eco_burden_sweep':
        run_eco_burden_sweep()
        process_results_eco_burden_sweep()
        multi_worlds_eco_burden_sweep()
    elif args.experiment_name == 'lvt_rate_sweep':
        run_lvt_rate_sweep()
        process_results_lvt_rate_sweep()
        multi_worlds_lvt_rate_sweep()
    else:
        raise NotImplementedError
