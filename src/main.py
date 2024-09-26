from meta_simulation import run_sim, run_experiment, monte_carlo_spatial, monte_carlo_time
import random
import numpy as np
import settings
import pickle

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str, default='test', choices=['test', 'LU_scenarios', 'speculator_sweep', 'eco_burden_sweep', 'lvt_rate_sweep'])
parser.add_argument('--run_id', type=int, default=1)
parser.add_argument('--LU_scenario', type=str, default='baseline', choices=['baseline', 'sprawl', 'density'])
parser.add_argument('--tax_scheme', type=str, default='SQ', choices=['SQ', 'LVT', 'ELVT'])
parser.add_argument('--speculator_nr', type=int, default=1)
parser.add_argument('--eco_burden_denom', type=float, default=100)
parser.add_argument('--lvt_rate', type=float, default=0.05)

args = parser.parse_args()

if __name__=='__main__':
    # set the random seeds
    random.seed(args.run_id)
    np.random.seed(args.run_id)
    if args.experiment_name == 'test':
        # run
        world = run_sim() 
    elif args.experiment_name == 'LU_scenarios':
        settings.set_parameters_for(args.LU_scenario)
        if args.tax_scheme == 'SQ':
            settings.TAX_SCHEME = 'SQ'
            settings.IVT_RATE = 0.035
            settings.LVT_RATE = 0.035
        elif args.tax_scheme == 'LVT':
            settings.TAX_SCHEME = 'LVT'
            if args.LU_scenario == 'baseline':
                settings.LVT_RATE = 0.1
            elif args.LU_scenario == 'sprawl':
                settings.LVT_RATE = 0.135
            elif args.LU_scenario == 'density':
                settings.LVT_RATE = 0.1
        elif args.tax_scheme == 'ELVT':
            settings.TAX_SCHEME = 'ELVT'
            if args.LU_scenario == 'baseline':
                settings.LVT_RATE = 0.1
            elif args.LU_scenario == 'sprawl':
                settings.LVT_RATE = 0.135
            elif args.LU_scenario == 'density':
                settings.LVT_RATE = 0.1
        # run
        history, _ = run_sim()
        # save
        with open(f'outputs/history_{args.LU_scenario}_{args.tax_scheme}_{args.run_id}.pkl', 'wb') as f:
            pickle.dump(history, f)
    elif args.experiment_name == 'speculator_sweep':
        settings.set_parameters_for('baseline')
        if args.tax_scheme == 'SQ':
            settings.TAX_SCHEME = 'SQ'
        elif args.tax_scheme == 'LVT':
            settings.TAX_SCHEME = 'LVT'
            #settings.LVT_RATE = 0.15
        elif args.tax_scheme == 'ELVT':
            settings.TAX_SCHEME = 'ELVT'
            #settings.LVT_RATE = 0.15
        settings.AG_TYPE_WT[2] = np.linspace(1, 40, 5)[args.speculator_nr]
        settings.N_INIT_AGENTS = 300 + 30*args.speculator_nr
        settings.N_NEW_AGENTS = 30 + 3*args.speculator_nr
        # run
        history, _ = run_sim()
        # save
        with open(f'outputs/history_speculator_sweep_{args.tax_scheme}_{args.speculator_nr}_{args.run_id}.pkl', 'wb') as f:
            pickle.dump(history, f)
    elif args.experiment_name == 'eco_burden_sweep':
        assert args.tax_scheme in ['LVT', 'ELVT'], 'eco_burden_sweep only works for LVT and ELVT'
        settings.set_parameters_for('baseline')
        if args.tax_scheme == 'LVT':
            settings.TAX_SCHEME = 'LVT'
        elif args.tax_scheme == 'ELVT':
            settings.TAX_SCHEME = 'ELVT'
            settings.ECO_BURDEN_DENOM = args.eco_burden_denom
        # run
        history, _ = run_sim()
        # save
        with open(f'outputs/history_eco_burden_sweep_{args.tax_scheme}_{args.eco_burden_denom}_{args.run_id}.pkl', 'wb') as f:
            pickle.dump(history, f)
    elif args.experiment_name == 'lvt_rate_sweep':
        assert args.tax_scheme in ['LVT', 'ELVT'], 'lvt_rate_sweep only works for LVT and ELVT'
        settings.set_parameters_for('baseline')
        settings.TAX_SCHEME = args.tax_scheme
        settings.LVT_RATE = args.lvt_rate
        # run
        history, _ = run_sim()
        # save
        with open(f'outputs/history_lvt_rate_sweep_{args.tax_scheme}_{args.lvt_rate}_{args.run_id}.pkl', 'wb') as f:
            pickle.dump(history, f)








