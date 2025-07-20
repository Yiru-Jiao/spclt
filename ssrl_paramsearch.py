'''
For each dataset, search strategy:

Fix `repr_dims`=320 and `lr`=0.001, the training score is the contrastive learning loss (without regularization)

- TS2Vec (no soft labels, no regularizer):
  Phase 1: with other parameters default, search for best `batch_size`

- TopoTS2Vec (no soft labels, topology regularizer):
  Phase 0: set default `batch_size` to the best tuned value from TS2Vec
  Phase 1: with other parameters default, search for best `weight_lr`

- GGeoTS2Vec (no soft labels, geometry regularizer):
  Phase 0: set default `batch_size` to the best tuned value from TS2Vec
  Phase 1: with other parameters default, search for best `bandwidth` and `weight_lr`

- SoftCLT (use soft labels, no regularizer):
  Phase 1: with other parameters default, search for best `tau_temp` and `temporal_hierarchy`
  Phase 2: with best `tau_temp` and `temporal_hierarchy`, search for best `tau_inst` and `batch_size`

- TopoSoftCLT (use soft labels, topology regularizer):
  Phase 0: set default `tau_inst`, `tau_temp`, `temporal_hierarchy`, `batch_size` to the best tuned values from SoftCLT
  Phase 1: with other parameters default, search for best `weight_lr`

- GGeoSoftCLT (use soft labels, geometry regularizer):
  Phase 0: set default `tau_inst`, `tau_temp`, `temporal_hierarchy`, `batch_size` to the best tuned values from SoftCLT
  Phase 1: with other parameters default, search for best `bandwidth` and `weight_lr`

-------------------------------------------------------------------------------------------------------
|            |  TS2Vec  | TopoTS2Vec | GGeoTS2Vec |  SoftCLT | TopoSoftCLT | GGeoSoftCLT |  in total  |
|    runs    |at most 4 |     2      |     2x5    |  5x3+4x5 |      2      |    2x5      |    <=63    |
-------------------------------------------------------------------------------------------------------
'''

import os
import sys
import time as systime
import numpy as np
import pandas as pd
import argparse
from model_utils.utils_paramsearch import *
import model_utils.utils_data as datautils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data. This can be set to UCR, UEA, INT')
    parser.add_argument('--gpu', type=str, default='0', help='The gpu number to use for training and inference (defaults to 0 for CPU only, can be "1,2" for multi-gpu)')
    parser.add_argument('--n_fold', type=int, default=0, help='The number of folds for cross-validation (defaults to 0 for no cross-validation)')
    parser.add_argument('--n_jobs', type=int, default=-1, help='The number of parallel jobs to run for grid search (defaults to -1 for all available cores)')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--reproduction', type=int, default=1, help='Whether this run is for reproduction, if set to True, the random seed would be fixed (defaults to True)')
    args = parser.parse_args()
    args.reproduction = bool(args.reproduction)
    return args


def main(args):
    initial_time = systime.time()
    print('Available cpus:', torch.get_num_threads(), 'available gpus:', torch.cuda.device_count())
    
    # Set the random seed
    if args.reproduction:
        args.seed = 131 # Fix the random seed for reproduction
    if args.seed is None:
        args.seed = random.randint(0, 1000)
    print(f"Random seed is set to {args.seed}")
    fix_seed(args.seed, deterministic=args.reproduction)

    # Initialize the deep learning program, `init_dl_program` is defined in `utils_general.py`
    print(f'--- Cuda available: {torch.cuda.is_available()} ---')
    if torch.cuda.is_available(): 
        print(f'--- Cuda device count: {torch.cuda.device_count()}, Cuda device name: {torch.cuda.get_device_name()}, Cuda version: {torch.version.cuda}, Cudnn version: {torch.backends.cudnn.version()} ---')
    device = init_dl_program(args.gpu)
    print(f'--- Device: {device}, Pytorch version: {torch.__version__}, Available threads: {os.cpu_count()} ---')
    
    # Set result-saving directory
    save_dir = f'results/hyper_parameters/{args.loader}/'
    os.makedirs(save_dir, exist_ok=True)

    def use_best_params(best_param_log, phase):
        best_params = best_param_log[phase]
        if 'score' in best_params:
            best_params.pop('score')
        for key, value in best_params.items():
            if key == 'batch_size':
                best_params[key] = [int(value)]
            elif key == 'temporal_hierarchy':
                if value == 'linear' or value == 'exponential':
                    best_params[key] = [value]
                else:
                    best_params[key] = [None]
            else:
                best_params[key] = [value]
        return best_params

    def search_best_params(parameters2search, params, search_space, grid_search_args, args=args):
        for parameter in parameters2search:
            params = {**params, parameter: search_space[parameter]}
        if args.reproduction: # Reset the random seed for each run
            fix_seed(args.seed, deterministic=args.reproduction)
        best_params, best_score = grid_search(params, **grid_search_args)
        for parameter in parameters2search:
            params = {**params, parameter: [best_params[parameter]]}
        return params, best_score
    
    def save_best_params(best_param_log, log_dir):
        log2save = best_param_log.copy()
        for tuned_phase, tuned_params in log2save.items():
            log2save[tuned_phase] = {key:value[0] if isinstance(value, list) else value for key,value in tuned_params.items()}
        log2save = pd.DataFrame(log2save).T # index: phase, columns: hyperparameters and score
        log2save.to_csv(log_dir)

    # Read the dataset list
    if args.loader == 'UEA':
        dataset_dir = os.path.join('datasets/', args.loader)
        dataset_list = [entry.name for entry in os.scandir(dataset_dir) if entry.is_dir()]
        dataset_list.sort()
    elif 'Macro' in args.loader:
        dataset_list = [['2019']]
    elif args.loader == 'MicroTraffic':
        dataset_list = [['train1']]
    else:
        raise ValueError(f"Unknown dataset loader: {args.loader}")

    # Search for each dataset
    for dataset in dataset_list:
        start_time = systime.time()

        # Load dataset
        if args.loader == 'UEA':
            loaded_data = datautils.load_UEA(dataset)
            train_data, train_labels, _, _ = loaded_data
        elif 'Macro' in args.loader:
            loaded_data = datautils.load_MacroTraffic(dataset, time_interval=5, horizon=15, observation=20)
            train_data, _, _ = loaded_data
            dataset = '2019'
        elif args.loader == 'MicroTraffic':
            loaded_data = datautils.load_MicroTraffic(dataset)
            train_data, _, _ = loaded_data
            dataset = 'train'+''.join(dataset).replace('train', '')
        print(f"------Loaded dataset: {args.loader}-{dataset}------")

        # Load precomputed similarity matrix
        dist_metric = 'EUC'
        sim_mat = datautils.get_sim_mat(args.loader, train_data, dataset, dist_metric)
        if sim_mat is None:
            sim_mat = np.nan * np.ones((train_data.shape[0], 1))
            indexed_sim_mat = np.hstack((np.arange(train_data.shape[0]).reshape(train_data.shape[0], 1), sim_mat))
            print('Nan similarity matrix:', sim_mat.shape, ' train data shape:', train_data.shape)
        else:
            # Stack index and sim_mat for easier pass to grid_search
            indexed_sim_mat = np.hstack((np.arange(train_data.shape[0]).reshape(train_data.shape[0], 1), sim_mat))
            print('Similarity matrix shape:', sim_mat.shape, ' train data shape:', train_data.shape)

        # Predefine default params and search spacef
        default_params = {'tau_inst': [0],
                          'tau_temp': [0],
                          'temporal_hierarchy': [None],
                          'bandwidth': [1.],
                          'batch_size': [8],
                          'weight_lr': [0.05]}

        # Define the search space
        if args.n_fold < 1:
            max_batch_size = int(0.7*train_data.shape[0]).bit_length()+1
        elif args.n_fold > 1:
            max_batch_size = int((args.n_fold-1)/args.n_fold*train_data.shape[0]).bit_length()+1
        else:
            raise ValueError('n_fold must be either 0 or larger than 1')
        search_space = {'tau_inst': [1, 3, 5, 10, 20], # used in softclt study
                        'tau_temp': [0.5, 1., 1.5, 2., 2.5], # used in softclt study
                        'temporal_hierarchy': [None, 'linear', 'exponential'],
                        'bandwidth': [0.25, 1., 9., 25., 49.], # used in geometry regularizer only
                        'batch_size': [2**i for i in range(3, max(4, min(6, max_batch_size)))], # 8, 16, 32
                        'weight_lr': [0.01, 0.05]}
        print(f"--- batch_size search space: {search_space['batch_size']} ---")

        # Initialize the best_param_log
        log_dir = os.path.join(save_dir, f'{dataset}_tuned_hyperparameters.csv')
        if os.path.exists(log_dir):
            best_param_log = pd.read_csv(log_dir, index_col=0)
            best_param_log = best_param_log.to_dict(orient='index')
        else:
            other_log_dir = f'results/hyper_parameters/{args.loader}/{dataset}_tuned_hyperparameters.csv'
            if os.path.exists(other_log_dir):
                best_param_log = pd.read_csv(other_log_dir, index_col=0)
                best_param_log = best_param_log.loc[['TS2Vec_Phase1','SoftCLT_Phase1','SoftCLT_Phase2']]
                best_param_log = best_param_log.to_dict(orient='index')
            else:
                best_param_log = {}

        # Define the grid search arguments
        grid_search_args = {'loader': args.loader,
                            'dataset': dataset + '_size_' + str(train_data.shape[0]),
                            'dist_metric': dist_metric,
                            'train_data': train_data, 
                            'indexed_sim_mat': indexed_sim_mat,
                            'n_fold': args.n_fold if dataset not in ['EigenWorms','MotorImagery'] else 0,
                            'n_jobs': args.n_jobs if dataset not in ['EigenWorms','MotorImagery'] else 1,
                            'fit_config': {'device': device, 'regularizer': None, 'baseline': False}}

        # Initialize the dict of parameters
        params = default_params.copy()

        # TS2Vec (tau_inst=0, tau_temp=0, no regularizer)
        if 'TS2Vec_Phase1' in best_param_log:
            params = use_best_params(best_param_log, 'TS2Vec_Phase1')
            print(f'--- TS2Vec_Phase1 search already completed ---')
        else:
            params, best_score = search_best_params(['batch_size'], params, search_space, grid_search_args)
            best_param_log['TS2Vec_Phase1'] = params
            save_best_params(best_param_log, log_dir)
            print('--- TS2Vec_Phase1 | time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - start_time)) + f' | best score: {best_score} ---')

        # TopoTS2Vec (tau_inst=0, tau_temp=0, topology regularizer)
        grid_search_args['fit_config'] = {'device': device, 'regularizer': 'topology', 'baseline': False}

        if 'TopoTS2Vec_Phase1' in best_param_log:
            params = use_best_params(best_param_log, 'TopoTS2Vec_Phase1')
            print(f'--- TopoTS2Vec_Phase1 search already completed ---')
        else:
            params, best_score = search_best_params(['weight_lr'], params, search_space, grid_search_args)
            best_param_log['TopoTS2Vec_Phase1'] = params
            save_best_params(best_param_log, log_dir)
            print('--- TopoTS2Vec_Phase1 | time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - start_time)) + f' | best score: {best_score} ---')

        # GGeoTS2Vec (tau_inst=0, tau_temp=0, geometry regularizer)
        grid_search_args['fit_config'] = {'device': device, 'regularizer': 'geometry', 'baseline': False}

        if 'GGeoTS2Vec_Phase1' in best_param_log:
            params = use_best_params(best_param_log, 'GGeoTS2Vec_Phase1')
            print(f'--- GGeoTS2Vec_Phase1 hyperparameter search already completed ---')
        else:
            params, best_score = search_best_params(['bandwidth', 'weight_lr'], params, search_space, grid_search_args)
            best_param_log['GGeoTS2Vec_Phase1'] = params
            save_best_params(best_param_log, log_dir)
            print('--- GGeoTS2Vec_Phase1 | time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - start_time)) + f' | best score: {best_score} ---')

        # SoftCLT (use soft labels, no regularizer)
        params = default_params.copy()
        grid_search_args['fit_config'] = {'device': device, 'regularizer': None, 'baseline': False}

        if 'SoftCLT_Phase1' in best_param_log:
            params = use_best_params(best_param_log, 'SoftCLT_Phase1')
            print(f'--- SoftCLT_Phase1 search already completed ---')
        else:
            params, best_score = search_best_params(['tau_temp', 'temporal_hierarchy'], params, search_space, grid_search_args)
            best_param_log['SoftCLT_Phase1'] = params
            save_best_params(best_param_log, log_dir)
            print('--- SoftCLT_Phase1 | time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - start_time)) + f' | best score: {best_score} ---')

        if 'SoftCLT_Phase2' in best_param_log:
            params = use_best_params(best_param_log, 'SoftCLT_Phase2')
            print(f'--- SoftCLT_Phase2 search already completed ---')
        else:
            params, best_score = search_best_params(['tau_inst', 'batch_size'], params, search_space, grid_search_args)
            best_param_log['SoftCLT_Phase2'] = params
            save_best_params(best_param_log, log_dir)
            print('--- SoftCLT_Phase2 | time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - start_time)) + f' | best score: {best_score} ---')

        # TopoSoftCLT (use soft labels, topology regularizer)
        grid_search_args['fit_config'] = {'device': device, 'regularizer': 'topology', 'baseline': False}

        if 'TopoSoftCLT_Phase1' in best_param_log:
            params = use_best_params(best_param_log, 'TopoSoftCLT_Phase1')
            print(f'--- TopoSoftCLT_Phase1 search already completed ---')
        else:
            params, best_score = search_best_params(['weight_lr'], params, search_space, grid_search_args)
            best_param_log['TopoSoftCLT_Phase1'] = params
            save_best_params(best_param_log, log_dir)
            print('--- TopoSoftCLT_Phase1 | time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - start_time)) + f' | best score: {best_score} ---')

        # GGeoSoftCLT (use soft labels, geometry regularizer)
        grid_search_args['fit_config'] = {'device': device, 'regularizer': 'geometry', 'baseline': False}

        if 'GGeoSoftCLT_Phase1' in best_param_log:
            params = use_best_params(best_param_log, 'GGeoSoftCLT_Phase1')
            print(f'--- GGeoSoftCLT_Phase1 hyperparameter search already completed ---')
        else:
            params, best_score = search_best_params(['bandwidth', 'weight_lr'], params, search_space, grid_search_args)
            best_param_log['GGeoSoftCLT_Phase1'] = params
            save_best_params(best_param_log, log_dir)
            print('--- GGeoSoftCLT_Phase1 | time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - start_time)) + f' | best score: {best_score} ---')

        print(f'--- {dataset} hyperparameter search completed, time elapsed : ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time()-start_time)) + ' ---')
        
    print('--- Time elapsed in total : ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time()-initial_time)) + ' ---')
    sys.exit(0)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    main(args)
    