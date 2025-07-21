'''
This script is used to train and evaluate models.
Tuned hyperparameters are loaded from the hyper_parameters directory.
The trained models are saved in the results directory.
The evaluation results are saved in the evaluation directory.
'''

import os
import sys
import time as systime
import glob
import numpy as np
import pandas as pd
import argparse
import torch
from model import spclt
from model_utils.utils_general import *
import model_utils.utils_data as datautils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data. This can be set to UCR, UEA, INT')
    parser.add_argument('--gpu', type=str, default='0', help='The gpu number to use for training and inference (defaults to 0 for CPU only, can be "1,2" for multi-gpu)')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--reproduction', type=int, default=1, help='Whether this run is for reproduction, if set to True, the random seed would be fixed (defaults to True)')
    args = parser.parse_args()
    args.reproduction = bool(args.reproduction)

    # Set default parameters
    args.sliding_padding = 0
    args.repr_dims = 320
    args.tau_inst = 0
    args.tau_temp = 0
    args.temporal_hierarchy = None
    args.regularizer = None
    args.baseline = False
    args.bandwidth = 1.
    args.iters = None
    args.epochs = 100
    args.batch_size = 8
    args.lr = 0.001
    args.weight_lr = 0.01

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

    # Initialize the deep learning program
    print(f'--- Cuda available: {torch.cuda.is_available()} ---')
    if torch.cuda.is_available(): 
        print(f'--- Cuda device count: {torch.cuda.device_count()}, Cuda device name: {torch.cuda.get_device_name()}, Cuda version: {torch.version.cuda}, Cudnn version: {torch.backends.cudnn.version()} ---')
    device = init_dl_program(args.gpu)
    print(f'--- Device: {device}, Pytorch version: {torch.__version__} ---')

    # Create the directory to save the evaluation results
    run_dir = f'results/pretrain/{args.loader}/'
    results_dir = f'results/evaluation/{args.loader}_training_efficiency.csv'
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs('results/evaluation', exist_ok=True)

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

    # Initialize evaluation dataframe for training efficiency
    if args.loader == 'UEA':
        model_list = ['ts2vec', 'topo-ts2vec', 'topo-ts2vec-baseline', 'ggeo-ts2vec', 'ggeo-ts2vec-baseline', 
                      'softclt', 'topo-softclt', 'topo-softclt-baseline', 'ggeo-softclt', 'ggeo-softclt-baseline']
    else:
        model_list = ['ts2vec', 'topo-ts2vec', 'ggeo-ts2vec', 'softclt', 'topo-softclt', 'ggeo-softclt']

    def read_saved_results():
        eval_results = pd.read_csv(results_dir)
        eval_results['dataset'] = eval_results['dataset'].astype(str)
        eval_results = eval_results.set_index(['model', 'dataset'])
        return eval_results
    
    if os.path.exists(results_dir):
        eval_results = read_saved_results()
    else:
        metrics = ['training_time', 'training_epochs', 'training_time_per_epoch']
        eval_results = pd.DataFrame(np.zeros((len(dataset_list)*len(model_list), 3), dtype=np.float32), columns=metrics,
                                    index=pd.MultiIndex.from_product([model_list, dataset_list if args.loader=='UEA' else dataset_list[0]], names=['model','dataset']))
        eval_results.to_csv(results_dir)

    # Train for each dataset
    for dataset in dataset_list:
        # Load dataset
        if args.loader == 'UEA':
            loaded_data = datautils.load_UEA(dataset)
            train_data, _, _, _ = loaded_data
        elif 'Macro' in args.loader:
            loaded_data = datautils.load_MacroTraffic(dataset, time_interval=5, horizon=15, observation=20)
            train_data, _, _ = loaded_data
            dataset = '2019'
        elif args.loader == 'MicroTraffic':
            loaded_data = datautils.load_MicroTraffic(dataset)
            train_data, _, _ = loaded_data
            dataset = 'train'+''.join(dataset).replace('train', '')
        
        # Load tuned hyperparameters
        tuned_params_dir = f'results/hyper_parameters/{args.loader}/{dataset}_tuned_hyperparameters.csv'
        if os.path.exists(tuned_params_dir):
            tuned_params = pd.read_csv(tuned_params_dir, index_col=0)
        else:
            print(f'****** {tuned_params_dir} not found ******')
            continue

        # Compute similarity matrix
        args.dist_metric = 'EUC'
        sim_mat = datautils.get_sim_mat(args.loader, train_data, dataset, args.dist_metric)
        
        # Set training epochs and verbose
        train_size = train_data.shape[0]
        feature_size = train_data.shape[-1]
        if args.loader != 'UEA':
            args.epochs = 250
            verbose = 3
        else:
            if train_size < 1000 and train_data.shape[-2] < 1000:
                args.epochs = 500
                verbose = 10
            elif train_size < 3000:
                args.epochs = 300
                verbose = 5
            else:
                args.epochs = 150
                verbose = 3

        # Iterate over different losses
        for model_type in model_list:
            # Set hyperparameters and configure model
            if 'baseline' in model_type:
                args.baseline = True
                para2load = model_type.split('-base')[0]
            else:
                args.baseline = False
                para2load = model_type
            try:
                args = load_tuned_hyperparameters(args, tuned_params, para2load)
            except:
                print(f'****** {model_type} hyperparameters not found ******')
                continue
            model_config = configure_model(args, feature_size, device)

            model_dir = os.path.join(run_dir, f'{model_type}/{dataset}')
            os.makedirs(model_dir, exist_ok=True)

            # Train model if not already trained or if training time is not recorded
            loss_log_exist =  os.path.exists(f'{model_dir}/loss_log.csv')
            if loss_log_exist and (eval_results.loc[(model_type, dataset), 'training_time'] > 0):
                print(f'--- {model_type} {dataset} has been trained, skip training ---')
            else:
                # Create model
                model_config['after_epoch_callback'] = save_checkpoint_callback(model_dir, 0, unit='epoch')
                if args.reproduction: # Reset the random seed for each run
                    fix_seed(args.seed, deterministic=args.reproduction)
                model = spclt(args.loader, **model_config)

                scheduler = 'reduced'
                print(f'--- {args.loader}_{model_type}_{dataset} training with ReduceLROnPlateau scheduler ---')
                soft_assignments = datautils.assign_soft_labels(sim_mat, args.tau_inst)
                start_time = systime.time()
                loss_log = model.fit(dataset, train_data, soft_assignments, args.epochs, args.iters, scheduler, verbose=verbose)
                training_time = systime.time() - start_time

                eval_results = read_saved_results()
                eval_results.loc[(model_type, dataset), 'training_time'] = training_time
                eval_results.loc[(model_type, dataset), 'training_epochs'] = model.epoch_n
                eval_results.loc[(model_type, dataset), 'training_time_per_epoch'] = training_time/model.epoch_n
                eval_results.to_csv(results_dir)

                # Save loss log
                save_loss_log(loss_log, model_dir, regularizer=args.regularizer)
                print(f'Training time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(training_time)))
            
            # Reserve the latest model and remove the rest
            existing_models = glob.glob(f'{model_dir}/*_net.pth')
            if len(existing_models)>1:
                existing_models.sort(key=os.path.getmtime, reverse=True)
                for model_epoch in existing_models[1:]:
                    os.remove(model_epoch)
                    if not model_type in ['ts2vec', 'softclt']:
                        os.remove(model_epoch.replace('_net.pth', '_loss_log_vars.npy'))
            latest_model = 'model' + existing_models[0].split('model')[-1].split('_net')[0]

            # Save evaluation results per dataset and model
            eval_results = read_saved_results()
            print(f'Best model {latest_model} will be evaluated in downstream tasks on {dataset}')
            eval_results.loc[(model_type, dataset), 'model_used'] = latest_model
            eval_results.to_csv(results_dir)

    print('--- Total time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - initial_time)) + ' ---')
    sys.exit(0)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    main(args)

