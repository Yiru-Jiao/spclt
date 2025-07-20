'''
This script is used to evaluate models.
The evaluation results are saved in the evaluation directory.
Random seed is fixed.
'''

import os
import sys
import time as systime
import glob
import numpy as np
import pandas as pd
import argparse
import torch
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import spclt
from model_utils.utils_general import *
import model_utils.utils_data as datautils
from model_utils.utils_eval import *
from tasks.task_utils.svm_eval import eval_classification


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='The gpu number to use for training and inference (defaults to 0 for CPU only, can be "1,2" for multi-gpu)')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--reproduction', type=int, default=1, help='Whether this run is for reproduction, if set to True, the random seed would be fixed (defaults to True)')
    args = parser.parse_args()
    args.reproduction = bool(args.reproduction)

    # Set default parameters
    args.loader = 'UEA'
    args.dist_metric = 'EUC'
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
    results_dir = f'results/evaluation/{args.loader}_evaluation.csv'
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(f'results/evaluation/', exist_ok=True)

    # Read the dataset list
    dataset_dir = os.path.join('datasets/', args.loader)
    dataset_list = [entry.name for entry in os.scandir(dataset_dir) if entry.is_dir()]
    dataset_list.sort()

    # Initialize evaluation dataframe for UEA classification
    model_list = ['ts2vec', 'topo-ts2vec', 'topo-ts2vec-baseline', 'ggeo-ts2vec', 'ggeo-ts2vec-baseline', 
                  'softclt', 'topo-softclt', 'topo-softclt-baseline', 'ggeo-softclt', 'ggeo-softclt-baseline']
    clf_clr_metrics = ['svm_acc', 'svm_auprc'] # Classification results
    knn_metrics = ['mean_shared_neighbours', 'mean_dist_mrre', 'mean_trustworthiness', 'mean_continuity'] # kNN-based, averaged over various k

    def read_saved_results():
        eval_results = pd.read_csv(results_dir)
        eval_results['dataset'] = eval_results['dataset'].astype(str)
        eval_results = eval_results.set_index(['model', 'dataset'])
        return eval_results

    if os.path.exists(results_dir):
        eval_results = read_saved_results()
    else:
        metrics = clf_clr_metrics + ['local_'+metric for metric in knn_metrics] + ['global_'+metric for metric in knn_metrics]
        eval_results = pd.DataFrame(np.zeros((len(dataset_list)*len(model_list), 10), dtype=np.float32), columns=metrics,
                                    index=pd.MultiIndex.from_product([model_list,dataset_list], names=['model','dataset']))
        eval_results.to_csv(results_dir)

    # Evaluate for each dataset
    for dataset in dataset_list:
        # Load dataset
        loaded_data = datautils.load_UEA(dataset)
        train_data, train_labels, test_data, test_labels = loaded_data
        
        # Load tuned hyperparameters
        tuned_params_dir = f'results/hyper_parameters/{args.loader}/{dataset}_tuned_hyperparameters.csv'
        if os.path.exists(tuned_params_dir):
            tuned_params = pd.read_csv(tuned_params_dir, index_col=0)
        else:
            print(f'****** {tuned_params_dir} not found ******')
            continue

        feature_size = test_data.shape[-1]
        # Iterate over different models
        for model_type in model_list:
            # Skip if the model has been evaluated
            if eval_results.loc[(model_type, dataset), 'svm_acc'] > 0:
                print(f'--- {model_type} {dataset} has been evaluated, skipping ---')
                continue

            model_dir = os.path.join(run_dir, f'{model_type}/{dataset}')
            os.makedirs(model_dir, exist_ok=True)

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

            # Load the best model for evaluation
            if os.path.exists(f'{model_dir}/loss_log.csv'):
                print(f'--- {model_type} {dataset} has been trained, loading final model ---')
                existing_models = glob.glob(f'{model_dir}/*_net.pth')
                best_model = 'model' + existing_models[0].split('model')[-1].split('_net')[0]
            else:
                print(f'--- {model_type} {dataset} was not trained, skipping ---')
                continue
            if args.reproduction: # Reset the random seed for each run
                fix_seed(args.seed, deterministic=args.reproduction)
            model = spclt(args.loader, **model_config)
            model.load(f'{model_dir}/{best_model}')

            # Evaluate the model
            print(f'Evaluating with {best_model} ...')

            ## classification results
            _, acc = eval_classification(model, train_data, train_labels, test_data, test_labels)
            clf_clr_results = {'svm_acc': acc['acc'], 'svm_auprc': acc['auprc']}
            
            ## knn results
            eval_args = {'loader': args.loader, 
                         'dataset': dataset,
                         'data': test_data,
                         'labels': test_labels,
                         'model': model,
                         'batch_size': 128,
                         'save_dir': model_dir}
            local_dist_results = evaluate(local=True, save_latents=False, **eval_args)
            global_dist_results = evaluate(local=False, save_latents=True, **eval_args)

            ## loss results
            test_data, test_labels = datautils.modify_train_data(test_data, test_labels)
            test_sim_mat = datautils.get_sim_mat(args.loader, test_data, dataset, args.dist_metric, prefix='test')
            test_soft_assignments = datautils.assign_soft_labels(test_sim_mat, args.tau_inst)
            loss_results = model.compute_loss(test_data, test_soft_assignments, non_regularized=False)
            loss_results = {'scl_loss': loss_results[1],
                            'sp_loss': loss_results[3] if args.regularizer is not None else np.nan}

            # Save evaluation results
            key_values = {**clf_clr_results, **loss_results, **local_dist_results, **global_dist_results}
            keys = list(key_values.keys())
            values = np.array(list(key_values.values())).astype(np.float32)
            eval_results = read_saved_results() # read saved results again to avoid overwriting
            eval_results.loc[(model_type, dataset), keys] = values
            eval_results.loc[(model_type, dataset), 'inference_time'] = acc['inference_time']
            eval_results.loc[(model_type, dataset), 'num_samples'] = acc['num_samples']

            # Save evaluation results per dataset and model
            eval_results.to_csv(results_dir)

    print('--- Total time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - initial_time)) + ' ---')
    sys.exit(0)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    main(args)



