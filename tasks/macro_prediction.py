'''
This script is used to train and evaluate MacroTraffic prediction.
The model is reused from https://github.com/RomainLITUD/uncertainty-aware-traffic-speed-flow-demand-prediction
Random seed is fixed.
'''

import os
import sys
import time as systime
import random
import torch
import argparse
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tasks.task_utils.utils_pretrain as utils_pre
from tasks.macro_modules.models import DGCN, MSE_scale, LSTM, GRU
from tasks.macro_modules.utils_macro import *
from tasks.macro_modules.training import *
from tasks.macro_modules.custom_dataset import *
from model_utils.utils_eval import *
from model_utils.utils_data import load_MacroTraffic


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='The gpu number to use for training and inference (defaults to 0 for CPU only, can be "1,2" for multi-gpu)')
    parser.add_argument('--prediction_model', type=str, default='DGCN', help='The prediction model to use (defaults to DGCN)')
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--reproduction', type=int, default=1, help='Whether this run is for reproduction, if set to True, the random seed would be fixed (defaults to False)')
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
    utils_pre.fix_seed(args.seed, deterministic=args.reproduction)
    
    # Initialize the deep learning program
    print(f'--- Cuda available: {torch.cuda.is_available()} ---')
    if torch.cuda.is_available(): 
        print(f'--- Cuda device count: {torch.cuda.device_count()}, Cuda device name: {torch.cuda.get_device_name()}, Cuda version: {torch.version.cuda}, Cudnn version: {torch.backends.cudnn.version()} ---')
    device = utils_pre.init_dl_program(args.gpu)
    print(f'--- Device: {device}, Pytorch version: {torch.__version__} ---')

    # Create the directory to save evaluation results
    if args.prediction_model == 'DGCN':
        continued_results_dir = './results/evaluation/MacroTraffic_continued_evaluation.csv'
        continued_save_dir = './results/finetune/MacroTraffic_continued/'
        fixed_results_dir = './results/evaluation/MacroTraffic_fixed_evaluation.csv'
        fixed_save_dir = './results/finetune/MacroTraffic_fixed/'
    else:
        continued_results_dir = f'./results/evaluation/Macro{args.prediction_model}_continued_evaluation.csv'
        continued_save_dir = f'./results/finetune/Macro{args.prediction_model}_continued/'
        fixed_results_dir = f'./results/evaluation/Macro{args.prediction_model}_fixed_evaluation.csv'
        fixed_save_dir = f'./results/finetune/Macro{args.prediction_model}_fixed/'
    # Make sure the directories exist
    for save_dir in [continued_save_dir, fixed_save_dir]:
        os.makedirs(save_dir, exist_ok=True)
    print(os.path.exists(os.path.dirname(continued_results_dir)), os.path.exists(os.path.dirname(fixed_results_dir)))

    # Define hyper parameters
    para = {}
    para['time_invertal'] = 5
    para['horizon'] = 15 # predict next 30 min
    para['observation'] = 20 # observe last 40 min
    para['nb_node'] = 193
    para['dim_feature'] = 128
    A = adjacency_matrix(3)
    B = adjacency_matrixq(3, 8)
    years = ['2019']
    dataset = '2019'
    def define_model(prediction_model, sp_encoder=None):
        if prediction_model == 'DGCN':
            model = DGCN(para, A, B, uncertainty=False, encoder=sp_encoder).to(device)
            BATCH_SIZE = 16
        elif prediction_model == 'LSTM':
            model = LSTM(para, num_layers=2, encoder=sp_encoder).to(device)
            BATCH_SIZE = 512
        elif prediction_model == 'GRU':
            model = GRU(para, num_layers=2, encoder=sp_encoder).to(device)
            BATCH_SIZE = 512
        return model, BATCH_SIZE

    # Initialize evaluation results
    model_list = ['original', 'ts2vec', 'topo-ts2vec', 'ggeo-ts2vec', 'softclt', 'topo-softclt', 'ggeo-softclt']
    pred_metrics = ['mae', 'rmse', 'error_std', 'explained_variance'] # Prediction-based
    knn_metrics = ['mean_shared_neighbours', 'mean_dist_mrre', 'mean_trustworthiness', 'mean_continuity'] # kNN-based, averaged over various k
    
    for continue_training, results_dir, save_dir in zip([False, True], [fixed_results_dir, continued_results_dir], [fixed_save_dir, continued_save_dir]):
        if not continue_training:
            continue
        def read_saved_results():
            eval_results = pd.read_csv(results_dir)
            eval_results['dataset'] = eval_results['dataset'].astype(str)
            eval_results = eval_results.set_index(['model', 'dataset'])
            return eval_results
        
        if os.path.exists(results_dir):
            eval_results = read_saved_results()
        else:
            metrics = pred_metrics + ['local_'+metric for metric in knn_metrics] + ['global_'+metric for metric in knn_metrics]
            eval_results = pd.DataFrame(np.zeros((len(model_list), 12), dtype=np.float32), columns=metrics,
                                        index=pd.MultiIndex.from_product([model_list,years], names=['model','dataset']))
            eval_results.to_csv(results_dir)

        # Load dataset
        trainset = AMSdataset(years, para, 'train', device)
        validationset = AMSdataset(years, para, 'validation', device)
        testset = AMSdataset(years, para, 'test', device)

        for model_type in model_list:
            # Define model
            if model_type == 'original':
                sp_encoder = None
            else:
                if args.prediction_model == 'DGCN':
                    loader = 'MacroTraffic'
                elif args.prediction_model == 'LSTM':
                    loader = 'MacroLSTM'
                elif args.prediction_model == 'GRU':
                    loader = 'MacroGRU'
                model_dir = f'./results/pretrain/{loader}/{model_type}/{dataset}/'
                sp_encoder = utils_pre.define_encoder(loader, device, model_dir, continue_training=continue_training)
            learning_rate = 0.001
            model, BATCH_SIZE = define_model(args.prediction_model, sp_encoder)

            if args.prediction_model == 'DGCN':
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            else:
                optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, cooldown=10,
                                          threshold=1e-3, threshold_mode='rel', min_lr=learning_rate*0.5**15)
            validation_loader = DataLoader(validationset, batch_size=BATCH_SIZE, shuffle=False)

            # Train model if not already trained
            if os.path.exists(os.path.join(save_dir, f'{model_type}.pth')):
                print(f'--- {model_type} has been trained ---')
            else:
                print(f'--- Training {model_type} ---')
                start_time = systime.time()
                train_model(args.epochs, BATCH_SIZE, trainset, model,
                            optimizer, validation_loader, MSE_scale().to(device),
                            scheduler, para['horizon'], beta=-0.07)
                finetuning_time = systime.time() - start_time
                if model_type == 'original':
                    torch.save(model.state_dict(), os.path.join(fixed_save_dir, f'{model_type}.pth'))
                    torch.save(model.state_dict(), os.path.join(continued_save_dir, f'{model_type}.pth'))
                else:
                    torch.save(model.state_dict(), os.path.join(save_dir, f'{model_type}.pth'))
                print(f'Training time for {model_type}: ' + systime.strftime('%H:%M:%S', systime.gmtime(finetuning_time)))

            # Evaluate models
            if eval_results.loc[(model_type, dataset), 'global_mean_continuity'] > 0:
                print(f'--- {model_type} {dataset} has been evaluated, skipping evaluation ---')
                continue

            model.load_state_dict(torch.load(os.path.join(save_dir, f'{model_type}.pth'), map_location=device, weights_only=True))
            model = model.to(device)
            model.eval()
            prediction = test_run_point(testset, model, BATCH_SIZE)

            # Prediction evaluation
            # scale back to original values (km/h)
            prediction = prediction[...,0]*130 # (N, 15, 193, 1)
            X = testset.X.copy()
            X = X[:,-15:,:,0]*130 # (N, 15, 193, 1)
            pred_results = {'mae': np.mean(np.abs(prediction-X)),
                            'rmse': np.mean((prediction-X)**2)**0.5,
                            'error_std': np.std(prediction-X),
                            'explained_variance': 1-np.var(prediction-X)/np.var(X)}

            # Encoding evaluation
            _, _, test_data = load_MacroTraffic(years, para['time_invertal'], para['horizon'], para['observation'],
                                                dataset_dir='./datasets')
            test_labels = np.zeros(test_data.shape[0])
            eval_args = {'loader': 'MacroTraffic',
                         'dataset': dataset,
                         'data': test_data,
                         'labels': test_labels,
                         'model': model,
                         'batch_size': 128}
            local_dist_results = evaluate(local=True, **eval_args)
            global_dist_results = evaluate(local=False, **eval_args)

            key_values = {**pred_results, **local_dist_results, **global_dist_results}
            keys = list(key_values.keys())
            values = np.array(list(key_values.values())).astype(np.float32)
            eval_results = read_saved_results() # read saved results again to avoid overwriting
            eval_results.loc[(model_type, dataset), keys] = values

            # Save evaluation results per dataset and model
            eval_results.to_csv(results_dir)

    print(f"Total time: {systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - initial_time))}")
    sys.exit(0)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    main(args)
