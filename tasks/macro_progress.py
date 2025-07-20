'''
This script is used to train and evaluate MacroTraffic prediction.
The model is reused from https://github.com/RomainLITUD/uncertainty-aware-traffic-speed-flow-demand-prediction
'''

import os
import sys
import time as systime
import random
import torch
import argparse
import numpy as np
import pandas as pd
from glob import glob
from torch.optim.lr_scheduler import StepLR
from macro_modules.models import DGCN, MSE_scale, LSTM, GRU
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
    
    # Initialize the deep learning program, `init_dl_program` is defined in `utils_general.py`
    print(f'--- Cuda available: {torch.cuda.is_available()} ---')
    if torch.cuda.is_available(): 
        print(f'--- Cuda device count: {torch.cuda.device_count()}, Cuda device name: {torch.cuda.get_device_name()}, Cuda version: {torch.version.cuda}, Cudnn version: {torch.backends.cudnn.version()} ---')
    device = utils_pre.init_dl_program(args.gpu)
    print(f'--- Device: {device}, Pytorch version: {torch.__version__} ---')

    # Create the directory to save the evaluation results
    if args.prediction_model == 'DGCN':
        results_dir = './results/evaluation/MacroTraffic_progress_evaluation.csv'
        save_dir = './results/finetune/MacroTraffic_progress/'
    else:
        results_dir = f'./results/evaluation/Macro{args.prediction_model}_progress_evaluation.csv'
        save_dir = f'./results/finetune/Macro{args.prediction_model}_progress/'
    # Make sure the directories exist
    os.makedirs(save_dir, exist_ok=True)
    print(os.path.exists(os.path.dirname(results_dir)))

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
    EPOCH_NUMBER = args.epochs

    # Initialize evaluation results
    model_list = ['original', 'ts2vec', 'topo-ts2vec', 'ggeo-ts2vec', 'softclt', 'topo-softclt', 'ggeo-softclt']
    for model_type in model_list:
        os.makedirs(save_dir+model_type, exist_ok=True)

    # Train and evaluate models
    def read_saved_results():
        eval_results = pd.read_csv(results_dir)
        eval_results['epoch'] = eval_results['epoch'].astype(int)
        eval_results = eval_results.set_index(['model', 'epoch'])
        return eval_results

    pred_metrics = ['mae', 'rmse', 'error_std', 'explained_variance'] # Prediction-based
    knn_metrics = ['mean_shared_neighbours', 'mean_dist_mrre', 'mean_trustworthiness', 'mean_continuity'] # kNN-based, averaged over various k
    if os.path.exists(results_dir):
        eval_results = read_saved_results()
    else:
        eval_results = pd.DataFrame(np.zeros((len(model_list)*int(EPOCH_NUMBER/6), 4), dtype=np.float32), columns=pred_metrics,
                                    index=pd.MultiIndex.from_product([model_list, list(range(int(EPOCH_NUMBER/6)))], names=['model', 'epoch']))
        eval_results.to_csv(results_dir)

    for model_type in model_list:
        # Load dataset
        trainset = AMSdataset(years, para, 'train', device)
        validationset = AMSdataset(years, para, 'validation', device)
        testset = AMSdataset(years, para, 'test', device)

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
            sp_encoder = utils_pre.define_encoder(loader, device, model_dir, continue_training=True)
        learning_rate = 0.001
        if args.reproduction: # Reset the random seed for each run
            utils_pre.fix_seed(args.seed, deterministic=args.reproduction)
        model, BATCH_SIZE = define_model(args.prediction_model, sp_encoder)

        if args.prediction_model == 'DGCN':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = StepLR(optimizer, step_size=1, gamma=0.98)
        else:
            optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
            scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
        validation_loader = DataLoader(validationset, batch_size=BATCH_SIZE, shuffle=False)

        # Train model if not already trained
        progress_list = glob(os.path.join(save_dir+model_type, '*.pth'))
        if len(progress_list) >= int(EPOCH_NUMBER/6):
            print(f'--- {model_type} has been trained, skipping training ---')
        else:
            print(f'--- Training {model_type} ---')
            start_time = systime.time()
            finetuning_epochs = train_model(EPOCH_NUMBER, BATCH_SIZE, trainset, model,
                        optimizer, validation_loader, MSE_scale().to(device),
                        scheduler, para['horizon'], beta=-0.07, save_progress=save_dir+model_type)
            finetuning_time = systime.time() - start_time

            eval_results = read_saved_results() # read saved results again to avoid overwriting
            eval_results.loc[(model_type, dataset), 'finetuning_time'] = finetuning_time
            eval_results.loc[(model_type, dataset), 'finetuning_epochs'] = finetuning_epochs
            eval_results.loc[(model_type, dataset), 'finetuning_time_per_epoch'] = finetuning_time / finetuning_epochs
            eval_results.to_csv(results_dir)

            print(f'Training time for {model_type}: ' + systime.strftime('%H:%M:%S', systime.gmtime(finetuning_time)))

        # Evaluate models
        ## Load standardized data for structure similarity evaluation
        _, _, test_data = load_MacroTraffic(years, para['time_invertal'], para['horizon'], para['observation'],
                                            dataset_dir='./datasets')
        progress_list = glob(os.path.join(save_dir+model_type, 'ckpt_*.pth'))
        progress_list = sorted(progress_list, key=lambda x: int(x.split('ckpt_')[1].split('.pth')[0]))
        initial_file_saved_time = os.path.getmtime(progress_list[0])
        epoch_indecies = [int(epoch_path.split('ckpt_')[1].split('.pth')[0]) for epoch_path in progress_list]
        for epoch_path, epoch_index in tqdm(zip(progress_list, epoch_indecies), desc=f'Evaluating {model_type}', ascii=True, dynamic_ncols=False,miniters=10, total=len(progress_list)):
            if eval_results.loc[(model_type, epoch_index), 'mae'] > 0 and args.prediction_model != 'DGCN':
                print(f'--- {model_type}-{epoch_index} has been evaluated, skipping evaluation ---')
                continue
            model.load_state_dict(torch.load(epoch_path, map_location=device, weights_only=True))
            model = model.to(device)
            model.eval()
            prediction = test_run_point(testset, model, BATCH_SIZE)

            ## Prediction evaluation, scale back to original values (km/h)
            prediction = prediction[...,0]*130 # (N, 15, 193, 1)
            X = testset.X[:,-15:,:,0]*130 # (N, 15, 193, 1)
            pred_results = {'mae': np.mean(np.abs(prediction-X)),
                            'rmse': np.mean((prediction-X)**2)**0.5,
                            'error_std': np.std(prediction-X),
                            'explained_variance': 1-np.var(prediction-X)/np.var(X)}

            ## Encoding evaluation
            test_labels = np.zeros(test_data.shape[0])
            eval_args = {'loader': 'MacroTraffic',
                         'dataset': dataset,
                         'data': test_data,
                         'labels': test_labels,
                         'model': model,
                         'batch_size': 128}
            global_eval_exist = 'global_mean_dist_mrre' in eval_results.columns
            if global_eval_exist:
                global_evaluated = eval_results.loc[(model_type, epoch_index), 'global_mean_dist_mrre'] > 0
            else:
                global_evaluated = False
            if global_eval_exist and global_evaluated:
                global_dist_results = {}
                for metric in knn_metrics:
                    global_dist_results[f'global_{metric}'] = eval_results.loc[(model_type, epoch_index), f'global_{metric}']
            else:
                global_dist_results = evaluate(local=False, drmse_only=False, **eval_args)
            if args.prediction_model == 'DGCN':
                local_eval_exist = 'local_mean_dist_mrre' in eval_results.columns
                if local_eval_exist:
                    local_evaluated = eval_results.loc[(model_type, epoch_index), 'local_mean_dist_mrre'] > 0
                else:
                    local_evaluated = False
                if local_eval_exist and local_evaluated:
                    local_dist_results = {}
                    for metric in knn_metrics:
                        local_dist_results[f'local_{metric}'] = eval_results.loc[(model_type, epoch_index), f'local_{metric}']
                else:
                    local_dist_results = evaluate(local=True, drmse_only=False, **eval_args)
                key_values = {**pred_results, **local_dist_results, **global_dist_results}
            else:
                key_values = {**pred_results, **global_dist_results}
            keys = list(key_values.keys())
            values = np.array(list(key_values.values())).astype(np.float32)
            eval_results = read_saved_results() # read saved results again to avoid overwriting
            eval_results.loc[(model_type, epoch_index), keys] = values
            file_saved_time = os.path.getmtime(epoch_path)
            eval_results.loc[(model_type, epoch_index), 'clocktime_passed'] = file_saved_time - initial_file_saved_time

            # Save evaluation results per dataset and model
            eval_results.to_csv(results_dir)

    print(f"Total time: {systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - initial_time))}")
    sys.exit(0)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    main(args)
