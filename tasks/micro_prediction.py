'''
This script is used to train and evaluate MicroTraffic prediction.
The model is reused from https://github.com/RomainLITUD/UQnet-arxiv
'''

import os
import sys
import time as systime
import random
import torch
import argparse
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tasks.task_utils.utils_pretrain as utils_pre
from tasks.micro_modules.interaction_model import UQnet
from tasks.micro_modules.utils_micro import *
from tasks.micro_modules.training import *
from tasks.micro_modules.interaction_dataset import *
from tasks.micro_modules.losses import *
from model_utils.utils_eval import *
from model_utils.utils_data import load_MicroTraffic


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='The gpu number to use for training and inference (defaults to 0 for CPU only, can be "1,2" for multi-gpu)')
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

    # Create the directory to save the evaluation results
    results_dir = './results/evaluation/MicroTraffic_continued_evaluation.csv'
    save_dir = './results/finetune/MicroTraffic_continued/'
    # Make sure the directories exist
    os.makedirs(save_dir, exist_ok=True)
    print(os.path.exists(os.path.dirname(results_dir)))

    # Define hyper parameters
    paralist = utils_pre.config_micro()
    paralist['encoder_attention_size'] = 128
    paralist['use_sem'] = False
    paralist['epochs'] = args.epochs
    paralist['mode'] = 'lanescore'
    paralist['prob_mode'] = 'ce'
    paralist['batch_size'] = 8
    train_set = ['train1']
    dataset = 'train1'

    # Initialize evaluation results
    model_list = ['original', 'ts2vec', 'topo-ts2vec', 'ggeo-ts2vec', 'softclt', 'topo-softclt', 'ggeo-softclt']
    pred_metrics = ['min_fde', 'mr_05', 'mr_1', 'mr_2']
    knn_metrics = ['mean_shared_neighbours', 'mean_dist_mrre', 'mean_trustworthiness', 'mean_continuity'] # kNN-based, averaged over various k

    def read_saved_results():
        eval_results = pd.read_csv(results_dir)
        eval_results['dataset'] = eval_results['dataset'].astype(str)
        eval_results = eval_results.set_index(['model', 'dataset'])
        return eval_results
    
    if os.path.exists(results_dir):
        eval_results = read_saved_results()
    else:
        metrics = pred_metrics + ['local_'+metric for metric in knn_metrics] + ['global_'+metric for metric in knn_metrics]
        metrics = metrics + ['finetuning_time', 'finetuning_epochs', 'finetuning_time_per_epoch']
        eval_results = pd.DataFrame(np.zeros((len(model_list), 12), dtype=np.float32), columns=metrics,
                                    index=pd.MultiIndex.from_product([model_list,train_set], names=['model','dataset']))
        eval_results.to_csv(results_dir)

    # Load the dataset
    trainset = InteractionDataset(train_set, 'train', paralist, paralist['mode'], device)
    validationset = InteractionDataset(['val'], 'val', paralist, paralist['mode'], device)
    validation_loader = DataLoader(validationset, batch_size=paralist['batch_size'], shuffle=False)
    testset = InteractionDataset(['test'], 'test', paralist, paralist['mode'], device)

    for model_type in model_list:
        # Define model
        paralist['resolution'] = 1.
        paralist['inference'] = False
        if args.reproduction: # Reset the random seed for each run
            utils_pre.fix_seed(args.seed, deterministic=args.reproduction)

        if model_type == 'original':
            model = UQnet(paralist, test=False, drivable=False).to(device)
        else:
            model_dir = f'./results/pretrain/MicroTraffic/{model_type}/train1'
            sp_encoder = utils_pre.define_encoder('MicroTraffic', device, model_dir=model_dir)
            model = UQnet(paralist, test=False, drivable=False, traj_encoder=sp_encoder).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler_heatmap = StepLR(optimizer, step_size=1, gamma=0.975)
        scheduler_epoch = ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=4, cooldown=0,
                                            threshold=1e-3, threshold_mode='rel', min_lr=0.001*0.6**50)

        # Train model if not already trained or if finetuning time is not recorded
        model_trained = os.path.exists(os.path.join(save_dir, f'decoder_{model_type}.pth'))
        if model_trained and (eval_results.loc[(model_type, dataset), 'finetuning_time'] > 0):
            print(f'--- {model_type} has been trained ---')
        else:
            print(f'--- Training {model_type} ---')
            start_time = systime.time()
            finetuning_epochs = train_model(paralist['epochs'], paralist['batch_size'], trainset, model, 
                        optimizer, validation_loader, OverAllLoss(paralist).to(device),
                        scheduler_heatmap, scheduler_epoch, mode=paralist['mode'])
            finetuning_time = systime.time() - start_time
            torch.save(model.encoder.state_dict(), os.path.join(save_dir, f'encoder_{model_type}.pth'))
            torch.save(model.decoder.state_dict(), os.path.join(save_dir, f'decoder_{model_type}.pth'))

            eval_results = read_saved_results() # read saved results again to avoid overwriting
            eval_results.loc[(model_type, dataset), 'finetuning_time'] = finetuning_time
            eval_results.loc[(model_type, dataset), 'finetuning_epochs'] = finetuning_epochs
            eval_results.loc[(model_type, dataset), 'finetuning_time_per_epoch'] = finetuning_time / finetuning_epochs
            eval_results.to_csv(results_dir)
            print(f'Finetuning time for {model_type}: ' + systime.strftime('%H:%M:%S', systime.gmtime(finetuning_time)))

        # Evaluate models
        if eval_results.loc[(model_type, dataset), 'global_mean_continuity'] > 0:
            print(f'--- {model_type} {dataset} has been evaluated, skipping evaluation ---')
            continue

        paralist['resolution'] = 0.5
        paralist['inference'] = True
        if model_type == 'original':
            model = UQnet(paralist, test=True, drivable=False) # set test=True here
        else:
            sp_encoder = utils_pre.define_encoder('MicroTraffic', device, model_dir=model_dir)
            model = UQnet(paralist, test=True, drivable=False, traj_encoder=sp_encoder)

        model.encoder.load_state_dict(torch.load(os.path.join(save_dir, f'encoder_{model_type}.pth'), 
                                                    map_location=device, weights_only=True))
        model.decoder.load_state_dict(torch.load(os.path.join(save_dir, f'decoder_{model_type}.pth'), 
                                                    map_location=device, weights_only=True))
        model = model.to(device)
        model.eval()
        Yp, Ua, Ue, Y = inference_model([model], testset, paralist)

        # Prediction evaluation
        metric_mean, metric_std = ComputeError(Yp, Y, r_list=[0.5,1.,2.], sh=6) # r is the radius of error in meters
        pred_results = {'min_fde_mean': metric_mean[0], 'min_fde_std': metric_std[0],
                        'mr_05_mean': metric_mean[1], 'mr_05_std': metric_std[1],
                        'mr_1_mean': metric_mean[2], 'mr_1_std': metric_std[2],
                        'mr_2_mean': metric_mean[3], 'mr_2_std': metric_std[3]}

        # Encoding evaluation
        _, _, test_data = load_MicroTraffic(train_set, dataset_dir='./datasets')
        test_labels = np.zeros(test_data.shape[0])
        eval_args = {'loader': 'MicroTraffic',
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
    