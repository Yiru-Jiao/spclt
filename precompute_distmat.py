'''
This script is used to precompute the similarity matrix for each dataset in the dataset list.
The similarity matrix is computed based on the instance-wise distance between each pair of instances in the dataset.
This saves time when training the model.
'''

import os
import sys
import time as systime
import argparse
import model_utils.utils_data as datautils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data. This can be set to UCR, UEA, INT')
    parser.add_argument('--max_train_length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    args = parser.parse_args()
    return args


def main(args):
    start_time = systime.time()
    
    # Read the dataset list
    if args.loader == 'UEA':
        dataset_dir = os.path.join('datasets/', args.loader)
        dataset_list = [entry.name for entry in os.scandir(dataset_dir) if entry.is_dir()]
        dataset_list.sort()
    elif args.loader == 'MacroTraffic':
        dataset_list = [['2019']]
    elif args.loader == 'MicroTraffic':
        dataset_list = [['train1']]
    else:
        raise ValueError(f"Unknown dataset loader: {args.loader}")

    for dataset in dataset_list:
        # Load dataset
        if args.loader == 'UEA':
            loaded_data = datautils.load_UEA(dataset)
            train_data, _, test_data, _ = loaded_data
        elif 'Macro' in args.loader:
            loaded_data = datautils.load_MacroTraffic(dataset, time_interval=5, horizon=15, observation=20)
            train_data, _, test_data = loaded_data
            dataset = '2019'
        elif args.loader == 'MicroTraffic':
            loaded_data = datautils.load_MicroTraffic(dataset)
            train_data, _, test_data = loaded_data
            dataset = 'train'+''.join(dataset).replace('train', '')
        print(f"------ Loaded dataset: {args.loader}-{dataset}, train shape {train_data.shape}, test shape {test_data.shape} ------")

        # Compute similarity matrix (this is instance-wise only)
        dist_metric = 'EUC'
        sim_mat = datautils.get_sim_mat(args.loader, train_data, dataset, dist_metric=dist_metric, prefix='train')
        if sim_mat is None:
            print(f'Metric: {dist_metric}, shape: None')
        else:
            print(f'Metric: {dist_metric}, shape: {sim_mat.shape}, max.: {sim_mat.max():.2f}, min.: {sim_mat.min():.2f}')
        
        sim_mat = datautils.get_sim_mat(args.loader, test_data, dataset, dist_metric=dist_metric, prefix='test')
        if sim_mat is None:
            print(f'Metric: {dist_metric}, shape: None')
        else:
            print(f'Metric: {dist_metric}, shape: {sim_mat.shape}, max.: {sim_mat.max():.2f}, min.: {sim_mat.min():.2f}')
        print('--- Similarity precomputing completed, time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time()-start_time)) + ' ---')

    print('--- Time elapsed in total: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time()-start_time)) + ' ---')
    sys.exit(0)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    main(args)
    