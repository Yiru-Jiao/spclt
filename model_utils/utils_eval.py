'''
This script includes utility functions for evaluation of the models.
'''

import os
import sys
from tqdm import tqdm
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.measures import MeasureCalculator
from model_utils.utils_data import get_sim_mat
from model_utils.utils_distance_matrix import get_EUC


class Multi_Evaluation:
    def __init__(self, loader, dataset, data, latent):
        self.loader = loader
        self.dataset = dataset
        self.data = data
        self.latent = latent

    def define_ks(self, dist_mat_X):
        # define k values for evaluation, logarithmically spaced
        k_neighbours = np.unique(np.logspace(1, np.log(min(dist_mat_X.shape[0]/3,200))/np.log(5), num=10, base=5).astype(int))
        return k_neighbours

    def get_multi_evals(self, local=False, drmse_only=False, verbose=False):
        """
        Performs multiple evaluations for nonlinear dimensionality
        reduction.

        - data: data samples as matrix
        - latent: latent samples as matrix
        - local: whether to use local or global evaluation
        - ks: list of k values for evaluation
        """
        if local:
            dep_measures_list = {'mean_shared_neighbours':0., 
                                 'mean_dist_mrre':0., 
                                 'mean_trustworthiness':0, 
                                 'mean_continuity':0.}
            
            # for UEA N is time series length, for MacroTraffic N is number of nodes, for MicroTraffic N is number of agents
            N = self.data.shape[-2]
            sample_indices = np.arange(self.data.shape[0])
            sample_count = 0
            dist_mat_measure = {'local_distmat_rmse': 0}
            if verbose:
                progress_bar = tqdm(sample_indices, desc='Local Evaluation', ascii=True, dynamic_ncols=False, miniters=5)
            else:
                progress_bar = sample_indices
            for sample_index in progress_bar:
                data = self.data[sample_index].reshape(N, -1)
                dist_mat_X = get_EUC(data)
                latent = self.latent[sample_index].reshape(N, -1)
                dist_mat_Z = get_EUC(latent)

                if dist_mat_X.max()-dist_mat_X.min() == 0 or dist_mat_Z.max()-dist_mat_Z.min() == 0:
                    continue
                else:
                    dist_mat_X = abs((dist_mat_X - dist_mat_X.min()) / (dist_mat_X.max() - dist_mat_X.min()))
                    dist_mat_Z = abs((dist_mat_Z - dist_mat_Z.min()) / (dist_mat_Z.max() - dist_mat_Z.min()))

                dist_mat_measure['local_distmat_rmse'] += np.sqrt(np.mean((dist_mat_X - dist_mat_Z)**2))

                if not drmse_only:
                    ks = self.define_ks(dist_mat_X)
                    calc = MeasureCalculator(dist_mat_X, dist_mat_Z, max(ks))

                    dep_measures = calc.compute_measures_for_ks(ks)
                    mean_dep_measures = {'mean_'+key: np.nanmean(values) for key, values in dep_measures.items()}
                    for key, value in mean_dep_measures.items():
                        dep_measures_list[key] += value

                sample_count += 1
                if sample_count >= 500:
                    break
            
            dist_mat_measure['local_distmat_rmse'] /= sample_count
            if drmse_only:
                results = dist_mat_measure
            else:
                dep_measures = {'local_'+key: value/sample_count for key, value in dep_measures_list.items()}
                results = {**dist_mat_measure, **dep_measures}
        else:
            sim_mat_X = get_sim_mat(self.loader, self.data, self.dataset, dist_metric='EUC', prefix='test')
            sim_mat_X = abs((sim_mat_X - sim_mat_X.min()) / (sim_mat_X.max() - sim_mat_X.min()))
            dist_mat_X = abs(1 - sim_mat_X)
            dist_mat_Z = get_EUC(self.latent.reshape(self.latent.shape[0], -1))
            dist_mat_Z = abs((dist_mat_Z - dist_mat_Z.min()) / (dist_mat_Z.max() - dist_mat_Z.min()))

            dist_mat_measure = {'global_distmat_rmse': np.sqrt(np.mean((dist_mat_X - dist_mat_Z)**2))}

            if drmse_only:
                results = dist_mat_measure
            else:
                ks = self.define_ks(dist_mat_X)
                calc = MeasureCalculator(dist_mat_X, dist_mat_Z, max(ks))

                dep_measures = calc.compute_measures_for_ks(ks)
                mean_dep_measures = {'global_mean_' + key: np.nanmean(values) for key, values in dep_measures.items()}

                results = {**dist_mat_measure, **mean_dep_measures}
            
        return results


def evaluate(loader, dataset, data, labels, model, batch_size, local=False, drmse_only=False, save_latents=False, save_dir=None):
    # encode data into latent space
    if local:
        latent = model.encode(data, batch_size=batch_size).detach().cpu().numpy() # (N, T, P)
    else:
        latent = model.encode(data, batch_size=batch_size, encoding_window='full_series').detach().cpu().numpy() # (N, P)
        if save_latents:
            np.savez(
                os.path.join(save_dir, 'latents_local.npz' if local else 'latents_global.npz'),
                latents=latent, labels=labels
            )

    # switch axes to (n_samples, n_timesteps, n_agents, n_features) for MicroTraffic data
    if model.loader == 'MicroTraffic':
        data = data.transpose(0, 2, 1, 3)

    evaluator = Multi_Evaluation(loader, dataset, data, latent)
    if dataset in ['EigenWorms', 'MotorImagery']:
        verbose = True
    else:
        verbose = False
    ev_result = evaluator.get_multi_evals(local, drmse_only, verbose)

    return ev_result

