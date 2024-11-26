'''
This script defines the trainer class and grid search function for hyperparameter tuning.
'''

from model import spclt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from model_utils.utils_general import *
import model_utils.utils_data as datautils
random_seed = 131


# Define trainer
class trainer():
    def __init__(self, dist_metric='DTW', tau_inst=0, tau_temp=0, temporal_hierarchy=None, 
                 bandwidth=1., batch_size=8, weight_lr=0.05):
        self.dist_metric = dist_metric
        self.tau_inst = tau_inst
        self.tau_temp = tau_temp
        self.temporal_hierarchy = temporal_hierarchy
        self.bandwidth = bandwidth
        self.batch_size = batch_size
        self.weight_lr = weight_lr

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def define_encoder(self, loader, sim_mat, input_dims, device, regularizer=None):
        self.model_config = dict(
            input_dims = input_dims,
            output_dims = 320,
            dist_metric = self.dist_metric,
            device = device,
            batch_size = self.batch_size,
            lr = 0.001,
            weight_lr = self.weight_lr,
            loss_config = dict(
                tau_inst = self.tau_inst,
                tau_temp = self.tau_temp,
                lambda_inst = 0.5,
                temporal_hierarchy = self.temporal_hierarchy),
            regularizer_config = dict(
                reserve = regularizer,
                bandwidth = self.bandwidth),
            encode_args = dict(),
            )
        self.encoder = spclt(loader, **self.model_config)

        self.soft_assignments = datautils.assign_soft_labels(sim_mat, self.tau_inst)
        if self.soft_assignments is None:
            print('Soft assignment is not used in this run.')
        
        return self

    def fit(self, train_data, indexed_sim_mat, loader, dataset, encoder_config=None):
        self.original_size = int(dataset.split('_size_')[1])
        dataset = dataset.split('_size_')[0]
        if np.any(np.isnan(indexed_sim_mat[:,1:])):
            sim_mat = None
        else:
            selected_indices = indexed_sim_mat[:,0].reshape(-1).astype(int)
            sim_mat = indexed_sim_mat[:,-self.original_size:][:,selected_indices]
        encoder_config['input_dims'] = train_data.shape[-1]
        self = self.define_encoder(loader, sim_mat, **encoder_config)
        self.loss_log = self.encoder.fit(dataset, train_data, self.soft_assignments, 
                                         scheduler='constant', verbose=1)
        return self
    
    def get_params(self, deep=False):
        return dict(
            tau_inst = self.tau_inst,
            tau_temp = self.tau_temp,
            temporal_hierarchy = self.temporal_hierarchy,
            bandwidth = self.bandwidth,
            batch_size = self.batch_size,
            weight_lr = self.weight_lr
        )

    def score(self, test_data, indexed_sim_mat):
        if np.any(np.isnan(indexed_sim_mat[:,1:])):
            sim_mat = None
        else:
            selected_indices = indexed_sim_mat[:,0].reshape(-1).astype(int)
            sim_mat = indexed_sim_mat[:,-self.original_size:][:,selected_indices]
        soft_assignments = datautils.assign_soft_labels(sim_mat, self.tau_inst)
        return -self.encoder.compute_loss(test_data, soft_assignments, non_regularized=True)
    

def grid_search(params, loader, dataset, dist_metric, 
                train_data, indexed_sim_mat, n_fold, n_jobs, fit_config):
    if n_fold == 0:
        # Make sure the random seed is fixed for splitting data
        n_fold = ShuffleSplit(n_splits=1, test_size=0.3, random_state=random_seed)

    scorer = trainer(dist_metric)
    gs = GridSearchCV(scorer, params, cv=n_fold, n_jobs=n_jobs, verbose=0, refit=False)
    gs.fit(train_data, indexed_sim_mat, **{'loader': loader, 'dataset': dataset, 'encoder_config': fit_config})
    best_params, best_score = gs.best_params_, round(gs.best_score_, 4)

    del scorer
    del gs
    torch.cuda.empty_cache()
    return best_params, best_score
