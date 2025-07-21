'''
This script defines models for structure-preserving time series contrastive learning.
The backbone is adapted from TS2Vec https://github.com/zhihanyue/ts2vec and SoftCLT https://github.com/seunghan96/softclt
'''

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from modules import encoders, losses
import model_utils.utils_data as datautils


class spclt():
    def __init__(
        self, loader,
        input_dims, output_dims=320, hidden_dims=64, dist_metric='EUC',
        depth=10, device='cuda', lr=0.001, weight_lr=0.05, batch_size=8,
        after_iter_callback=None, after_epoch_callback=None,
        loss_config=None,
        regularizer_config=None,
        encode_args=None,
        ):
        """
        Initialize the spclt model.
        """
        super(spclt, self).__init__()
        self.loader = loader
        self.dist_metric = dist_metric
        self.device = device
        self.lr = lr
        self.weight_lr = weight_lr
        self.batch_size = batch_size
        self.loss_config = loss_config
        self.regularizer_config = regularizer_config
        self.encode_args = encode_args
                
        # define encoder
        if self.loader == 'UEA':
            self.net = encoders.TSEncoder(input_dims=input_dims,
                                          output_dims=output_dims,
                                          hidden_dims=hidden_dims, 
                                          depth=depth).to(self.device)
        elif self.loader == 'MacroTraffic':
            mat_A = encoders.adjacency_matrix(3)
            mat_B = encoders.adjacency_matrixq(3, 8)
            self.net = encoders.DGCNEncoder(nb_node=193, 
                                            dim_feature=128, 
                                            A=mat_A, B=mat_B).to(self.device)
        elif self.loader == 'MicroTraffic':
            self.net = encoders.SubGraph(8, 128, 9, 3).to(self.device)
        elif self.loader == 'MacroLSTM':
            self.net = encoders.LSTMEncoder(input_dim=193*2,
                                            hidden_dim=193*4,
                                            num_layers=2, 
                                            single_output=True).to(self.device)
        elif self.loader == 'MacroGRU':
            self.net = encoders.GRUEncoder(input_dim=193*2,
                                           hidden_dim=193*4,
                                           num_layers=2,
                                           single_output=True).to(self.device)
        else:
            raise ValueError(f'Undefined loader: {self.loader}')

        # define learner of log variances used for weighing losses
        if self.regularizer_config['reserve'] == 'both':
            self.loss_log_vars = torch.nn.Parameter(torch.zeros(3, device=self.device))
        elif self.regularizer_config['reserve'] in ['topology', 'geometry']:
            self.loss_log_vars = torch.nn.Parameter(torch.zeros(2, device=self.device))
        if self.regularizer_config['baseline']:
            # let log(sigma^2) = 0, i.e., exp(-log(sigma^2)) = 1 and thus coefficient = 0.5
            self.loss_log_vars = torch.zeros(2, device=self.device, requires_grad=False)
        
        # define callback functions
        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback

    # define eval() and train() functions
    def eval(self,):
        if self.regularizer_config['reserve'] is None:
            self.net.eval()
        else:
            self.net.eval()
            self.loss_log_vars.requires_grad = False

    def train(self,):
        if self.regularizer_config['reserve'] is None or self.regularizer_config['baseline']:
            self.net.train()
        else:
            self.net.train()
            self.loss_log_vars.requires_grad = True

    # define fit() function
    def fit(self, name_data, train_data, soft_assignments=None, n_epochs=None, n_iters=None, scheduler='constant', verbose=0):
        """
        Fit the model to the training data.
        """
        if isinstance(soft_assignments, np.ndarray):
            assert soft_assignments.shape[0] == soft_assignments.shape[1]
            assert train_data.shape[0] == soft_assignments.shape[0]

        self.train()

        # Set default number for n_iters, this is intended for underfitting the model
        if n_iters is None and n_epochs is None:
            num_samples = train_data.shape[0]
            n_iters = num_samples // 2
            sample_bounds = [100, 500, 1000, 10000]
            coefs = [2, 1, 0.5, 0.25]
            for bound, coef in zip(sample_bounds, coefs):
                if num_samples < bound:
                    n_iters = int(n_iters * coef)
                    break
            if num_samples >= sample_bounds[-1]:
                n_iters = n_iters // 8

        # define a progress bar
        if n_epochs is not None:
            if verbose:
                progress_bar = tqdm(range(n_epochs), desc=f'Train {name_data} epoch', ascii=True, dynamic_ncols=False)
            else:
                progress_bar = range(n_epochs)
        elif n_iters is not None:
            if verbose:
                progress_bar = tqdm(range(n_iters), desc=f'Train {name_data} iter', ascii=True, dynamic_ncols=False)
            else:
                progress_bar = range(n_iters)
        else:
            ValueError('At least one between n_epochs and n_iters should be specified')
        
        # define optimizer
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr)
        if self.regularizer_config['reserve'] in ['topology', 'geometry', 'both']:
            self.optimizer_weight = torch.optim.AdamW([self.loss_log_vars], lr=self.weight_lr)
            def optimizer_zero_grad():
                self.optimizer.zero_grad()
                self.optimizer_weight.zero_grad()
            def optimizer_step():
                self.optimizer.step()
                self.optimizer_weight.step()
        else:
            def optimizer_zero_grad():
                self.optimizer.zero_grad()
            def optimizer_step():
                self.optimizer.step()

        # exclude instances with all missing values
        isnanmat = np.isnan(train_data)
        while isnanmat.ndim > 1:
            isnanmat = isnanmat.all(axis=-1)
        reserved_idx = ~isnanmat

        # define training and validation data
        if scheduler == 'constant':
            train_data = train_data[reserved_idx]            
            if soft_assignments is None:
                train_soft_assignments = None
            elif isinstance(soft_assignments, str):
                train_soft_assignments = 'compute'
            else:
                train_soft_assignments = soft_assignments[reserved_idx][:,reserved_idx].copy()
            del soft_assignments, reserved_idx
        elif scheduler == 'reduced':
            train_val_data = train_data[reserved_idx]
            # randomly split the training data into training and validation sets, fix seed for consistency across losses
            val_indices = np.random.RandomState(131).choice(len(train_val_data), int(len(train_val_data)*0.25), replace=False)
            train_indices = np.setdiff1d(np.arange(len(train_val_data)), val_indices)
            train_data, val_data = train_val_data[train_indices].copy(), train_val_data[val_indices].copy()
            if soft_assignments is None:
                train_soft_assignments, val_soft_assignments = None, None
            elif isinstance(soft_assignments, str):
                train_soft_assignments, val_soft_assignments = 'compute', 'compute'
            else:
                train_val_soft_assignments = soft_assignments[reserved_idx][:,reserved_idx]
                train_soft_assignments = train_val_soft_assignments[train_indices][:,train_indices].copy()
                val_soft_assignments = train_val_soft_assignments[val_indices][:,val_indices].copy()
                del train_val_soft_assignments
            del soft_assignments, reserved_idx, train_val_data, train_indices, val_indices
            val_dataset = datautils.custom_dataset(torch.from_numpy(val_data).float(), self.loader)
            val_loader = DataLoader(val_dataset, batch_size=min(self.batch_size, len(val_dataset)), shuffle=False, drop_last=True)
            
            # define scheduler
            if self.loader == 'UEA':
                if train_data.shape[0]<5000:
                    patience = 10
                    self.initial_cooldown_weight = 50
                    self.initial_cooldown_model = 25
                else:
                    patience = 6
                    self.initial_cooldown_weight = 20
                    self.initial_cooldown_model = 10
            else:
                if self.loader == 'MacroTraffic':
                    patience = 8
                    self.initial_cooldown_weight = 10
                    self.initial_cooldown_model = 5
                elif self.loader == 'MicroTraffic':
                    patience = 4
                    self.initial_cooldown_weight = 4
                    self.initial_cooldown_model = 2
                elif self.loader in ['MacroLSTM', 'MacroGRU']:
                    patience = 8
                    self.initial_cooldown_weight = 8
                    self.initial_cooldown_model = 4

            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.4, patience=patience, cooldown=0,
                threshold=1e-3, threshold_mode='rel', min_lr=self.lr*0.4**30
                )
            
            if self.regularizer_config['reserve'] is not None and not self.regularizer_config['baseline']:
                self.scheduler_weight = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer_weight, mode='min', factor=0.6, patience=patience//2, cooldown=0,
                    threshold=1e-3, threshold_mode='rel', min_lr=self.weight_lr*0.6**50
                    )
                def scheduler_update(val_loss, regularizer_loss, loss):
                    log_sum = 0.5*self.loss_log_vars.sum()
                    val_loss += (loss - log_sum)
                    regularizer_loss += log_sum
                    return val_loss, regularizer_loss
                def scheduler_step(val_loss_log, val_loss, regularizer_loss):
                    val_loss_log[self.epoch_n+4, 0] = val_loss
                    val_loss_log[self.epoch_n+4, 1] = regularizer_loss
                    if self.epoch_n > self.initial_cooldown_model:
                        self.scheduler.step(val_loss)
                    if self.epoch_n > self.initial_cooldown_weight:
                        self.scheduler_weight.step(regularizer_loss)
                    return val_loss_log
            else:
                def scheduler_update(val_loss, regularizer_loss, loss):
                    val_loss += loss
                    return val_loss, val_loss
                def scheduler_step(val_loss_log, val_loss, regularizer_loss):
                    val_loss_log[self.epoch_n+4, 0] = val_loss
                    val_loss_log[self.epoch_n+4, 1] = val_loss
                    if self.epoch_n > self.initial_cooldown_model:
                        self.scheduler.step(val_loss)
                    return val_loss_log

            if n_epochs is not None:
                val_loss_log = np.zeros((n_epochs+4, 2)) * np.nan
            else:
                val_loss_log = np.zeros((int(n_iters/len(val_loader))+4, 2)) * np.nan
            val_loss_log[:4,:] = np.array([[init_loss]*2 for init_loss in range(5, 1, -1)])
        else:
            ValueError("Undefined scheduler: should be either 'constant' or 'reduced'.")

        # create training dataset, dataloader, and loss log
        train_dataset = datautils.custom_dataset(torch.from_numpy(train_data).float(), self.loader)
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True, drop_last=True)
        if n_iters is None:
            log_len = n_epochs
        else:
            log_len = n_iters
        if self.regularizer_config['reserve'] is None:
            loss_log = np.zeros((log_len, 2)) * np.nan
        elif self.regularizer_config['reserve'] == 'both':
            loss_log = np.zeros((log_len, 7)) * np.nan
        elif self.regularizer_config['reserve'] in ['topology', 'geometry']:
            loss_log = np.zeros((log_len, 5)) * np.nan

        # define loss function
        self.loss_func = losses.combined_loss

        # training loop
        self.epoch_n = 0
        self.iter_n = 0
        if self.loss_config is None:
            self.loss_config = {'temporal_unit': 0, 'tau_inst': 0}
        else:
            self.loss_config['temporal_unit'] = 0  ## The minimum unit to perform temporal contrast. 
                                                   ## When training on a very long sequence, increasing this helps to reduce the cost of time and memory.
        continue_training = True
        while continue_training:
            train_loss = torch.tensor(0., device=self.device, requires_grad=False)
            train_loss_comp = {}
            for train_batch_iter, (x, idx) in enumerate(train_loader, start=1):
                if train_soft_assignments is None:
                    soft_labels = None
                elif isinstance(train_soft_assignments, str):
                    batch_sim_mat = datautils.compute_sim_mat(x.numpy(), self.dist_metric)
                    soft_labels = datautils.assign_soft_labels(batch_sim_mat, self.loss_config['tau_inst'])
                    soft_labels = torch.from_numpy(soft_labels).float().to(self.device)
                else:
                    soft_labels = train_soft_assignments[idx][:,idx] # (B, B)
                    soft_labels = torch.from_numpy(soft_labels).float().to(self.device)
                train_loss_config = self.loss_config.copy()
                train_loss_config['soft_labels'] = soft_labels

                optimizer_zero_grad()
                loss, loss_comp = self.loss_func(self, x.to(self.device),
                                                 train_loss_config, 
                                                 self.regularizer_config)
                loss.backward()
                optimizer_step()

                train_loss += loss
                if len(train_loss_comp) == 0:
                    for key in loss_comp:
                        train_loss_comp[key] = torch.tensor(0., device=self.device, requires_grad=False)                    
                for key in train_loss_comp:
                    train_loss_comp[key] += loss_comp[key]

                # save model if callback every several iterations
                if self.after_iter_callback is not None:
                    self.after_iter_callback(self)

                # update progress bar if n_iters is specified
                if n_iters is not None and verbose:
                    if n_iters % verbose == (verbose-1):
                        progress_bar.set_postfix(loss=loss.item(), refresh=False)
                        progress_bar.update(verbose)

                self.iter_n += 1
                if n_iters is not None and self.iter_n >= n_iters:
                    continue_training = False
                    break

            # save average loss per epoch
            train_loss /= train_batch_iter
            for key in train_loss_comp:
                train_loss_comp[key] = (train_loss_comp[key] / train_batch_iter).item()
            loss_log[self.epoch_n] = [train_loss.item()] + list(train_loss_comp.values())

            # if the scheduler is set to 'reduced', evaluate validation loss and update learning rate
            if scheduler == 'reduced':
                self.eval()
                val_loss = torch.tensor(0., device=self.device, requires_grad=False)
                regularizer_loss = torch.tensor(0., device=self.device, requires_grad=False)
                with torch.no_grad():
                    for val_batch_iter, (x, idx) in enumerate(val_loader, start=1):
                        if val_soft_assignments is None:
                            soft_labels = None
                        elif isinstance(val_soft_assignments, str):
                            batch_sim_mat = datautils.compute_sim_mat(x.numpy(), self.dist_metric)
                            soft_labels = datautils.assign_soft_labels(batch_sim_mat, self.loss_config['tau_inst'])
                            soft_labels = torch.from_numpy(soft_labels).float().to(self.device)
                        else:
                            soft_labels = val_soft_assignments[idx][:,idx]
                            soft_labels = torch.from_numpy(soft_labels).float().to(self.device)
                        val_loss_config = self.loss_config.copy()
                        val_loss_config['soft_labels'] = soft_labels

                        loss, _ = self.loss_func(self, x.to(self.device),
                                                     val_loss_config, 
                                                     self.regularizer_config)
                        val_loss, regularizer_loss = scheduler_update(val_loss, regularizer_loss, loss)
                val_loss /= val_batch_iter
                regularizer_loss /= val_batch_iter
                scheduler_step(val_loss_log, val_loss.item(), regularizer_loss.item())
                self.train()

                if self.epoch_n>10 and np.all(abs(np.diff(val_loss_log[self.epoch_n:self.epoch_n+5, 0])/val_loss_log[self.epoch_n:self.epoch_n+4, 0]) < 5e-4):
                    Warning('Early stopping if validation loss converges.')
                    continue_training = False
                    break

            # save model if callback every several epochs
            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self)

            # update progress bar if n_epochs is specified
            if n_epochs is not None and verbose:
                avg_batch_loss = loss_log[self.epoch_n, 0]
                if scheduler == 'reduced':
                    avg_val_loss = val_loss_log[self.epoch_n+4, 0]
                    current_lr = self.optimizer.param_groups[0]['lr']
                if n_epochs % verbose == (verbose-1):
                    if scheduler == 'reduced':
                        progress_bar.set_postfix(loss=avg_batch_loss, val_loss=avg_val_loss, lr=current_lr, refresh=False)
                    else:
                        progress_bar.set_postfix(loss=avg_batch_loss, refresh=False)
                    progress_bar.update(verbose)

            self.epoch_n += 1
            if n_epochs is not None and self.epoch_n >= n_epochs:
                continue_training = False
                break
        
        progress_bar.close()
        if self.after_iter_callback is not None:
            self.after_iter_callback(self, finish=True)
        if self.after_epoch_callback is not None:
            self.after_epoch_callback(self, finish=True)

        return loss_log
    

    def compute_loss(self, val_data, soft_assignments, non_regularized=False, loss_config=None):
        """
        Computes the loss for the given validation data and soft assignments.
        """
        assert self.net is not None, 'please train or load a model first'
        if isinstance(soft_assignments, np.ndarray):
            assert soft_assignments.shape[0] == soft_assignments.shape[1]
            assert val_data.shape[0] == soft_assignments.shape[0] if soft_assignments is not None else True

        # create test dataset and dataloader
        val_dataset = datautils.custom_dataset(torch.from_numpy(val_data).float(), self.loader)
        val_loader = DataLoader(val_dataset, batch_size=min(self.batch_size, len(val_dataset)), shuffle=False, drop_last=True)

        # define loss function
        self.loss_func = losses.combined_loss

        if self.loss_config is None:
            self.loss_config = {'temporal_unit': 0, 'tau_inst': 0}
        else:
            self.loss_config['temporal_unit'] = 0  ## The minimum unit to perform temporal contrast. 
                                                   ## When training on a very long sequence, increasing this helps to reduce the cost of time and memory.
        if loss_config is None:
            loss_config = self.loss_config
        else:
            loss_config['temporal_unit'] = 0
 
        org_training = self.net.training
        self.eval()
        val_loss_comp = {}
        with torch.no_grad():
            for val_batch_iter, (x, idx) in enumerate(val_loader, start=1):
                if soft_assignments is None:
                    soft_labels = None
                elif isinstance(soft_assignments, str):
                    batch_sim_mat = datautils.compute_sim_mat(x.numpy(), self.dist_metric)
                    soft_labels = datautils.assign_soft_labels(batch_sim_mat, loss_config['tau_inst'])
                    soft_labels = torch.from_numpy(soft_labels).float().to(self.device)
                else:
                    soft_labels = soft_assignments[idx][:,idx]
                    soft_labels = torch.from_numpy(soft_labels).float().to(self.device)
                val_loss_config = loss_config.copy()
                val_loss_config['soft_labels'] = soft_labels

                loss, loss_comp = self.loss_func(self, x.to(self.device),
                                                         val_loss_config, 
                                                         self.regularizer_config)
                if len(val_loss_comp) == 0:
                    for key in loss_comp:
                        val_loss_comp[key] = torch.tensor(0., device=self.device, requires_grad=False)
                for key in val_loss_comp:
                    val_loss_comp[key] += loss_comp[key]
                
        if org_training:
            self.train()

        for key in val_loss_comp:
            val_loss_comp[key] = (val_loss_comp[key] / val_batch_iter).item()

        if non_regularized:
            return val_loss_comp['loss_scl']
        else:
            return val_loss_comp.values()


    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        """
        Evaluate the network output with optional pooling and slicing.
        """
        out = self.net(x.to(self.device), mask)
        if encoding_window == 'full_series':
            if slicing is not None:
                out = out[:, slicing]
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = out.size(1),
            ).transpose(1, 2)
            
        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = encoding_window,
                stride = 1,
                padding = encoding_window // 2
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]
            
        elif encoding_window == 'multiscale':
            p = 0
            reprs = []
            while (1 << p) + 1 < out.size(1):
                t_out = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size = (1 << (p + 1)) + 1,
                    stride = 1,
                    padding = 1 << p
                ).transpose(1, 2)
                if slicing is not None:
                    t_out = t_out[:, slicing]
                reprs.append(t_out)
                p += 1
            out = torch.cat(reprs, dim=-1)
            
        else:
            if slicing is not None:
                out = out[:, slicing]
            
        return out


    def torch_pad_nan(arr, left=0, right=0, dim=0):
        if left > 0:
            padshape = list(arr.shape)
            padshape[dim] = left
            arr = torch.cat((torch.full(padshape, np.nan), arr), dim=dim)
        if right > 0:
            padshape = list(arr.shape)
            padshape[dim] = right
            arr = torch.cat((arr, torch.full(padshape, np.nan)), dim=dim)
        return arr    
    
    
    def encode(self, data, mask=None, encoding_window=None, causal=False, sliding_length=None, sliding_padding=0, batch_size=None):
        """
        Encode the input data using the trained neural network model.
        """
        assert self.net is not None, 'please train or load a net first'
        org_training = self.net.training
        self.net.eval()
        
        if self.loader == 'UEA':
            assert data.ndim == 3
            n_samples, ts_l, _ = data.shape

        if batch_size is None:
            batch_size = self.batch_size

        if isinstance(data, torch.Tensor):
            dataset = datautils.custom_dataset(data, self.loader)
        else:
            dataset = datautils.custom_dataset(torch.from_numpy(data).float(), self.loader)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)        

        if 'Macro' in self.loader or 'Micro' in self.loader:
            with torch.no_grad():
                output = []
                for x, _ in dataloader:
                    x = x.to(self.device)
                    out = self.net(x) # MacroTraffic: (B, n_node=193, dim_feature/2=64)
                                      # MicroTraffic: (B, n_agent=26, dim_feature=128)
                                      # MacroLSTM: (B, n_node=193, dim_feature=4)
                                      # MacroGRU: (B, n_node=193, dim_feature=4)
                    output.append(out)
                output = torch.cat(output, dim=0)
        elif self.loader == 'UEA':
            with torch.no_grad():
                output = []
                for x, _ in dataloader:
                    x = x.to(self.device)
                    if sliding_length is not None:
                        reprs = []
                        if n_samples < batch_size:
                            calc_buffer = []
                            calc_buffer_l = 0
                        for i in range(0, ts_l, sliding_length):
                            l = i - sliding_padding
                            r = i + sliding_length + (sliding_padding if not causal else 0)
                            x_sliding = self.torch_pad_nan(
                                x[:, max(l, 0) : min(r, ts_l)],
                                left=-l if l<0 else 0,
                                right=r-ts_l if r>ts_l else 0,
                                dim=1
                            )
                            if n_samples < batch_size:
                                if calc_buffer_l + n_samples > batch_size:
                                    out = self._eval_with_pooling(
                                        torch.cat(calc_buffer, dim=0),
                                        mask,
                                        slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                        encoding_window=encoding_window
                                    )
                                    reprs += torch.split(out, n_samples)
                                    calc_buffer = []
                                    calc_buffer_l = 0
                                calc_buffer.append(x_sliding)
                                calc_buffer_l += n_samples
                            else:
                                out = self._eval_with_pooling(
                                    x_sliding,
                                    mask,
                                    slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                    encoding_window=encoding_window
                                )
                                reprs.append(out)

                        if n_samples < batch_size:
                            if calc_buffer_l > 0:
                                out = self._eval_with_pooling(
                                    torch.cat(calc_buffer, dim=0),
                                    mask,
                                    slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                    encoding_window=encoding_window
                                )
                                reprs += torch.split(out, n_samples)
                                calc_buffer = []
                                calc_buffer_l = 0
                        
                        out = torch.cat(reprs, dim=1)
                        if encoding_window == 'full_series':
                            out = F.max_pool1d(
                                out.transpose(1, 2).contiguous(),
                                kernel_size = out.size(1),
                            ).squeeze(1)
                    else:
                        out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                        if encoding_window == 'full_series':
                            out = out.squeeze(1)
                            
                    output.append(out)
                    
                output = torch.cat(output, dim=0)
            
        self.net.train(org_training)
        return output


    def save(self, fn):
        """
        Save the model's state dictionary and loss log variables to files.
        """
        torch.save(self.net.state_dict(), fn+'_net.pth')
        if self.regularizer_config['reserve'] is not None:
            state_loss_log_vars = self.loss_log_vars.detach().cpu().numpy()
            np.save(fn+'_loss_log_vars.npy', state_loss_log_vars)


    def load(self, fn):
        """
        Load the model state and associated parameters from the specified file.
        """
        state_dict = torch.load(fn+'_net.pth', map_location=self.device, weights_only=True)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        if self.regularizer_config['reserve'] is not None:
            state_loss_log_vars = np.load(fn+'_loss_log_vars.npy')
            state_loss_log_vars = torch.from_numpy(state_loss_log_vars).to(self.device)
            self.loss_log_vars = torch.nn.Parameter(state_loss_log_vars, requires_grad=False)
        if self.regularizer_config['baseline']:
            self.loss_log_vars = torch.zeros(2, device=self.device, requires_grad=False)
