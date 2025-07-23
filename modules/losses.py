'''
This script defines the loss functions.
The hierarchical contrastive loss reuses and adapts the code of TS2Vec and SoftCLT.
This file contains the implementation of the topological and geometric regularizers.
'''

import torch.nn.functional as F
from modules.loss_utils import *


def combined_loss(model, x, loss_config, regularizer_config):
    temporal_unit = loss_config['temporal_unit']
    out1, out2 = mask_and_crop(model.net, x, temporal_unit)
    loss_scl = hierarchical_contrastive_loss(out1, out2, **loss_config)
    loss_components = {}

    if regularizer_config['reserve'] is None:
        loss = loss_scl
        loss_components['loss_scl'] = loss_scl
    else:
        loss = 0.5 * torch.exp(-model.loss_log_vars[0]) * loss_scl*(1-torch.exp(-loss_scl)) + 0.5 * model.loss_log_vars[0]
        loss_components['loss_scl'] = loss_scl
        loss_components['log_var_scl'] = model.loss_log_vars[0]
        
        if regularizer_config['reserve'] == 'topology':
            loss_topo_regularizer = topo_loss(model, x, regularizer_config['x_max'], regularizer_config['x_min'])
            loss += 0.5 * torch.exp(-model.loss_log_vars[1]) * loss_topo_regularizer*(1-torch.exp(-loss_topo_regularizer)) + 0.5 * model.loss_log_vars[1]
            loss_components['loss_topo_regularizer'] = loss_topo_regularizer
            loss_components['log_var_topo'] = model.loss_log_vars[1]
        elif regularizer_config['reserve'] == 'geometry':
            loss_geo_regularizer = geo_loss(model, x, regularizer_config['x_max'], regularizer_config['x_min'], regularizer_config['bandwidth'])
            loss += 0.5 * torch.exp(-model.loss_log_vars[1]) * loss_geo_regularizer*(1-torch.exp(-loss_geo_regularizer)) + 0.5 * model.loss_log_vars[1]
            loss_components['loss_geo_regularizer'] = loss_geo_regularizer
            loss_components['log_var_geo'] = model.loss_log_vars[1]
        elif regularizer_config['reserve'] == 'both':
            loss_topo_regularizer = topo_loss(model, x, regularizer_config['x_max'], regularizer_config['x_min'])
            loss_geo_regularizer = geo_loss(model, x, regularizer_config['x_max'], regularizer_config['x_min'], regularizer_config['bandwidth'])
            loss += 0.5 * torch.exp(-model.loss_log_vars[1]) * loss_topo_regularizer*(1-torch.exp(-loss_topo_regularizer)) + 0.5 * model.loss_log_vars[1]
            loss += 0.5 * torch.exp(-model.loss_log_vars[2]) * loss_geo_regularizer*(1-torch.exp(-loss_geo_regularizer)) + 0.5 * model.loss_log_vars[2]
            loss_components['loss_topo_regularizer'] = loss_topo_regularizer
            loss_components['loss_geo_regularizer'] = loss_geo_regularizer
            loss_components['log_var_topo'] = model.loss_log_vars[1]
            loss_components['log_var_geo'] = model.loss_log_vars[2]
        else:
            raise ValueError('Undefined regularizer, should be either "topology", "geometry" or "both"')    
    return loss, loss_components


def instance_contrastive_loss(z1, z2, soft_or_hard=('hard',)):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    i = torch.arange(B, device=z1.device)
    if soft_or_hard[0] == 'hard':
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    elif soft_or_hard[0] == 'soft':
        soft_labels_L, soft_labels_R = soft_or_hard[1], soft_or_hard[2]
        loss = torch.sum(logits[:,i]*soft_labels_L)
        loss += torch.sum(logits[:,B + i]*soft_labels_R)
        loss /= (2*B*T)

    return loss


def temporal_contrastive_loss(z1, z2, soft_or_hard=('hard',)):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    t = torch.arange(T, device=z1.device)
    if soft_or_hard[0] == 'hard':
        loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    elif soft_or_hard[0] == 'soft':
        timelag_L, timelag_R = soft_or_hard[1], soft_or_hard[2]
        loss = torch.sum(logits[:,t]*timelag_L)
        loss += torch.sum(logits[:,T + t]*timelag_R)
        loss /= (2*B*T)
        
    return loss


def hierarchical_contrastive_loss(z1, z2, temporal_unit=0, lambda_inst=0.5, 
                                  soft_labels=None, tau_inst=0, tau_temp=0, temporal_hierarchy=None,
                                  ):
    
    if soft_labels is not None:
        soft_labels_L, soft_labels_R = dup_matrix(soft_labels)

    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if lambda_inst != 0:
            if tau_inst > 0:
                soft_or_hard = ('soft', soft_labels_L, soft_labels_R)
            else:
                soft_or_hard = ('hard',)
            loss += lambda_inst * instance_contrastive_loss(z1, z2, soft_or_hard)

        if d >= temporal_unit and 1 - lambda_inst != 0:
            if tau_temp > 0:
                if temporal_hierarchy is None:
                    timelag = timelag_sigmoid(z1, tau_temp)
                else:
                    if temporal_hierarchy=='exponential':
                        timelag = timelag_sigmoid(z1, tau_temp*(2**d)) # 2**d because kernel_size in max_pool1d is 2
                    elif temporal_hierarchy=='linear':
                        timelag = timelag_sigmoid(z1, tau_temp*(d+1))
                    else:
                        raise ValueError('Undefined temporal_hierarchy, should be either "exponential" or "linear"')
                    
                timelag_L, timelag_R = dup_matrix(timelag)
                soft_or_hard = ('soft', timelag_L, timelag_R)
            else:
                soft_or_hard = ('hard',)
            loss += (1 - lambda_inst) * temporal_contrastive_loss(z1, z2, soft_or_hard)

        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1.size(1) == 1 and lambda_inst != 0:
        if tau_inst > 0:
            loss += lambda_inst * instance_contrastive_loss(z1, z2, ('soft', soft_labels_L, soft_labels_R))
        else:
            loss += lambda_inst * instance_contrastive_loss(z1, z2)
        d += 1

    return loss / d


def topo_loss(model, x, x_max=None, x_min=None):
    # encode using model
    latent = model.encode(x, **model.encode_args)

    # compute and normalize distances in the original sapce and latent space
    x_distances = topo_euclidean_distance_matrix(x, x_max, x_min) # (B, N, N)
    x_distances = x_distances / max(x_distances.max(), 1.)
    latent_distances = topo_euclidean_distance_matrix(latent) # (B, N, N)
    latent_distances = latent_distances / max(latent_distances.max(), 1.)

    # compute topological signature distance
    topo_sig = TopologicalSignatureDistance()
    topo_error = topo_sig(x_distances, latent_distances)

    # normalize topo_error according to batch_size
    batch_size = x.size()[0]
    topo_error = topo_error / float(batch_size)

    return topo_error


def geo_loss(model, x, x_max=None, x_min=None, bandwidth=1.):
    # encode using model
    latent = model.encode(x)
    """
    For MicroTraffic data,
    the original shape of x is (n_samples, n_agents, n_timesteps, n_features), 
    the latent shape is (n_samples, n_agents=26, n_latent_features=128)
    ggeo_loss will be computed for agent dimension
    For MacroTraffic data, the shape of x is (n_samples, n_timesteps, n_nodes, n_features)
    the latent will be computed for node dimension
    """    
    # Switch axes to (n_samples, n_timesteps, n_agents, n_features) for MicroTraffic data
    if model.loader == 'MicroTraffic':
        x = x.permute(0, 2, 1, 3)
    
    L = get_laplacian(x, x_max, x_min, bandwidth=bandwidth)
    distortion, n = relaxed_distortion_measure_JGinvJT(L, latent)

    # Normalize distortion
    iso_loss = distortion / n + 1.
    return iso_loss
