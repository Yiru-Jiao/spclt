'''
This file contains utility functions for pretaining the encoders in the traffic tasks.
'''

import os
import sys
import torch
import random
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tasks.macro_modules.baselayer import Encoder
from tasks.macro_modules.utils_macro import adjacency_matrix, adjacency_matrixq
from tasks.micro_modules.baselayers import SubGraph
from modules.encoders import LSTMEncoder, GRUEncoder


def define_encoder(loader, device, model_dir, continue_training=True):
    encoder = spclt_copy(loader)
    best_model = [entry.name for entry in os.scandir(f'{model_dir}') if entry.name.split('_')[-1] == 'net.pth'][0]
    encoder.load(os.path.join(model_dir, best_model), device)
    if not continue_training:
        for param in encoder.net.parameters():
            param.requires_grad = False
    return encoder.net


class spclt_copy():
    def __init__(self, loader):
        super(spclt_copy, self).__init__()

        if loader == 'MacroTraffic':
            mat_A = adjacency_matrix(3)
            mat_B = adjacency_matrixq(3, 8)
            self.net = Encoder(nb_node=193, dim_feature=128, A=mat_A, B=mat_B)
        elif loader == 'MicroTraffic':
            self.net = SubGraph(8, 128, 9, 3)
        elif loader == 'MacroLSTM':
            self.net = LSTMEncoder(input_dim=193*2, 
                                   hidden_dim=193*4, 
                                   num_layers=2, 
                                   single_output=False)
        elif loader == 'MacroGRU':
            self.net = GRUEncoder(input_dim=193*2, 
                                  hidden_dim=193*4, 
                                  num_layers=2, 
                                  single_output=False)
        else:
            raise ValueError(f'Undefined loader: {loader}')

    def load(self, fn, device):
        state_dict = torch.load(fn, map_location=device, weights_only=True)
        self.net = self.net.to(device)
        self.net.load_state_dict(state_dict)


def fix_seed(seed, deterministic=False):
    random.seed(seed)
    seed += 1
    np.random.seed(seed)
    seed += 1
    torch.manual_seed(seed)
    seed += 1
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False


def init_dl_program(gpu_num=0, max_threads=None, use_tf32=False):

    if max_threads is not None:
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        else:
            try:
                import mkl
                mkl.set_num_threads(max_threads)
            except:
                pass
        
    if isinstance(gpu_num, (str, int)):
        if gpu_num == '0':
            device_name = ['cpu']
        elif ',' in gpu_num:
            device_name = ['cuda:'+idx for idx in gpu_num.split(',')]
            # Reduce VRAM usage by reducing fragmentation
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        else:
            device_name = [f'cuda:{idx}' for idx in range(int(gpu_num))]
            # Reduce VRAM usage by reducing fragmentation
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    devices = []
    for device in reversed(device_name):
        torch_device = torch.device(device)
        devices.append(torch_device)
        if torch_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(torch_device)
    devices.reverse()

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = use_tf32
            torch.backends.cuda.matmul.allow_tf32 = use_tf32
        
    return devices if len(devices) > 1 else devices[0]


def config_micro():
    paralist = dict(xmax = 23, ymin = -12, ymax = 75,
                    resolution = 1.,
                    nb_map_vectors = 5, nb_traj_vectors = 9,
                    map_dim = 5, traj_dim = 8,
                    nb_map_gnn = 5, nb_traj_gnn = 5, nb_mlp_layers = 3,
                    c_out_half = 32, c_mlp = 64, c_out = 96,
                    encoder_nb_heads = 3, encoder_attention_size = 64, encoder_agg_mode = "cat",
                    decoder_attention_size = 64, decoder_nb_heads = 3, decoder_agg_mode = "cat",
                    decoder_masker = False,
                    sigmax = 0.6, sigmay = 0.6,
                    r_list = [2,4,8,16], kf = 1,
                    model = 'densetnt', sample_range=1,
                    use_masker=False, lane2agent='lanegcn')
    return paralist