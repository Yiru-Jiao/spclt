'''
This script is the encoders for the models.
TSEncoder for UEA datasets is adapted from TS2Vec https://github.com/zhihanyue/ts2vec
DGCNEncoder for macro-traffic data is adapted from https://github.com/RomainLITUD/uncertainty-aware-traffic-speed-flow-demand-prediction
GraphEncoder for micro-traffic data is adapted from https://github.com/RomainLITUD/UQnet-arxiv
LSTMEncoder and GRUEncoder are classical baselines for traffic prediction.
'''

import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


########################################################
##             TSEncoder for UEA data                 ##
########################################################

def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            res[i, t:t+l] = False
    return res


def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    
    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])
        
    def forward(self, x):
        return self.net(x)


class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial'):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        
    def forward(self, x, mask=None):
        nan_mask = torch.isnan(x).any(axis=-1)
        x = torch.where(nan_mask, torch.zeros_like(x), x)
        x = self.input_fc(x)  # B x T x Ch
        
        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        
        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        
        mask &= nan_mask
        x[~mask] = 0
        
        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co
        
        return x


########################################################
##        DGCNEncoder for macro-traffic data          ##
########################################################

def adjacency_matrix(k):
    acc = [-1,36,47,58,74,99,110,144,154,192]
    
    base = np.identity(193,dtype=bool)
    
    for i in range(0,193):
        if i not in acc:
            base[i][i+1]=True
            
    base[36][37]=True
    base[36][75]=True
    base[47][48]=True
    base[110][48]=True
    base[99][100]=True
    base[99][111]=True
    base[58][59]=True
    base[58][145]=True
    base[144][155]=True
    base[154][155]=True
    base[192][0]=True
    base[74][0]=True
    
    both = np.logical_or(base, base.transpose())

    output = np.linalg.matrix_power(both, k)
    output[output > 0] = 1.
    
    return torch.Tensor(output)


def adjacency_matrixq(k1, k2):
    acc = [-1,36,47,58,74,99,110,144,154,192]
    
    base = np.identity(193,dtype=bool)
    
    for i in range(0,193):
        if i not in acc:
            base[i][i+1]=True
            
    base[36][37]=True
    base[36][75]=True
    base[47][48]=True
    base[110][48]=True
    base[99][100]=True
    base[99][111]=True
    base[58][59]=True
    base[58][145]=True
    base[144][155]=True
    base[154][155]=True
    base[192][0]=True
    base[74][0]=True
    
    downstream = np.linalg.matrix_power(base, k1)
    upstream = np.linalg.matrix_power(base.T, k2)
    both = np.logical_or(upstream, downstream)

    both[both > 0] = 1.
    
    return torch.Tensor(both)


class LocalGC(nn.Module):
    def __init__(self, nb_node, dim_feature, A):
        super(LocalGC, self).__init__()
        self.N = nb_node
        self.F = dim_feature

        self.w1 = nn.Parameter(nn.init.xavier_normal_(torch.empty(self.N, self.N)))
        self.bias = nn.Parameter(torch.zeros(self.F,))

        self.A = nn.Parameter(A, requires_grad=False)

    def forward(self, x):
        W = self.A*self.w1
        x = torch.matmul(x.transpose(-1,-2), W)
        output = x.transpose(-1,-2) + self.bias

        return output


class DoubleDGC(nn.Module):
    def __init__(self, nb_node, dim_feature, A, B):
        super(DoubleDGC, self).__init__()
        self.N = nb_node
        self.F = dim_feature

        self.w1 = nn.Parameter(nn.init.xavier_normal_(torch.empty(2, self.F, self.N)))
        self.w2 = nn.Parameter(nn.init.xavier_normal_(torch.empty(2, self.N, self.N)))
        self.bias = nn.Parameter(torch.zeros(2, self.N, 1))

        self.convert = nn.Linear(self.F, 2)

        self.A = nn.Parameter(A, requires_grad=False)
        self.B = nn.Parameter(B, requires_grad=False)

    def forward(self, x, state):

        demand = self.convert(x)
        demand = torch.tanh(demand)*0.5

        x1 = torch.matmul(x.transpose(-1, -2), self.A*self.w2[0])
        mask1 = torch.matmul(x1.transpose(-1, -2), self.w1[0]) + self.bias[0]
        mask1 = mask1 + -10e15 * (1.0 - self.A)
        mask1 = torch.softmax(mask1, -1)

        x2 = torch.matmul(x.transpose(-1, -2), self.B*self.w2[1])
        mask2 = torch.matmul(x2.transpose(-1, -2), self.w1[1]) + self.bias[1]
        mask2 = mask2 + -10e15 * (1.0 - self.B)
        mask2 = torch.softmax(mask2, -1)

        v = torch.matmul(mask1, state[...,:1])
        q = torch.matmul(mask2, state[...,1:])

        output = torch.cat((v,q), -1) + demand

        return output, mask1, mask2, demand[...,-1]


class DGCNcell(nn.Module):
    def __init__(self, nb_node, dim_feature, A, B, return_interpret=False):
        super(DGCNcell, self).__init__()
        self.N = nb_node
        self.F = dim_feature
        self.A = A
        self.B = B
        self.dim = self.F//2+2

        self.interpret = return_interpret

        self.dgc_r = LocalGC(self.N, self.dim, self.A)
        self.lin_r = nn.Linear(self.dim, self.F//2)

        self.dgc_u = LocalGC(self.N, self.dim, self.A)
        self.lin_u = nn.Linear(self.dim, self.F//2)

        self.dgc_c = LocalGC(self.N, self.F, self.A)
        self.lin_c = nn.Linear(self.F, self.F//2)

        #self.core = DynamicGC(self.N, self.F, self.A)
        self.core = DoubleDGC(self.N, self.F, self.A, self.B)

        #self.lin_out = nn.Linear(self.F, 1)

        self.lin_in = nn.Linear(2, self.F//2)


    def forward(self, input, state):
        #print(state.size())
        x = self.lin_in(input)
        gru_input = torch.cat([x, state], -1)

        p, mask1, mask2, demand = self.core(gru_input, input)
        feature_ru = torch.cat([p, state], -1)

        r = self.dgc_r(feature_ru)
        r = self.lin_r(r)
        r = torch.sigmoid(r)

        u = self.dgc_u(feature_ru)
        u = self.lin_u(u)
        u = torch.sigmoid(u)

        s = r*state
        feature_c = torch.cat([x, s], -1)
        c = self.dgc_c(feature_c)
        c = self.lin_c(c)
        c = torch.tanh(c)

        H = u*state + (1-u)*c
        #print(H.size())
        if self.interpret:
            return  p, H, mask1, mask2, demand
        return p, H


class DGCNEncoder(nn.Module):
    def __init__(self, nb_node, dim_feature, A, B):
        super(DGCNEncoder, self).__init__()
        self.N = nb_node
        self.F = dim_feature
        self.A = A
        self.B = B

        self.encodercell = DGCNcell(self.N, self.F, self.A, self.B)

        self.init_state = nn.Parameter(torch.zeros(self.N, self.F//2), requires_grad=False)
      
    def forward(self, x):
        x = x[:,:20] # Make sure only the first 20 time steps are used
        for i in range(x.size(1)):
            if i==0:
                _, state = self.encodercell(x[:,i], self.init_state.repeat(x.size(0), 1, 1))
            else:
                _, state = self.encodercell(x[:,i], state)

        return state # (B, n_nodes, F//2)


#########################################################
##        GraphEncoder for micro-traffic data          ##
#########################################################

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class MLP(nn.Module):
    def __init__(self, hidden_size, out_features=None):
        super(MLP, self).__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size, out_features)
        self.layer_norm = LayerNorm(out_features)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = torch.nn.functional.relu(hidden_states)
        return hidden_states
    

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, attention_size,
                 nb_heads,
                 aggregation_mode="cat",
                 use_decay=False,
                 scale=True):
        super(MultiHeadSelfAttention, self).__init__()
        self.nb_heads = nb_heads
        self.attention_size = attention_size
        self.aggregation_mode = aggregation_mode
        self.in_channels = in_channels
        self.use_decay = use_decay
        self.q_layer = nn.Linear(self.in_channels, self.attention_size*self.nb_heads)
        self.k_layer = nn.Linear(self.in_channels, self.attention_size*self.nb_heads)
        self.v_layer = nn.Linear(self.in_channels, self.attention_size*self.nb_heads)
        self.attention_decay = nn.Parameter(torch.ones(1) * 0.5, requires_grad=False)
        if scale:
            self.d = math.sqrt(self.attention_size)
        else:
            self.d = 1
                        
    def transpose_attention(self, x):
        z = x.size()[:-1] + (self.nb_heads,self.attention_size)
        x = x.view(*z)
        return x.permute(0, 2, 1, 3) #(B, nb_heads, N, attention_size)

    def forward(self, x, adj):
        q_ini = self.q_layer(x)
        k_ini = self.k_layer(x)
        v_ini = self.v_layer(x)
        
        q = self.transpose_attention(q_ini)
        k = self.transpose_attention(k_ini)
        v = self.transpose_attention(v_ini)
        mask = adj.repeat(self.nb_heads,1,1,1).transpose(1,0)
        scores = torch.matmul(q/self.d, k.transpose(-1, -2))
        attention_weights = nn.Softmax(dim=-1)(scores-1e5*(1-mask))
        
        if self.use_decay:
            v = torch.cat([v[:, 0:1, 0:1, :] * self.attention_decay, v[:, 0:1, 1:, :]], dim=2)
                
        c = torch.matmul(attention_weights, v)
        c = c.permute(0, 2, 1, 3).contiguous()
        
        if self.aggregation_mode == "cat":
            new_shape = c.size()[:-2] +(self.nb_heads*self.attention_size,)
            out = c.view(*new_shape)
        elif self.aggregation_mode == "max":
            out = nn.MaxPool2d((self.nb_heads, 1))(c)
            out = out.squeeze(-2)
        return out


class SubGraph(nn.Module):
    def __init__(self, c_in, hidden_size, length, depth):
        super(SubGraph, self).__init__()
        self.hidden_size = hidden_size
        self.c_in = c_in
        self.depth = depth
        self.fc = MLP(c_in, hidden_size)
        self.fc2 = MLP(hidden_size)

        self.layers = nn.ModuleList([MultiHeadSelfAttention(hidden_size, hidden_size //2, 2) for _ in range(depth)])
        self.layers_2 = nn.ModuleList([LayerNorm(hidden_size) for _ in range(depth)])
        
        adj_ = torch.ones(length, length).float()
        self.adj = nn.Parameter(adj_, requires_grad = False)

    def forward(self, x):
        h = x.reshape(-1, x.size(-2), x.size(-1))
        h = self.fc(h)
        h = self.fc2(h)
        
        A = self.adj.unsqueeze(0).repeat(h.size(0), 1, 1)

        for layer_index, layer in enumerate(self.layers):
            temp = h
            h = layer(h, A)
            h = F.relu(h)
            h = h + temp
            h = self.layers_2[layer_index](h)
        h = h.reshape(x.size(0), x.size(1), x.size(2), self.hidden_size)
        return torch.max(h, dim=2)[0] #(B, n_agents, hidden_size)


########################################################
##   LSTM Encoder as baseline for traffic prediction  ##
########################################################

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, single_output=False):
        super(LSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.single_output = single_output
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x, thred=None):
        # Reshape input from (B, seq_len, num_node, dim_feature) to (B, seq_len, num_node*dim_feature)
        x = x[:,:20] # Make sure only the first 20 time steps are used
        x = x.view(x.size(0), x.size(1), -1)

        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate through LSTM
        output, (hidden, cell) = self.lstm(x, (h0, c0))
        if self.single_output:
            # Only return the hidden state of the last LSTM layer
            return hidden[-1].view(x.size(0), 193, -1)
        else:
            # Return all
            return output, (hidden, cell)
        

########################################################
##   GRU Encoder as baseline for traffic prediction   ##
########################################################

class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, single_output=False):
        super(GRUEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.single_output = single_output
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x, thred=None):
        # Reshape input from (B, seq_len, num_node, dim_feature) to (B, seq_len, num_node*dim_feature)
        x = x[:,:20] # Make sure only the first 20 time steps are used
        x = x.view(x.size(0), x.size(1), -1)

        # Initialize hidden states for the encoder
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Encoder: Process input sequence
        output, hidden = self.gru(x, h0)

        if self.single_output:
            # Only return the hidden state of the last GRU layer
            return hidden[-1].view(x.size(0), 193, -1)
        else:
            # Return all
            return output, hidden