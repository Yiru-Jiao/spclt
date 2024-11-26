import numpy as np
import torch
from numpy.linalg import matrix_power
import pickle

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

    output = matrix_power(both, k)
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
    
    downstream = matrix_power(base, k1)
    upstream = matrix_power(base.T, k2)
    both = np.logical_or(upstream, downstream)

    both[both > 0] = 1.
    
    return torch.Tensor(both)

def pure_adj(k):
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

    output = matrix_power(both, k)
    output[output > 0] = 1.
    
    return output

def get_directed_connection():
    Av = pure_adj(3)
    Aq = pure_adj(8)
    return Av, Aq

def get_roads():
    with open('./datasets/segments.pkl', 'rb') as f:
        roads = pickle.load(f)

    lengths = [len(s[0]) for s in list(roads.values())]
    indicators = np.zeros(sum(lengths))
    start = 0
    for i in range(len(roads)):
        indicators[start:start+lengths[i]] = i+1
        start = start+lengths[i]

    return roads, lengths, indicators