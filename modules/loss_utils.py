'''
This module contains utility functions and classes for computing the contrastive loss and regularizers.
Functions and classes are adjusted from the original implementation in the SoftCLT repository, TopoAE repository, and GGAE repository.
SoftCLT https://github.com/seunghan96/softclt
TopoAE https://github.com/BorgwardtLab/topological-autoencoders
GGAE https://github.com/JungbinLim/GGAE-public
'''

import torch
import random


#####################################
## functions for contrastive_loss  ##
#####################################

def take_per_row(A, indx, num_elem):
    """
    Selects a specified number of elements per row from a 2D tensor.
    """
    all_indx = indx[:, None] + torch.arange(num_elem, device=A.device)
    return A[torch.arange(all_indx.shape[0])[:, None], all_indx]


def dup_matrix(mat):
    """
    Duplicates a matrix by appending its transpose to itself.
    """
    mat0 = torch.tril(mat, diagonal=-1)[:, :-1]   
    mat0 += torch.triu(mat, diagonal=1)[:, 1:]
    mat1 = torch.cat([mat0, mat], dim=1)
    mat2 = torch.cat([mat, mat0], dim=1)
    return mat1, mat2


def mask_and_crop(encoder, x, temporal_unit=0):
    """
    Crops the masked input tensor and computes the output of the encoder.
    Note: input mask is embedded in the encoder.
    """
    ts_l = x.size(1)
    if encoder.training:
        crop_l = random.randint(2**(temporal_unit+1), ts_l)
        crop_left = random.randint(0, ts_l-crop_l)
        crop_right = crop_left + crop_l
        crop_eleft = random.randint(0, crop_left)
        crop_eright = random.randint(crop_right, ts_l)
        crop_offset = torch.randint(-crop_eleft, ts_l-crop_eright+1, size=(x.size(0),), device=x.device)

        out1 = encoder(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
        out1 = out1[:, -crop_l:]
        
        out2 = encoder(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))
        out2 = out2[:, :crop_l]
    else:
        out1 = encoder(x)
        out2 = out1

    return out1, out2


def timelag_sigmoid(z1, sigma=1):
    """
    Computes a time-lagged sigmoid matrix based on the input tensor.
    """
    T = z1.size(1)
    dist = torch.arange(T, device=z1.device).float()
    dist = torch.abs(dist[:, None] - dist[None, :])
    matrix = 2 / (1 + torch.exp(dist*sigma))
    matrix = torch.where(matrix < 1e-6, torch.zeros_like(matrix), matrix)  # set very small values to 0
    return matrix


###############################################################
## functions and classes for topology preserving regularizer ##
###############################################################

def tensor_norm_ts(TS):
    """
    Normalize a time series tensor per feature.
    """
    TS_max, _ = torch.max(TS, dim=0, keepdim=True)
    TS_min, _ = torch.min(TS, dim=0, keepdim=True)
    TS_range = TS_max - TS_min
    TS = (TS - TS_min) / torch.where(TS_range < 1e-6, torch.ones_like(TS_range), TS_range)
    return TS


def topo_euclidean_distance_matrix(x, p=2):
    """
    Computes the pairwise Euclidean distance matrix between the rows of a 2D tensor.
    """
    x = tensor_norm_ts(x)
    x_flat = x.view(x.size(0), -1)
    distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=p)
    return distances


class UnionFind:
    '''
    An implementation of a Union--Find class. The class performs path
    compression by default. It uses integers for storing one disjoint
    set, assuming that vertices are zero-indexed.
    '''

    def __init__(self, n_vertices):
        '''
        Initializes an empty Union--Find data structure for a given
        number of vertices.
        '''
        self._parent = torch.arange(n_vertices, dtype=torch.long)

    def find(self, u):
        '''
        Finds and returns the parent of u with respect to the hierarchy.
        '''
        if self._parent[u] == u:
            return u
        else:
            # Perform path collapse operation
            self._parent[u] = self.find(self._parent[u])
            return self._parent[u]

    def merge(self, u, v):
        '''
        Merges vertex u into the component of vertex v. Note the
        asymmetry of this operation.
        '''
        if u != v:
            self._parent[self.find(u)] = self.find(v)

    def roots(self):
        '''
        Generator expression for returning roots, i.e. components that
        are their own parents.
        '''
        for vertex, parent in enumerate(self._parent):
            if vertex == parent:
                yield vertex


class PersistentHomologyCalculation:
    def __call__(self, matrix):
        """
        Computes the persistence pairs of the given distance matrix.
        """
        n_vertices = matrix.shape[0]
        uf = UnionFind(n_vertices)

        triu_indices = torch.triu_indices(n_vertices, n_vertices)
        edge_weights = matrix[triu_indices[0], triu_indices[1]]
        edge_indices = torch.argsort(edge_weights, stable=True)

        # 1st dimension: 'source' vertex index of edge
        # 2nd dimension: 'target' vertex index of edge
        persistence_pairs = []

        for edge_index, edge_weight in zip(edge_indices, edge_weights[edge_indices]):

            u = triu_indices[0][edge_index]
            v = triu_indices[1][edge_index]

            younger_component = uf.find(u)
            older_component = uf.find(v)

            # Not an edge of the MST, so skip it
            if younger_component == older_component:
                continue
            elif younger_component > older_component:
                uf.merge(v, u)
            else:
                uf.merge(u, v)

            if u < v:
                persistence_pairs.append((u, v))
            else:
                persistence_pairs.append((v, u))

        return torch.tensor(persistence_pairs)


class TopologicalSignatureDistance(torch.nn.Module):
    def __init__(self, match_edges='symmetric'):
        """
        Topological signature computation.
        """
        super().__init__()
        self.match_edges = match_edges

        self.signature_calculator = PersistentHomologyCalculation()

    def _get_pairings(self, distances):
        """
        Returns pairings for 0-dimensional features (ignore cycles).
        """
        return self.signature_calculator(distances)

    def _select_distances_from_pairs(self, distance_matrix, pairs):
        """
        Selects distances from the distance matrix based on the given pairs.
        """
        selected_distances = distance_matrix[(pairs[:, 0], pairs[:, 1])]

        return selected_distances

    @staticmethod
    def sig_error(signature1, signature2):
        """
        Compute distance between two topological signatures.
        """
        return ((signature1 - signature2)**2).sum(dim=-1)

    # pylint: disable=W0221
    def forward(self, distances1, distances2):
        """
        Return topological distance of two pairwise distance matrices.
        """
        pairs1 = self._get_pairings(distances1)
        pairs2 = self._get_pairings(distances2)

        if self.match_edges is None:
            sig1 = self._select_distances_from_pairs(distances1, pairs1)
            sig2 = self._select_distances_from_pairs(distances2, pairs2)
            distance = self.sig_error(sig1, sig2)

        elif self.match_edges == 'symmetric':
            sig1 = self._select_distances_from_pairs(distances1, pairs1)
            sig2 = self._select_distances_from_pairs(distances2, pairs2)
            # Selected pairs of 1 on distances of 2 and vice versa
            sig1_2 = self._select_distances_from_pairs(distances2, pairs1)
            sig2_1 = self._select_distances_from_pairs(distances1, pairs2)

            distance1_2 = self.sig_error(sig1, sig1_2)
            distance2_1 = self.sig_error(sig2, sig2_1)

            distance = distance1_2 + distance2_1

        return distance


###############################################################
## functions and classes for geometry preserving regularizer ##
###############################################################

def get_laplacian(X, bandwidth=50): # bandwidth tuning should increase exponentially like bw**2
    """
    Calculate the Normalized Graph Laplacian for a given set of data points.
    """
    X = tensor_norm_ts(X)

    if X.ndim == 3:
        B, N, _ = X.shape
    elif X.ndim == 4:
        X = X.contiguous().view(X.size(0), X.size(2), -1) # use N as the number of nodes for MacroTraffic
        B, N, _ = X.shape                                 # use N as the number of timesteps for MicroTraffic
    c = 1/4

    dist_XX = torch.cdist(X, X, p=2)
    K = torch.exp(-dist_XX**2 / bandwidth)
    d_i = K.sum(dim=1)
    D_inv = torch.diag_embed(1/d_i)
    K_tilde = D_inv @ K @ D_inv
    d_i_tilde = K_tilde.sum(dim=1)
    D_tilde_inv = torch.diag_embed(1/d_i_tilde)
    I = torch.diag_embed(torch.ones(B, N, device=X.device))
    L = (D_tilde_inv@K_tilde - I)/(c*bandwidth)

    return L # (B, N, N) or (B, T, T)


def get_JGinvJT(L, Y, k_chunk=512):
    """
    Memory-efficient computation of  H̃ = ½·(L·Y_kY_kᵀ - Y·(LY)ᵀ - (LY)·Yᵀ)

    Shapes
    ------
    L : (B, N, N)
    Y : (B, N, n)
    Returns  (B, N, n, n)
    """

    B, N, n = Y.shape
    device  = Y.device

    # Pre-compute LY once
    LY = torch.matmul(L, Y)                      # (B,N,n)
    H_tilde = torch.zeros(B, N, n, n, dtype=Y.dtype, device=device)

    # First term: Σ_k L_{ik}·Y_kY_kᵀ  (done in k-sized slices)
    for k0 in range(0, N, k_chunk):
        k1 = min(k0 + k_chunk, N)
        L_blk = L[:, :, k0:k1]                   # (B,N,k)
        Y_blk = Y[:, k0:k1, :]                   # (B,k,n)

        # use distinct letters: i = node, p/q = latent dims
        # (B,N,k)·(B,k,n)·(B,k,n) → (B,N,n,n)
        term1_blk = torch.einsum(
            'bik,bkp,bkq->bipq',                # no optimise kw-arg – works on any version
            L_blk, Y_blk, Y_blk
        )
        H_tilde.add_(term1_blk)                 # in-place accumulate keeps memory flat

    # Second & third terms
    Y_col = Y.unsqueeze(-1)                     # (B,N,n,1)
    LY_row = LY.unsqueeze(-2)                   # (B,N,1,n)

    H_tilde.mul_(0.5)                           # apply ½ to term-1
    H_tilde.add_(
        -0.5 * (Y_col * LY_row +                # term-2
                Y_col.transpose(-1, -2) *       # term-3
                LY_row.transpose(-1, -2))
    )

    return H_tilde


def relaxed_distortion_measure_JGinvJT(H):
    """
    Calculate the relaxed distortion measure for a given JGinvJT matrix.
    """
    TrH = H.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    TrH2 = (H @ H).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    n = H.size(-1)

    distortion = (TrH2).mean() - 2 * (TrH).mean() + n
    return distortion / n


def iso_loss_stream(L, Y, node_chunk=128, k_chunk=512):
    """
    Computes the relaxed distortion measure without ever materialising H̃.

    Parameters
    ----------
    L : (B, N, N)   normalised Laplacian
    Y : (B, N, n)   latent vectors
    node_chunk :    how many 'i'-nodes to process at once  (memory–speed trade-off)
    k_chunk   :     how many 'k'-columns of L to load at once (as before)
    dtype      :    optional (float16 / bfloat16) down-cast for extra savings

    Returns
    -------
    distortion : scalar tensor on the same device
    """

    B, N, n = Y.shape
    device  = Y.device
    LY      = torch.matmul(L, Y)                     # (B,N,n)

    # running sums for TrH   and TrH²
    sum_tr  = torch.zeros(B, device=device, dtype=Y.dtype)
    sum_fro = torch.zeros(B, device=device, dtype=Y.dtype)

    for i0 in range(0, N, node_chunk):
        i1 = min(i0 + node_chunk, N)

        # ── term-1: Σ_k  L_{ik} · Y_k Y_kᵀ   (chunk in k)
        # we build it for node-block (B, i_block, n, n)
        H_blk = torch.zeros(B, i1 - i0, n, n, device=device, dtype=Y.dtype)

        for k0 in range(0, N, k_chunk):
            k1    = min(k0 + k_chunk, N)
            L_ik  = L[:, i0:i1, k0:k1]              # (B, i, k)
            Y_k   = Y[:, k0:k1, :]                  # (B, k, n)

            # einsum: (B,i,k) · (B,k,p) · (B,k,q)  →  (B,i,p,q)
            H_blk.add_(torch.einsum('bik,bkp,bkq->bipq', L_ik, Y_k, Y_k))

        # ── term-2, term-3 (cheap, no loop over k) ──────────────────
        Y_i  = Y[:, i0:i1, :]                       # (B,i,n)
        LY_i = LY[:, i0:i1, :]                      # (B,i,n)

        H_blk.mul_(0.5)
        H_blk.add_(
            -0.5 * (
                Y_i.unsqueeze(-1)  * LY_i.unsqueeze(-2) +
                LY_i.unsqueeze(-1) * Y_i.unsqueeze(-2)
            )
        )

        # ── accumulate statistics we really need ────────────────────
        tr  = H_blk.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)   # (B,i)
        fro = (H_blk * H_blk).sum((-1, -2))                        # (B,i)

        sum_tr  += tr.sum(dim=1)
        sum_fro += fro.sum(dim=1)

        # free the block ASAP
        del H_blk, tr, fro
        torch.cuda.empty_cache()            # keeps long runs flat

    # final distortion  =  E_i[TrH²] - 2·E_i[TrH] + n
    distortion = sum_fro.mean() / N  -  2 * (sum_tr.mean() / N) + n
    return distortion / n
