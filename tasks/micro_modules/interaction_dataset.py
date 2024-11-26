import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.transform import rescale

class InteractionDataset(Dataset):
    """
    filename: a list of files or one filename of the .npz file
    stage: {"train", "val", "test"}
    """
    def __init__(self, filenames, stage, para, mode, device):
        self.stage= stage
        self.para = para
        self.resolution = para['resolution']
        self.mode = mode
        self.device = device

        if stage == 'train':
            self.T = []
            self.M = []
            self.L = []
            self.N_agents = []
            self.N_splines = []
            self.Y = []
            self.Adj = []
            if mode=='lanescore':
                self.S = []
            for filename in filenames:
                data = np.load('./datasets/MicroTraffic/'+filename+'.npz', allow_pickle=True)

                self.T.append(data['trajectory'])
                self.M.append(data['maps'])
                self.L.append(data['lanefeature'])
                self.N_agents.append(data['nbagents'])
                self.N_splines.append(data['nbsplines'])
                self.Adj.append(data['adjacency'])
                self.Y.append(data['intention'])
                if mode=='lanescore':
                    self.S.append(data['lanescore'])
            self.T = np.concatenate(self.T, axis=0)
            self.M = np.concatenate(self.M, axis=0)
            self.L = np.concatenate(self.L, axis=0)
            self.N_agents = np.concatenate(self.N_agents, axis=0)
            self.N_splines = np.concatenate(self.N_splines, axis=0)
            self.Y = np.concatenate(self.Y, axis=0)
            if mode=='lanescore':
                self.S = np.concatenate(self.S, axis=0)
            self.Adj = np.concatenate(self.Adj, 0)
            
            data_mask = np.load('./datasets/MicroTraffic/mask_train.npz', allow_pickle=True)
            self.mask = data_mask['mask']
        else:
            data = np.load('./datasets/MicroTraffic/'+filenames[0]+'.npz', allow_pickle=True)
            self.T = data['trajectory']
            self.M = data['maps']
            self.L = data['lanefeature']
            self.N_agents = data['nbagents']
            self.N_splines = data['nbsplines']
            self.Adj = data['adjacency']

            if stage=='val':
                data_mask = np.load('./datasets/MicroTraffic/mask_val.npz', allow_pickle=True)
                self.Y = data['intention']
                if mode=='lanescore':
                    self.S = data['lanescore']
            if stage=='test':
                data_mask = np.load('./datasets/MicroTraffic/mask_test.npz', allow_pickle=True)
            self.mask = data_mask['mask']
        
    def __len__(self):
        return len(self.N_agents)
    
    def __getitem__(self, index):
        traj = torch.tensor(self.T[index]).float().to(self.device)
        splines = torch.tensor(self.M[index]).float().to(self.device)
        lanefeature = torch.tensor(self.L[index].toarray()).float().to(self.device)
        nb_agents = self.N_agents[index]
        nb_splines = self.N_splines[index]
        
        if self.mode=='densetnt':
            adjacency = np.zeros((81, 81))

            cross = np.zeros(81)
            cross[:nb_splines] = 1
            cross[55:nb_agents] = 1
            
            adjacency[:nb_splines][...,:nb_splines] = 1
            adjacency[55:55+nb_agents][...,55:55+nb_agents] = 1
            adjacency[:nb_splines][...,55:55+nb_agents] = 1
            adjacency[55:55+nb_agents][...,:nb_splines] = 1
            adj = torch.Tensor(adjacency).int().to(self.device)
            c_mask = torch.Tensor(cross).int().to(self.device)

            masker = self.mask[index].toarray()#.reshape((46,87,3))
            if self.resolution != 1:
                masker = rescale(masker, int(1/self.resolution))
            masker = torch.tensor(masker.copy()).float().to(self.device)
            
            if self.stage != "test":
                y = torch.tensor(self.Y[index]).float().to(self.device)
                return traj, splines, masker, lanefeature, adj, c_mask, y
            else:
                return traj, splines, masker, lanefeature, adj, c_mask
        
        if self.mode=='lanescore':
            a = self.Adj[index].toarray()
            af = a.copy()#+np.eye(55)
            af[af<0] = 0
            pad = np.zeros((55,55))
            pad[:nb_splines,:nb_splines]=np.eye(nb_splines)
            
            Af = np.linalg.matrix_power(af+pad+af.T, 4)
            Af[Af>0]=1
            
            A_f = torch.Tensor(Af).float().to(self.device)
            
            adjacency = np.zeros((81, 81))
            adjacency[:nb_splines][...,:nb_splines] = 1
            adjacency[55:55+nb_agents][...,55:55+nb_agents] = 1
            adjacency[:nb_splines][...,55:55+nb_agents] = 1 #optional
            adjacency[55:55+nb_agents][...,:nb_splines] = 1
            adj = torch.Tensor(adjacency).int().to(self.device)
            
            adjego = np.zeros((56, 56))
            adjego[:nb_splines][...,:nb_splines] = 1
            adjego[55,55] = 1
            adjego[55:56][...,:nb_splines] = 1
            A_r = torch.Tensor(adjego).int().to(self.device)
            
            c_mask = torch.Tensor(adjacency[:,0]).int().to(self.device)
            if self.stage!='test':
                ls = torch.tensor(self.S[index]).float().to(self.device)
            masker = self.mask[index].toarray()#.reshape(46, 87, 3)
            if self.resolution != 1:
                masker = rescale(masker, int(1/self.resolution))
            masker = torch.tensor(masker.copy()).float().to(self.device)

            if self.stage != "test":
                y = torch.tensor(self.Y[index]).float().to(self.device)
                return traj, splines, masker, lanefeature, adj, A_f, A_r, c_mask, y, ls
            else:
                return traj, splines, masker, lanefeature, adj, A_f, A_r, c_mask
            
        if self.mode=='testmodel':    
            a = self.Adj[index].toarray()
            af = a.copy()#+np.eye(55)
            al = a.copy()#+np.eye(55)
            af[af<0] = 0
            al[al>0] = 0
            al[al<0] = 1

            adjacency = np.zeros((81, 81))
            cross = np.zeros(81)
            cross[:nb_splines] = 1
            cross[55:nb_agents] = 1
            pad = np.zeros((55,55))
            pad[:nb_splines,:nb_splines]=np.eye(nb_splines)

            adjacency[:nb_splines][...,:nb_splines] = 1#np.eye(nb_splines)
            adjacency[55:55+nb_agents][...,55:55+nb_agents] = 1
            adjacency[:nb_splines][...,55:55+nb_agents] = 1
            adjacency[55:55+nb_agents][...,:nb_splines] = 1

            adj = torch.Tensor(adjacency).int().to(self.device)
            c_mask = torch.Tensor(cross).int().to(self.device)
            
            Af = af+pad#np.linalg.matrix_power(af+pad, 2)
            Al = al+pad
            Af[Af>0]=1
            Al[Al>0]=1
            
            A_f = torch.Tensor(Af).float().to(self.device)
            A_l = torch.Tensor(Al).float().to(self.device)
            
            masker = self.mask[index].toarray()#.reshape((46,87,3))
            if self.resolution != 1:
                masker = rescale(masker, int(1/self.resolution))
            masker = torch.tensor(masker.copy()).float().to(self.device)
            
            if self.stage != "test":
                y = torch.tensor(self.Y[index]).float().to(self.device)
                return traj, splines, masker, lanefeature, adj, A_f, A_l, c_mask, y
            else:
                return traj, splines, masker, lanefeature, adj, A_f, A_l, c_mask

class InteractionTraj(Dataset):
    def __init__(self, filename, stage, device):
        self.device = device
        data = np.load(filename, allow_pickle=True)
        self.D = data['traj']
        self.A = self.D[:,-1]
        self.Y = self.D[:,:-1]
        
        if stage == 'train':
            data = np.load('./datasets/MicroTraffic/train1.npz', allow_pickle=True)
        elif stage == 'test':
            data = np.load('./datasets/MicroTraffic/test.npz', allow_pickle=True)
        else:
            data = np.load('./datasets/MicroTraffic/val.npz', allow_pickle=True)
        self.T = data['trajectory'][:,1]
        
        assert len(self.A) == len(self.T)
        
    def __len__(self):
        return len(self.A)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.T[idx]).float().to(self.device)
        anchor = torch.tensor(self.A[idx]).float().to(self.device)
        y = torch.tensor((self.Y[idx]).reshape(-1)).float().to(self.device)
        
        return x, anchor, y
    
class InferenceTraj(Dataset):
    def __init__(self, d, device):
        self.D = d
        self.device = device
        
    def __len__(self):
        return len(self.D)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.D[idx]).float().to(self.device)
        return x
