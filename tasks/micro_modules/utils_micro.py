import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from numpy.lib.stride_tricks import as_strided


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def mask_softmax(x, test=None):

    if test == None:
        return F.softmax(x, dim=-1)
    else:
        shape = x.shape
        if test.dim() == 1:
            test = torch.repeat_interleave(
                test, repeats=shape[1], dim=0)
        else:
            test = test.reshape(-1)
        x = x.reshape(-1, shape[-1])
        for i, j in enumerate(x):
            j[int(test[i]):] = -1e5
        return F.softmax(x.reshape(shape), dim=-1)
    
def Entropy(x, resolution, epsilon=1e-6):
    x = np.reshape(x,(len(x),-1))
    x = x/np.sum(x,axis=-1)[:,np.newaxis]/resolution/resolution
    y = np.where(x>epsilon, -x*np.log(x)*resolution*resolution, 0)
    return np.sum(y,axis=-1)

def KLDivergence(x,y, resolution, epsilon=1e-5):
    assert x.shape == y.shape
    x = np.reshape(x,(len(x),-1))
    y = np.reshape(y,(len(y),-1))
    z = np.where(((y>epsilon)&(x>epsilon)), x*np.log(x/y)*resolution*resolution, 0)
    return np.sum(z,axis=-1)

def ComputeUQ(H, resolution, epsilon=1e-8):
    Ht = np.transpose(H, (2,3,0,1))
    Ht = (Ht/np.sum(Ht, axis=(0,1))).transpose((2,3,0,1))/resolution/resolution
    H_avr = np.mean(Ht, 0)
    #aleatoric = Entropy(H_avr, resolution, epsilon)
    aleatoric = np.zeros((len(H), H.shape[1]))
    for i in range(len(H)):
        aleatoric[i] = Entropy(Ht[i],resolution, epsilon)
    aleatoric = np.mean(aleatoric,0)
    
    epistemic = np.zeros((len(H), len(aleatoric)))
    for i in range(len(H)):
        epistemic[i] = KLDivergence(Ht[i], H_avr, resolution, epsilon)
    epistemic = np.mean(epistemic,0)
    return H_avr, aleatoric, epistemic

def inference_model(models, dataset, para, batch=8):
    Ua = []
    Ue = []
    Yp = []
    traj_data = np.load('./datasets/MicroTraffic/test.npz', allow_pickle=True)
    Y = traj_data['intention']
        
    nb = len(dataset)
    cut = list(range(0, nb, 400*batch)) + [nb]
    
    for i in tqdm(range(len(cut)-1), desc='TEST', ascii=True, mininterval=int(len(cut)/5)):
        ind = list(range(cut[i], cut[i+1]))
        testset = torch.utils.data.Subset(dataset, ind)
        loader = DataLoader(testset, batch_size=batch, shuffle=False)
        
        Hp = []
        Lp = []
        for model in models: # the original study trained 6 models as an ensemble, here we only use 1
            Hi = []
            Li = []
            for k, data in enumerate(loader):
                traj, splines, masker, lanefeature, adj, af, ar, c_mask = data
                lsp, heatmap = model(traj, splines, masker, lanefeature, adj, af, ar, c_mask)
                lsp = lsp.cpu().detach().numpy()
                heatmap = heatmap.cpu().detach().numpy()
                Hi.append(heatmap.reshape(-1, heatmap.shape[-2], heatmap.shape[-1]))
                Li.append(lsp.reshape(-1, lsp.shape[-1]))
            Hi = np.concatenate(Hi, 0)
            Li = np.concatenate(Li, 0)
            Hp.append(Hi)
            Lp.append(Li)
        Hp = np.stack(Hp, 0)
        Lp = np.stack(Lp, 0)
        hm, ua, ue = ComputeUQ(Hp, para['resolution'], epsilon=5e-4)
        # print(hm.shape)
        yp = ModalSampling(hm, para, r=3, k=10)
        Ua.append(ua)
        Ue.append(ue)
        Yp.append(yp)
            
    Ua = np.concatenate(Ua, 0) # (N,)
    Ue = np.concatenate(Ue, 0) # (N,)
    Yp = np.concatenate(Yp, 0) # (N,k,2)
    
    return Yp, Ua, Ue, Y

def pool2d_np(A, kernel_size, stride=1, padding=0):

    A = np.pad(A, padding, mode='constant')
    output_shape = ((A.shape[0] - kernel_size) // stride + 1,
                    (A.shape[1] - kernel_size) // stride + 1)
    
    shape_w = (output_shape[0], output_shape[1], kernel_size, kernel_size)
    strides_w = (stride*A.strides[0], stride*A.strides[1], A.strides[0], A.strides[1])
    
    A_w = as_strided(A, shape_w, strides_w)

    return A_w.max(axis=(2, 3))

def ModalSampling(H, paralist, r=2, k=10):
    # Hp = (N, H, W)
    dx, dy = paralist['resolution'], paralist['resolution']
    xmax, ymax = paralist['xmax'], paralist['ymax']
    ymin = paralist['ymin']
    Y = np.zeros((len(H), k, 2))
    for i in range(len(H)):
        # print(i, end='\r')
        Hp = H[i].copy()
        y = np.zeros((k,2))
        xc, yc = np.unravel_index(Hp.argmax(), Hp.shape)
        xc=xc+r
        yc=yc+r
        pred = [-xmax+xc*dx+dx/2, ymin+yc*dy+dy/2]
        y[0] = np.array(pred)
        Hp[xc-r:xc+r+1,yc-r:yc+r+1] = 0.
        for j in range(1,k):
            Hr = pool2d_np(Hp, kernel_size=2*r+1, stride=1, padding=r)
            xc, yc = np.unravel_index(Hr.argmax(), Hr.shape)
            xc=xc+r
            yc=yc+r
            pred = [-xmax+xc*dx+dx/2, ymin+yc*dy+dy/2]
            y[j] = np.array(pred)
            Hp[xc-r:xc+r+1,yc-r:yc+r+1] = 0.
        Y[i] = y
    return Y

def ComputeError(Yp, Y, r_list=[2], sh=6):
    assert sh <= Yp.shape[1]
    # Yp = [N,k,2], Y = [N,2] # k is the number of heatmap samples for each position to be predicted, k=10
    # Ua = [N], Ue = [N]
    E = np.abs(Yp.transpose((1,0,2))-Y) #(k,N,2)
    DE = np.sqrt(E[:sh,:,0]**2+E[:sh,:,1]**2) #(k,N)
    # print("minFDE:", np.mean(FDE),"m")
    # print("minMR:", np.mean(MR)*100,"%")
    minFDE = np.min(DE, axis=0) #(N,)
    MR_list = []
    for r in r_list:
        minMR = np.where(minFDE>r, np.ones_like(minFDE), np.zeros_like(minFDE))
        MR_list.append(minMR)

    return np.mean(minFDE), [np.mean(MR)*100 for MR in MR_list] # MR (%) FDE (m)