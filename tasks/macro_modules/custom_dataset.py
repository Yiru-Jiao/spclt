import numpy as np
import torch
import zarr
from torch.utils.data import Dataset


class AMSdataset(Dataset):
    def __init__(self, years, para, stage='train', device='cuda:0'):
        self.para = para
        self.interval = para['time_invertal']
        self.Tout = para['horizon']
        self.Tin = para['observation']
        self.stage = stage
        self.device = device

        X = []
        print(f'preprocessing {stage} data...')
        for year in years:
            dt = zarr.open('./datasets/MacroTraffic/'+year+'.zarr')
            xf = get_data(dt, self.Tout, self.Tin, self.interval, self.stage)
            X.append(xf)

        X = np.concatenate(X, 0)
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        out = torch.Tensor(self.X[index]).float().to(self.device)
        return out

    
def get_data(dt, tout, tin, interval, stage):

    x = []
    V_morning = np.transpose(dt.speed_morning, (0,2,1))
    V_evening = np.transpose(dt.speed_evening, (0,2,1))
    Q_morning = np.transpose(dt.flow_morning, (0,2,1))
    Q_evening = np.transpose(dt.flow_evening, (0,2,1))
    if stage != 'test1':
        V_morning[V_morning>130] = 100.
        V_evening[V_evening>130] = 100.

    V_morning = V_morning/130.
    V_evening = V_evening/130.

    if stage != 'test1':
        Q_morning[Q_morning>3000] = 1000.
        Q_evening[Q_evening>3000] = 1000.

    Q_morning = Q_morning/3000.
    Q_evening = Q_evening/3000.

    # K_morning = Q_morning/V_morning
    # K_evening = Q_evening/V_evening

    T = tout + tin
    if stage == 'train':
        for i in range(0, 120-T, interval):
            status = np.stack([V_morning[:-35,i:i+T], Q_morning[:-35,i:i+T]], -1)
            x.append(status)

        for i in range(0, 210-T, interval):
            status = np.stack([V_evening[:-35,i:i+T], Q_evening[:-35,i:i+T]], -1)
            x.append(status)

        x = np.concatenate(x, 0)
        #np.random.shuffle(x)

    if stage == 'validation':
        for d in range(35):
            for i in range(0, 120-T, interval):
                status = np.stack([V_morning[-d-1,i:i+T], Q_morning[-d-1,i:i+T]], -1)
                x.append(status)
                
            for i in range(0, 210-T, interval):
                status = np.stack([V_evening[-d-1,i:i+T], Q_evening[-d-1,i:i+T]], -1)
                x.append(status)

        x = np.array(x)

    if stage == 'test':
        for d in range(len(V_morning)):
            for i in range(0, 120-T, interval):
                status = np.stack([V_morning[d,i:i+T], Q_morning[d,i:i+T]], -1)
                x.append(status)

        for d in range(len(V_evening)):        
            for i in range(0, 210-T, interval):
                status = np.stack([V_evening[d,i:i+T], Q_evening[d,i:i+T]], -1)
                x.append(status)

        x = np.array(x)

    return x

    



