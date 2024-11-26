'''
This file is used to load the macroscopic traffic data from the zarr files.
The functions are adapted from https://github.com/RomainLITUD/uncertainty-aware-traffic-speed-flow-demand-prediction
This file is used to load the microscopic traffic data from the npz files.
The functions are adapted from https://github.com/RomainLITUD/UQnet-arxiv
'''

import numpy as np
import zarr


def get_AMS_dataset(years, time_interval, horizon, observation, dataset_dir='datasets'):
    trainset = []
    valset = []
    testset = []
    for year in years:
        print(f"Loading macroscopic traffic data in {year} ...")
        dt = zarr.open(f'{dataset_dir}/MacroTraffic/'+year+'.zarr')
        xf = get_data(dt, horizon, observation, time_interval, 'train')
        trainset.append(xf)
        xf = get_data(dt, horizon, observation, time_interval, 'validation')
        valset.append(xf)
        xf = get_data(dt, horizon, observation, time_interval, 'test')
        testset.append(xf)
    trainset = np.concatenate(trainset, axis=0)
    valset = np.concatenate(valset, axis=0)
    testset = np.concatenate(testset, axis=0)
    return trainset, valset, testset

    
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


def get_INT_dataset(setname, filenames=['train1', 'train2', 'train3', 'train4'], dataset_dir='datasets'):
    if setname == 'train':
        Trajectories = []
        for filename in filenames:
            data = np.load(f'{dataset_dir}/MicroTraffic/'+filename+'.npz', allow_pickle=True)
            Trajectories.append(data['trajectory'])
        Trajectories = np.concatenate(Trajectories, axis=0)
    else:
        data = np.load(f'{dataset_dir}/MicroTraffic/'+setname+'.npz', allow_pickle=True)
        Trajectories = data['trajectory']
    return Trajectories

