import os
import sys
import math
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from macro_modules.custom_dataset import *


def train_one_epoch(model, optimizer, loss_func, training_loader, beta, count, T):
    model.train()
    running_loss = 0.
    last_loss = 0.
    
    for j, data in enumerate(training_loader):
        x = data
        optimizer.zero_grad()
        p = model(x, math.exp(beta*count))
        loss = loss_func(p, x[:,-T:])
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    count += 1
    last_loss = running_loss / (j + 1)
    running_loss = 0.
    return last_loss, count


def train_model(epochs, batch_size, trainset, model, optimizer, validation_loader, loss_func, scheduler, T, beta=-0.03, save_progress=None):
    training_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    EPOCHS = epochs
    count = 0
    def update_scheduler(scheduler, avg_loss, vloss_log, epoch_number, progress_bar):
        model.eval()
        running_vloss = 0.0
        for i, vdata in enumerate(validation_loader):
            x = vdata
            p = model(x, 0)
            v_loss = loss_func(p, x[:,-T:])
            running_vloss += float(v_loss)

        avg_vloss = running_vloss / (i + 1)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if epoch_number >= 10: # manual initial cooldown for 10 epochs
                scheduler.step(avg_vloss)
            progress_bar.set_postfix({'loss': avg_loss, 'vloss': avg_vloss})
            stop_early = False
            vloss_log[5+epoch_number] = avg_vloss
            if np.all(abs(np.diff(vloss_log[epoch_number:epoch_number+6]))<1e-4):
                print('Early stopping at epoch', epoch_number, 'with validation loss', avg_vloss)
                stop_early = True
        elif isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
            if epoch_number >= 10: # manual initial cooldown for 10 epochs
                scheduler.step()
            progress_bar.set_postfix({'loss': avg_loss, 'vloss': avg_vloss})
            stop_early = False
        else:
            print('Unknown scheduler type')
            stop_early = True
        model.train()
        return scheduler, vloss_log, stop_early

    vloss_log = np.zeros(EPOCHS+5)
    vloss_log[:5] = [100, 99, 98, 97, 96]
    progress_bar = tqdm(range(EPOCHS), desc='EPOCH', ascii=True, miniters=int(EPOCHS/5))
    for epoch_number in progress_bar:
        avg_loss, count = train_one_epoch(model, optimizer, loss_func, training_loader, beta, count, T)
        if save_progress is not None:
            if epoch_number % 6 == 0:
                torch.save(model.state_dict(), os.path.join(save_progress, f'ckpt_{int(epoch_number/6)}.pth'))

        scheduler, vloss_log, stop_early = update_scheduler(scheduler, avg_loss, vloss_log, epoch_number, progress_bar)
        if stop_early:
            break


def test_run_point(testset, model, BATCH_SIZE):
    model.eval()
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
    len_test = len(testloader)

    prediction = []

    # for x in tqdm(testloader, desc='TEST', total=len_test, ascii=True, miniters=int(len_test/5)):
    for x in testloader:
        p = model(x, 0)

        gc.disable()
        prediction.append(np.array(p.detach().to('cpu'))[...,-3:]) 
        gc.enable()

    prediction = np.concatenate(prediction, 0)

    return prediction


