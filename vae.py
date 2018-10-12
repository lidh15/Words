# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 20:30:01 2018

@author: Li Denghao
"""
from random import shuffle

import numpy as np
import torch
import xlrd
from torch.utils.data import DataLoader

from models import EEGvae2d
from datasets import EEGvaeset
# from Cluster import Cluster

CUDA = 1
LR = 0.001
GPU = 7
EPOCHS = 5
PARA = 32

# Parameters
params = {'batch_size': PARA,
          'shuffle': True,
          'num_workers': 8}
workbook = xlrd.open_workbook('./ID.xlsx')
patient = [str(x)[:3] for x in workbook.sheet_by_name('Sheet1').col_values(0)]
control = [str(x)[:3] for x in workbook.sheet_by_name('Sheet2').col_values(0)]
shuffle(patient)
shuffle(control)

def adjust_learning_rate(optimizer, epoch, initial_lr=LR):
    '''Sets the learning rate to the initial LR decayed by 10 every 10 epochs'''
    lr = initial_lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def loss_func(output, x, mu, logvar):
    mse = torch.nn.MSELoss(reduction='sum')# size_average=False
    mse_loss = mse(output, x)
    z_loss = -0.5*(1+logvar.sum()-mu.pow(2).sum()-logvar.exp().sum())
    loss = mse_loss + z_loss
    return loss, mse_loss, z_loss


# Datasets
training_sett = EEGvaeset(test_IDs=patient+control, max_sample_num=200000)
training_setf = EEGvaeset(test_IDs=patient+control, max_sample_num=200000,freq=True)
t_len = len(training_sett)
# Generators
training_generatort = DataLoader(training_sett, **params)
training_generatorf = DataLoader(training_setf, **params)
# Network
if CUDA:
    eegnett = EEGvae2d(GPU).cuda(GPU)
    eegnetf = EEGvae2d(GPU, freq=True).cuda(GPU)
else:
    eegnett = EEGvae2d()
    eegnetf = EEGvae2d(freq=True)

optimizert = torch.optim.Adam(eegnett.parameters(), lr=LR)
optimizerf = torch.optim.Adam(eegnetf.parameters(), lr=LR)
# Training process
for epoch in range(EPOCHS):
    adjust_learning_rate(optimizert, epoch)
    adjust_learning_rate(optimizerf, epoch)
    # if epoch % 5 == 4:
    if True:
        total_mse_losst = 0
        total_z_losst = 0
        for local_batcht, _ in training_generatort:
            if CUDA:
                local_batcht = local_batcht.cuda(GPU)
            outputt, mut, logvart = eegnett(local_batcht)
            losst, mset, zt = loss_func(outputt, local_batcht, mut, logvart)
            total_mse_losst += mset.data.item()/t_len
            total_z_losst += zt.data.item()/t_len
            optimizert.zero_grad() # clear gradients for this training step
            losst.backward() # backpropagation, compute gradients
            optimizert.step() # apply gradients
        
        total_mse_lossf = 0
        total_z_lossf = 0
        for local_batchf, _ in training_generatorf:
            if CUDA:
                local_batchf = local_batchf.cuda(GPU)
            outputf, muf, logvarf = eegnetf(local_batchf)
            lossf, msef, zf = loss_func(outputf, local_batchf, muf, logvarf)
            total_mse_lossf += msef.data.item()/t_len
            total_z_lossf += zf.data.item()/t_len
            optimizerf.zero_grad() # clear gradients for this training step
            lossf.backward() # backpropagation, compute gradients
            optimizerf.step() # apply gradients
        # print('training done')

        # # Validation process
        # val_mse_loss = 0
        # val_z_loss = 0
        # for val_batch, _ in validation_generator:
        #     if CUDA:
        #         val_batch = val_batch.cuda(GPU)
        #     output, mu, logvar = eegnet(val_batch)
        #     loss, mse, z = loss_func(output, val_batch, mu, logvar)
        #     val_mse_loss += mse.data.item()/v_len
        #     val_z_loss += z.data.item()/v_len
        
        # print('epoch %3d | train loss: %2.3f | val loss: %2.3f | train z loss: %2.3f | val z loss: %2.3f' \
        #     % (epoch, total_mse_loss, val_mse_loss, total_z_loss, val_z_loss))
        print('epoch %3d | ttrain loss: %2.3f | ftrain loss: %2.3f | ttrain z loss: %2.3f | ftrain z loss: %2.3f' \
            % (epoch, total_mse_losst, total_mse_lossf, total_z_losst, total_z_lossf))
    if epoch == EPOCHS-1:
        data = np.zeros((t_len,4))
        i = 0
        eegnett.eval()
        for local_batcht, label in training_generatort:
            if CUDA:
                local_batcht = local_batcht.cuda(GPU)
            _, mut, _ = eegnett(local_batcht)
            if CUDA:
                mut = mut.cpu()
            data[i*PARA:i*PARA+PARA,:2] = mut.detach().numpy()
            data[i*PARA:i*PARA+PARA,-2] = label['label'].squeeze(-1)
            data[i*PARA:i*PARA+PARA,-1] = label['test ID'].squeeze(-1)
            i += 1
        np.save('tout.npy', data)
        # Cluster(data, patient+control)
        torch.save(eegnett.state_dict(), 'vaet.pkl')
        
        data = np.zeros((t_len,4))
        i = 0
        eegnetf.eval()
        for local_batchf, label in training_generatorf:
            if CUDA:
                local_batchf = local_batchf.cuda(GPU)
            _, muf, _ = eegnetf(local_batchf)
            if CUDA:
                muf = muf.cpu()
            data[i*PARA:i*PARA+PARA,:2] = muf.detach().numpy()
            data[i*PARA:i*PARA+PARA,-2] = label['label'].squeeze(-1)
            data[i*PARA:i*PARA+PARA,-1] = label['test ID'].squeeze(-1)
            i += 1
        np.save('fout.npy', data)
        # Cluster(data, patient+control, freq=True)
        torch.save(eegnetf.state_dict(), 'vaef.pkl')
    
    # else:
    #     for local_batch in training_generator:
    #         if CUDA:
    #             local_batch = local_batch.cuda(GPU)
    #         output, mu, logvar = eegnet(local_batch)
    #         loss = loss_func(output, local_batch, mu, logvar)
    #         optimizer.zero_grad() # clear gradients for this training step
    #         loss.backward() # backpropagation, compute gradients
    #         optimizer.step() # apply gradients
