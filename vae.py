# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 20:30:01 2018

@author: Li Denghao
"""
# from random import shuffle

import numpy as np
import torch
# import xlrd
from torch.utils.data import DataLoader

from models import EEGvae
from datasets import EEG500ms


CUDA = 1
LR = 0.001
GPU = 7
EPOCHS = 5
PARA = 32

# Parameters
params = {'batch_size': PARA,
          'shuffle': True,
          'num_workers': 8}
with open('groups.txt') as f:
    validation = [line[:-1].split(' ') for line in f.readlines()]
training = ['%03d'%(x+1) for x in range(50)]


def adjust_learning_rate(optimizer, epoch, initial_lr=LR):
    '''Sets the learning rate to the initial LR decayed by 10 every 10 epochs'''
    lr = initial_lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def loss_func(output, x, mu, logvar):
    mse = torch.nn.MSELoss(reduction='sum') # size_average=False
    mse_loss = mse(output, x)
    z_loss = -0.5*(1+logvar.sum()-mu.pow(2).sum()-logvar.exp().sum())
    loss = mse_loss + z_loss
    return loss, mse_loss, z_loss


# Datasets
training_set = EEG500ms(test_IDs=training, max_sample_num=200000)
t_len = len(training_set)
# Generators
training_generator = DataLoader(training_set, **params)
# Network
if CUDA:
    eegnet = EEGvae(GPU).cuda(GPU)
else:
    eegnet = EEGvae()

optimizer = torch.optim.Adam(eegnet.parameters(), lr=LR)
# Training process
for epoch in range(EPOCHS):
    adjust_learning_rate(optimizer, epoch)
    # if epoch % 5 == 4:
    if True:
        total_mse_loss = 0
        total_z_loss = 0
        for local_batch, target in training_generator:
            if CUDA:
                local_batch = local_batch.cuda(GPU)
            output, mu, logvar = eegnet(local_batch)
            loss, mse, z = loss_func(output, target, mu, logvar)
            total_mse_loss += mse.data.item()/t_len
            total_z_loss += z.data.item()/t_len
            optimizer.zero_grad() # clear gradients for this training step
            loss.backward() # backpropagation, compute gradients
            optimizer.step() # apply gradient
        
        # print('training done')

        # # Validation process
        # val_mse_loss = 0
        # val_z_loss = 0
        # for val_batch, target in validation_generator:
        #     if CUDA:
        #         val_batch = val_batch.cuda(GPU)
        #     output, mu, logvar = eegnet(val_batch)
        #     loss, mse, z = loss_func(output, target, mu, logvar)
        #     val_mse_loss += mse.data.item()/v_len
        #     val_z_loss += z.data.item()/v_len
        
        # print('epoch %3d | train loss: %2.3f | val loss: %2.3f | train z loss: %2.3f | val z loss: %2.3f' \
        #     % (epoch, total_mse_loss, val_mse_loss, total_z_loss, val_z_loss))
        print('epoch %3d | train mse loss: %2.3f | train z loss: %2.3f' % (epoch, total_mse_loss, total_z_loss))

    if epoch == EPOCHS-1:
        data = np.zeros((t_len,4))
        i = 0
        eegnet.eval()
        for local_batch, _ in training_generator :
            if CUDA:
                local_batch = local_batch.cuda(GPU)
            _, mu, logvar = eegnet(local_batch)
            if CUDA:
                mu = mu.cpu()
            data[i*PARA:i*PARA+PARA,:2] = mu.detach().numpy()
            data[i*PARA:i*PARA+PARA,2:4] = logvar.detach().numpy()
            i += 1
        np.save('out.npy', data)
        torch.save(eegnet.state_dict(), 'vae.pkl')
    
    # else:
    #     for local_batch in training_generator:
    #         if CUDA:
    #             local_batch = local_batch.cuda(GPU)
    #         output, mu, logvar = eegnet(local_batch)
    #         loss = loss_func(output, local_batch, mu, logvar)
    #         optimizer.zero_grad() # clear gradients for this training step
    #         loss.backward() # backpropagation, compute gradients
    #         optimizer.step() # apply gradients
