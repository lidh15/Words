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

from models import EEGnaive
from datasets import EEG500ms


CUDA = 1
LR = 0.0001
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

# Datasets
training_set = EEG500ms(test_IDs=training, max_sample_num=20000)
t_len = len(training_set)
# Generators
training_generator = DataLoader(training_set, **params)
# Network
if CUDA:
    eegnet = EEGnaive(GPU).cuda(GPU)
else:
    eegnet = EEGnaive()

loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(eegnet.parameters(), lr=LR)
# Training process
for epoch in range(EPOCHS):
    adjust_learning_rate(optimizer, epoch)
    # if epoch % 5 == 4:
    if True:
        total_loss = 0
        for local_batch, local_f, target in training_generator:
            if CUDA:
                local_batch = local_batch.cuda(GPU)
                local_f = local_f.cuda(GPU)
                target = target.cuda(GPU)
            output = eegnet(local_batch, local_f)
            loss = loss_func(output, target)
            total_loss += loss.data.item()/t_len
            optimizer.zero_grad() # clear gradients for this training step
            loss.backward() # backpropagation, compute gradients
            optimizer.step() # apply gradient
        
        # print('training done')

        # # Validation process
        # val_loss = 0
        # for val_batch, val_f, target in validation_generator:
        #     if CUDA:
        #         val_batch = val_batch.cuda(GPU)
        #         val_f = val_f.cuda(GPU)
        #         target = target.cuda(GPU)
        #     output = eegnet(val_batch, val_f)
        #     loss = loss_func(output, target)
        #     val_loss += loss.data.item()/v_len
        
        # print('epoch %3d | train loss: %2.3f | val loss: %2.3f % (epoch, total_loss, val_loss))
        print('epoch %3d | train loss: %2.3f ' % (epoch, total_loss))

    if epoch == EPOCHS-1:
        data = np.zeros((t_len, 100))
        i = 0
        eegnet.eval()
        for local_batch, local_f, target in training_generator:
            if CUDA:
                local_batch = local_batch.cuda(GPU)
                local_f = local_f.cuda(GPU)
            output = eegnet(local_batch, local_f)
            if CUDA:
                output = output.cpu()
            data[i*PARA:i*PARA+PARA, :50] = output.detach().numpy()
            data[i*PARA:i*PARA+PARA, 50:] = target
            i += 1
        np.save('out.npy', data)
        torch.save(eegnet.state_dict(), 'naive.pkl')
    
    # else:
    #     for local_batch in training_generator:
    #         if CUDA:
    #             local_batch = local_batch.cuda(GPU)
    #         output, mu, logvar = eegnet(local_batch)
    #         loss = loss_func(output, local_batch, mu, logvar)
    #         optimizer.zero_grad() # clear gradients for this training step
    #         loss.backward() # backpropagation, compute gradients
    #         optimizer.step() # apply gradients
