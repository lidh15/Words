# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 22:19:15 2018

@author: Li Denghao
"""
from torch import nn, transpose, cat, randn, zeros_like, unsqueeze

class EEGvae2d(nn.Module):
    def __init__(self, gpu=-1, chans=12, fband=18):
        super(EEGvae2d, self).__init__()
        chans1 = 16
        chans2 = 8
        step1 = 7
        self.en = nn.Sequential( 
            # input shape (1, chans*fband)
            nn.Conv1d(1, chans1, step1, step1),
            # output shape (chans1, chans*fband/step1)
            # nn.Dropout(p=0.2),
            nn.BatchNorm1d(chans1),
            nn.Tanh(),
            # input shape (chans1, chans*fband/step1)
            nn.Conv1d(chans1, chans2, step1, step1),
            # output shape (chans2, chans*fband/step1/step1)
            # nn.Dropout(p=0.2),
            nn.BatchNorm1d(chans2),
            nn.Tanh(),
        )
        step2 = 7
        self.de = nn.Sequential( 
            # input shape (1, 2)
            nn.Linear(2, 2*step2), 
            # output shape (1, 2*step2)
            # nn.Dropout(p=0.2),
            nn.Tanh(), 
            # input shape (1, 2*step2)
            nn.Linear(2*step2, 2*step2*step2), 
            # output shape (1, 2*step2*step2)
            # nn.Dropout(p=0.2),
            nn.Tanh(), 
            # input shape (1, 2*step2*step2)
            nn.Linear(2*step2*step2, chans*fband), 
            # output shape (1, chans*fband)
            # nn.Dropout(p=0.2),
        )
        self.h1 = nn.Linear(int(chans2*chans*fband/step1/step1), 2)
        self.h2 = nn.Linear(int(chans2*chans*fband/step1/step1), 2)
        self.activate_t = nn.Hardtanh(min_val=-5, max_val=5)
        self.activate_f = nn.ReLU()
        self.gpu = gpu
        self.freq = freq
        
    def forward(self, x):
        x = unsqueeze(x, -2)
        enout = self.en(x)
        enout =  enout.view(enout.size(0), -1)
        mu = self.h1(enout)
        logvar = self.h2(enout)
        std = logvar.mul(0.5).exp()
        eps = randn(mu.size(0), mu.size(1))
        if self.gpu >= 0:
            eps = eps.cuda(self.gpu)
        z = mu + eps * std
        z = unsqueeze(z, -2)
        output = self.de(z).squeeze(-2)
        if self.freq:
            output = self.activate_f(output)
        else:
            output = self.activate_t(output)
        return output, mu, logvar
