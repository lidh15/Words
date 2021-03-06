# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 22:19:15 2018

@author: Li Denghao
"""
from torch import nn, randn, transpose, cat, zeros_like, unsqueeze, tensor
# from concurrent.futures import ThreadPoolExecutor

class EEGvae(nn.Module):
    def __init__(self, gpu=-1, zdim=2, chans=12, length=63, fband=14):
        super(EEGvae, self).__init__()
        fg_chans = 35
        filter_len = 7
        self.filter_generator = nn.Sequential(
            nn.Linear(chans*fband, fg_chans),
            nn.Tanh(), 
            nn.Linear(fg_chans, filter_len),
            nn.Tanh(),
        )
        self.filter_padding = int((filter_len-1)/2)

        en_chans1 = 16
        en_chans2 = 32
        en_kernel = 6
        en_stride = 3
        self.encoder = nn.Sequential( 
            # input shape (chans, length-filter_len+1)
            nn.Conv1d(chans, en_chans1, en_kernel, en_stride),
            # output shape (en_chans1, 18) see pytorch doc for output shape calculation detail
            # nn.Dropout(p=0.2),
            nn.BatchNorm1d(en_chans1),
            nn.Tanh(),
            # input shape (en_chans1, 18)
            nn.Conv1d(en_chans1, en_chans2, en_kernel, en_stride),
            # output shape (en_chans2, 5) see pytorch doc for output shape calculation detail
            # nn.Dropout(p=0.2),
            nn.BatchNorm1d(en_chans2),
            nn.Tanh(),
        )
        en_out_size = en_chans2*5

        self.h_mu = nn.Linear(int(en_out_size), zdim)
        self.h_logvar = nn.Linear(int(en_out_size), zdim)

        de_step = 5
        de_out_size = 50
        self.decoder = nn.Sequential( 
            # input shape (1, zdim)
            nn.Linear(zdim, zdim*de_step), 
            # output shape (1, zdim*de_step)
            # nn.Dropout(p=0.2),
            nn.Tanh(),
            # input shape (1, zdim*de_step)
            nn.Linear(zdim*de_step, de_out_size), 
            # output shape (1, 2*step2*step2)
            # nn.Dropout(p=0.2),
            nn.Tanh(), 
        )

        self.gpu = gpu
        self.zdim = zdim
        self.chans = chans
        self.length = length
        self.fband = fband
        
    # Call function here made it stupidly slow.
    # def adaptive_filt(self, filter_input):
    #     adaptive_filter, raw_eeg = unsqueeze(unsqueeze(filter_input[0], 0), 0), unsqueeze(unsqueeze(filter_input[1], 0), 0)
    #     filtered_eeg = nn.functional.conv1d(raw_eeg, adaptive_filter, padding=self.filter_padding)
    #     return filtered_eeg.squeeze(0).squeeze(0)
        
    def forward(self, raw_eeg, eegf):
        '''

        '''
        batch_size = raw_eeg.shape[0]
        adaptive_filter = self.filter_generator(eegf)

        # eeg = [list(eeg_piece) for eeg_piece in ThreadPoolExecutor(max_workers=8).map(self.adaptive_filt, zip(adaptive_filter, raw_eeg))]
        # eeg = [list(self.adaptive_filt(eeg_piece)) for eeg_piece in zip(adaptive_filter, raw_eeg)]
        # It seems that multithreading acceleration is useless.
        # eeg = tensor(eeg)
        # if self.gpu >= 0:
        #     eeg = eeg.cuda(self.gpu)

        eeg = zeros_like(raw_eeg)
        for i in range(batch_size):
            eeg[i] = nn.functional.conv1d(
                unsqueeze(unsqueeze(raw_eeg[i], 0), 0),
                unsqueeze(unsqueeze(adaptive_filter[i], 0), 0),
                padding=self.filter_padding
                ).squeeze(0).squeeze(0)
        eeg = eeg.reshape((batch_size, self.chans, -1))[:, :, self.filter_padding:-self.filter_padding]

        en_out = self.encoder(eeg)
        en_out = en_out.view(en_out.size(0), -1)
        mu = self.h_mu(en_out)
        logvar = self.h_logvar(en_out)
        std = logvar.mul(0.5).exp()
        eps = randn(mu.size(0), mu.size(1))
        if self.gpu >= 0:
            eps = eps.cuda(self.gpu)
        z = mu + eps * std
        z = unsqueeze(z, -2)
        output = self.decoder(z).squeeze(-2)

        return output, mu, logvar

class EEGnaive(nn.Module):
    def __init__(self, gpu=-1, zdim=2, chans=12, length=63, fband=14):
        super(EEGnaive, self).__init__()
        fg_chans = 35
        filter_len = 7
        self.filter_generator = nn.Sequential(
            nn.Linear(chans*fband, fg_chans),
            nn.Tanh(),
            nn.Linear(fg_chans, filter_len),
            nn.Tanh(),
        )
        self.filter_padding = int((filter_len-1)/2)
        chans2 = 16
        out_size = chans2*5

        t1_chans1 = 16
        t1_kernel = 6
        t1_stride = 3
        self.conv_t1 = nn.Sequential( 
            # input shape (chans, length-filter_len+1)
            nn.Conv1d(chans, t1_chans1, t1_kernel, t1_stride, dilation=1),
            # output shape (t1_chans1, 18) see pytorch doc for output shape calculation detail
            # nn.Dropout(p=0.2),
            nn.BatchNorm1d(t1_chans1),
            nn.ReLU(),
            # input shape (t1_chans1, 18)
            nn.Conv1d(t1_chans1, chans2, t1_kernel, t1_stride, dilation=1),
            # output shape (chans2, 5) see pytorch doc for output shape calculation detail
            # nn.Dropout(p=0.2),
            nn.BatchNorm1d(chans2),
            nn.ReLU(),
        )

        t2_chans1 = 16
        t2_kernel = 3
        t2_stride = 3
        self.conv_t2 = nn.Sequential( 
            # input shape (chans, length-filter_len+1)
            nn.Conv1d(chans, t2_chans1, t2_kernel, t2_stride, dilation=2),
            # output shape (t2_chans1, 18) see pytorch doc for output shape calculation detail
            # nn.Dropout(p=0.2),
            nn.BatchNorm1d(t2_chans1),
            nn.ReLU(),
            # input shape (t2_chans1, 18)
            nn.Conv1d(t2_chans1, chans2, t2_kernel, t2_stride, dilation=2),
            # output shape (chans2, 5) see pytorch doc for output shape calculation detail
            # nn.Dropout(p=0.2),
            nn.BatchNorm1d(chans2),
            nn.ReLU(),
        )
        
        c0_chans = 9
        self.conv_c0 = nn.Sequential(
            nn.Linear(chans, c0_chans),
            nn.ReLU(),
        )

        c1_chans1 = 16
        c1_kernel = 5
        c1_stride = (3, 1)
        self.conv_c1 = nn.Sequential(
            # input shape (1, length-filter_len+1, c0_chans)
            nn.Conv2d(1, c1_chans1, (c1_kernel, c0_chans), c1_stride, dilation=1),
            # output shape (c1_chans1, 18, 1) see pytorch doc for output shape calculation detail
            # nn.Dropout(p=0.2),
            nn.BatchNorm1d(c1_chans1),
            nn.ReLU(),
            # input shape (c1_chans1, 18, 1)
            nn.Conv2d(c1_chans1, chans2, (c1_kernel, 1), c1_stride, dilation=1),
            # output shape (chans2, 5, 1) see pytorch doc for output shape calculation detail
            # nn.Dropout(p=0.2),
            nn.BatchNorm1d(chans2),
            nn.ReLU(),
        )

        c2_chans1 = 16
        c2_kernel = 11
        c2_stride = (5, 1)
        self.conv_c2 = nn.Sequential(
            # input shape (1, length-filter_len+1, c0_chans)
            nn.Conv2d(1, c2_chans1, (c2_kernel, c0_chans), c2_stride, dilation=1),
            # output shape (c2_chans1, 10, 1) see pytorch doc for output shape calculation detail
            # nn.Dropout(p=0.2),
            nn.BatchNorm1d(c2_chans1),
            nn.ReLU(),
            # input shape (c2_chans1, 10, 1)
            nn.Conv2d(c2_chans1, chans2, (3, 1), 2, padding=1, dilation=1),
            # output shape (chans2, 5, 1) see pytorch doc for output shape calculation detail
            # nn.Dropout(p=0.2),
            nn.BatchNorm1d(chans2),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(out_size*4, 50),
            nn.Tanh(),
        )

        self.gpu = gpu
        self.zdim = zdim
        self.chans = chans
        self.length = length
        self.fband = fband
        
    # Call function here made it stupidly slow.
    # def adaptive_filt(self, filter_input):
    #     adaptive_filter, raw_eeg = unsqueeze(unsqueeze(filter_input[0], 0), 0), unsqueeze(unsqueeze(filter_input[1], 0), 0)
    #     filtered_eeg = nn.functional.conv1d(raw_eeg, adaptive_filter, padding=self.filter_padding)
    #     return filtered_eeg.squeeze(0).squeeze(0)
        
    def forward(self, raw_eeg, eegf):
        '''

        '''
        batch_size = raw_eeg.shape[0]
        adaptive_filter = self.filter_generator(eegf)

        # eeg = [list(eeg_piece) for eeg_piece in ThreadPoolExecutor(max_workers=8).map(self.adaptive_filt, zip(adaptive_filter, raw_eeg))]
        # eeg = [list(self.adaptive_filt(eeg_piece)) for eeg_piece in zip(adaptive_filter, raw_eeg)]
        # It seems that multithreading acceleration is useless.
        # eeg = tensor(eeg)
        # if self.gpu >= 0:
        #     eeg = eeg.cuda(self.gpu)

        eeg = zeros_like(raw_eeg)
        for i in range(batch_size):
            eeg[i] = nn.functional.conv1d(
                unsqueeze(unsqueeze(raw_eeg[i], 0), 0),
                unsqueeze(unsqueeze(adaptive_filter[i], 0), 0),
                padding=self.filter_padding
                ).squeeze(0).squeeze(0)
        eeg = eeg.reshape((batch_size, self.chans, -1))[:, :, self.filter_padding:-self.filter_padding]

        t1_out = self.conv_t1(eeg).view(batch_size, -1)
        t2_out = self.conv_t2(eeg).view(batch_size, -1)
        eeg_c = self.conv_c0(transpose(eeg,-2,-1)).unsqueeze(1)
        c1_out = self.conv_c1(eeg_c).view(batch_size, -1)
        c2_out = self.conv_c1(eeg_c).view(batch_size, -1)
        fc_out = self.fc(cat((t1_out,t2_out,c1_out,c2_out),1))

        return fc_out
