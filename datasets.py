# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 10:45:20 2018

@author: Li Denghao

This is based on Shervine Amidi's tutorial: 
    https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel.html
"""
import os
from random import randint, shuffle

import h5py
import numpy as np
from scipy.fftpack import fft

from torch.utils import data


class EEG500ms(data.Dataset):
    def __init__(self, test_IDs=[], max_sample_num=200000, sample_shape=(12,63), data_path='./data/', label_path='./wordslist/', training=True):
        vocab = {}
        eeg_list = []
        label_list = []
        labels = []
        with open('newvocab.txt') as f:
            for line in f.readlines:
                line = line.split(' ')
                vocab[line[0]] = np.asarray([float(x) for x in line[1:]])

        if not test_IDs:
            test_IDs = list(set([file_name[:3] for file_name in os.listdir(data_path)]))
        file_names = os.listdir(data_path)
        for file_name in file_names:
            ID = file_name[:3]
            if ID in test_IDs:
                f = h5py.File(data_path+file_name, 'r')
                '''
                126.9Hz sample frequency (with a little fluctuation, see f['frequency'])
                500 words
                1.5s per word
                126.9 * 500 * 1.5 = 95175
                126.9 * 0.1 = 12.69
                '''
                ideal_len = 95175
                time_step = 12.69
                sample_num = int((len(f['data'])-ideal_len)/time_step)
                f.close()
                if training:
                    with open(label_path+ID+'.txt') as f:
                        i = 0
                        for line in f.readlines:
                            if line in vocab:
                                for j in range(sample_num):
                                    eeg_list.append(int(((i*1.5+1)*126.9)+time_step*j))
                                    label_list.append({'ID':ID, 'target':vocab[line]})
                                    labels.append(vocab[line])
                            i += 1

        
        self.sample_shape = sample_shape
        self.data_path = data_path
        self.label_path = label_path
        self.training = training
        self.eeg_list = eeg_list
        self.label_list = label_list
        self.labels = labels

        print('%6d samples loaded!' % (len(labels)))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        length = self.sample_shape[1]
        chans = self.sample_shape[0]
        pos = self.eeg_list[index]
        f = h5py.File(self.data_path+self.label_list[index]['ID']+'.h5','r')
        eeg = f['data'][pos:pos+length].T
        f.close()
        eeg_std = np.sqrt(np.var(np.sort(eeg,1)[:,3:-3],1))
        for i in range(chans):
            eeg[i] /= eeg_std[i]
        eegf = np.abs(fft(eeg)[:,4:18])
        eeg = eeg.reshape(length*chans)
        eegf = eegf.reshape(eegf.shape[0]*eegf.shape[1])
        if not self.training: # For validation, return all information
            y = self.label_list[index]
        else:
            y = self.labels[index]

        return eeg, eegf, y
