import csv
import os
import numpy as np
import h5py
from scipy.signal import cheb2ord, cheby2, filtfilt


chanLabels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
rawDataPath = './rawData/'
dataPath = './data/'
files = os.listdir(rawDataPath)
for csvFile in files:
    if csvFile.endswith('.pm.csv'):
        pass
    elif csvFile.endswith('.bp.csv'):
        pass
    elif csvFile.endswith('.csv') and not csvFile[:3]+'.h5' in os.listdir(dataPath):
        with open(rawDataPath+csvFile) as f:
            csvArray = np.asarray(list(csv.reader(f))[1:], dtype='float64')
            csvLen = len(csvArray)
            chans = len(chanLabels)
            data = np.zeros((csvLen,chans))
            data[:,:6] = csvArray[:,2:8]
            data[:,6:12] = csvArray[:,10:16]
            for i in [2,4,6,8]:
                marker = np.where(csvArray[:,20]==i)[0]
                if len(marker):
                    break
            start = marker[-1]+128                
            timeStamp = csvArray[:,22]*1000+csvArray[:,23]
            timeStamp = timeStamp[start:]
            timeStamp -= timeStamp[0]
            freq = len(timeStamp)/timeStamp[-1]*1000/2
            N, Wn = cheb2ord([8/freq, 35/freq], [5/freq, 40/freq], 3, 20)
            b, a = cheby2(N, 20, Wn, 'bandpass')
            data = data[start:]
            for i in range(chans):
                data[:,i] = filtfilt(b, a, data[:,i], method='gust')
            # np.save(data, './data/'+csvFile[:3]+'.npy')
        f = h5py.File(dataPath+csvFile[:3]+'.h5','w')
        f['data'] = data
        f['time'] = timeStamp
        f['frequency'] = freq*2
        f['marker'] = marker
        f.close()
        print(csvFile[:3]+' '+str(freq*2-126.89))
