import h5py


for i in range(50):
    f = h5py.File('%03d.h5'%(i+1), 'r')
    print(int((len(f['data'])-95175)/13))
    f.close()
