import h5py


data_path = './data/'
length = []
for i in range(50):
    f = h5py.File(data_path+'%03d.h5'%(i+1), 'r')
    length.append('%03d %3d\n'%(i+1,int((len(f['data'])-95175)/12.69)))
    f.close()
with open('length.txt','w') as f:
    f.writelines(length)
