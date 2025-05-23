import csv
import h5py
import os
import numpy as np

savepath = ''

filename = 'labels4DCVAEGCN.h5'
f = h5py.File(filename, 'r')
for idx in range(len(f.keys())):
    a_group_key = list(f.keys())[idx]
    data = np.array(f[a_group_key])  # Get the data
    np.savetxt(os.path.join(savepath, a_group_key+'.csv'), data, delimiter=',')  # Save as csv
