import numpy as np
from numpy import genfromtxt
import torch
from tqdm import tqdm


j = 30
m = 20
data = np.load('./validation_data/validation_data_and_Cmax_{}x{}_[1,99].npy'.format(j, m))
print(data[0, 1, 0])
# data[:, 1, :, :] += 1
# print(data[0, 1, 0])
# np.save('./validation_data/validation_data_and_Cmax_{}x{}_[1,99].npy'.format(j, m), data)

j = 200
m = 40
data = np.load('./test_data/syn{}x{}.npy'.format(j, m))
result = np.load('./test_data/syn{}x{}_result.npy'.format(j, m))
time = np.load('./test_data/syn{}x{}_time.npy'.format(j, m))
print(data[0].shape)
print(result)
print(time)
