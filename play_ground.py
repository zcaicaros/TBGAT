import numpy as np
from numpy import genfromtxt
import torch
from tqdm import tqdm


problem_type = 'JSSP'
j = 30
m = 20
data = np.load('./validation_data/{}_validation_data_and_Cmax_{}x{}_[1,99].npy'.format(problem_type, j, m))
print(data[0, 1, 0])
# data[:, 1, :, :] += 1
# print(data[0, 1, 0])
# np.save('./validation_data/validation_data_and_Cmax_{}x{}_[1,99].npy'.format(j, m), data)

j = 15
m = 15
tp = 'tai'
data = np.load('./test_data_jssp/{}{}x{}.npy'.format(tp, j, m))
result = np.load('./test_data_jssp/{}{}x{}_result.npy'.format(tp, j, m))
# time = np.load('./test_data_jssp/{}{}x{}_time.npy'.format(tp, j, m))
print(data[0].shape)
print(result)
# print(time)


