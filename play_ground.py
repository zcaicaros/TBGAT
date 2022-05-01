import numpy as np
from numpy import genfromtxt
import torch
from tqdm import tqdm


j = 30
m = 15
data = np.load('./validation_data/validation_data_and_Cmax_{}x{}_[1,99].npy'.format(j, m))

# data[:, 1, :, :] += 1

# np.save('./validation_data/validation_data_and_Cmax_{}x{}_[1,99].npy'.format(j, m), data)

print(data[0][1])
