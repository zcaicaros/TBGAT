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


j = 20
m = 5
start_segment_flag = 0
end_segment_flag = 10
tp = 'tai'
result = np.load('./test_data_fssp/ortools_result_FSSP-{}{}x{}[{},{}]_result.npy'.format(tp, j, m, start_segment_flag, end_segment_flag))
time = np.load('./test_data_fssp/ortools_result_FSSP-{}{}x{}[{},{}]_time.npy'.format(tp, j, m, start_segment_flag, end_segment_flag))
print(result)
print(time)


j = 20
m = 10
l2s_result = np.load('l2s_result_{}x{}.npy'.format(j, m))
ts_result = np.load('tabu_search_result_{}x{}.npy'.format(j, m))
print(l2s_result.shape)
print(ts_result.shape)

relative_error = (l2s_result[4] - ts_result[3]) / ts_result[3]

print(relative_error.mean())
