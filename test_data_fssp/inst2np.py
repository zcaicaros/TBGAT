import pandas
import numpy as np

j = 20
m = 20
processing_time = np.loadtxt('./tai_{}x{}.txt'.format(j, m), dtype=int)


processing_time_reshape = processing_time.reshape([-1, m, j]).transpose((0, 2, 1))
# print(processing_time_reshape.shape)
machine_sequence = np.tile(np.expand_dims(
    np.arange(
        1, processing_time_reshape.shape[2] + 1), axis=0
).repeat(repeats=processing_time_reshape.shape[1], axis=0), (processing_time_reshape.shape[0], 1, 1))
# print(machine_sequence.shape)
data = np.stack([processing_time_reshape, machine_sequence], axis=1)
# print(data)
np.save('tai{}x{}.npy'.format(j, m), data)

# print(np.load('../test_data_jssp/syn10x10_result.npy'))

upper_bound_tai_20x5 = np.array([1278, 1359, 1081, 1293, 1236, 1195, 1239, 1206, 1230, 1108], dtype=float)
upper_bound_tai_20x10 = np.array([1582, 1659, 1496, 1378, 1419, 1397, 1484, 1538, 1593, 1591], dtype=float)
upper_bound_tai_20x20 = np.array([2297, 2100, 2326, 2223, 2291, 2226, 2273, 2200, 2237, 2178], dtype=float)
np.save('tai20x5_result.npy', upper_bound_tai_20x5)
np.save('tai20x10_result.npy', upper_bound_tai_20x10)
np.save('tai20x20_result.npy', upper_bound_tai_20x20)

