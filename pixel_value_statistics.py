import numpy as np
from glob import glob
import matplotlib.pyplot as plt

file_list = glob('{}/train/*.{}'.format('/data/Amyloid_npz', 'npz'))

file_num = len(file_list)
stat = np.zeros((file_num, 4))
for i, file_name in enumerate(file_list):
    data = np.load(file_name)
    input = data['input']
    output = data['output']
    stat[i, 0] = np.amax(input)
    stat[i, 1] = np.mean(input)
    stat[i, 2] = np.amax(output)
    stat[i, 3] = np.mean(output)

np.save('pixel_value_statistics.npy', stat)

# stat = np.load('pixel_value_statistics.npy')
input_max = stat[:,0]
plt.hist(input_max, bins='auto')
plt.title('histogram of max in input')
plt.savefig('histogram_input_max.png')

input_mean = stat[:,1]
plt.hist(input_mean, bins='auto')
plt.title('histogram of mean in input')
plt.savefig('histogram_input_mean.png')

output_max = stat[:,2]
plt.hist(output_max, bins='auto')
plt.title('histogram of max in output')
plt.savefig('histogram_output_max.png')

output_mean = stat[:,3]
plt.hist(output_mean, bins='auto')
plt.title('histogram of mean in output')
plt.savefig('histogram_output_mean.png')
