import numpy as np
from glob import glob
import matplotlib.pyplot as plt
#
# file_list = glob('{}/test/*.{}'.format('../data/Amyloid', 'npz'))
#
# file_num = len(file_list)
# stat = np.zeros((file_num, 10))
# for i, file_name in enumerate(file_list):
#     data = np.load(file_name)
#     input = data['input']
#     output = data['output']
#     stat[i, 0] = np.amax(input[:,:,0])
#     stat[i, 1] = np.mean(input[:,:,0])
#     stat[i, 2] = np.amax(input[:,:,1])
#     stat[i, 3] = np.mean(input[:,:,1])
#     stat[i, 4] = np.amax(input[:,:,2])
#     stat[i, 5] = np.mean(input[:,:,2])
#     stat[i, 6] = np.amax(input[:,:,3])
#     stat[i, 7] = np.mean(input[:,:,3])
#     stat[i, 8] = np.amax(output)
#     stat[i, 9] = np.mean(output)
#
# np.save('pixel_value_statistics.npy', stat)

stat = np.load('pixel_value_statistics.npy')

# input_max = stat[:,0]
# plt.hist(input_max, bins='auto')
# plt.title('histogram of max in input PET')
# plt.savefig('histogram_input_pet_max.png')
# plt.clf()
#
# input_mean = stat[:,1]
# plt.hist(input_mean, bins='auto')
# plt.title('histogram of mean in input PET')
# plt.savefig('histogram_input_pet_mean.png')
# plt.clf()

input_max = stat[:,2]
plt.hist(input_max, bins='auto')
plt.title('histogram of max in input T1')
plt.savefig('histogram_input_T1_max.png')
plt.clf()
#
input_mean = stat[:,3]
plt.hist(input_mean, bins='auto')
plt.title('histogram of mean in input T1')
plt.savefig('histogram_input_T1_mean.png')
plt.clf()

input_max = stat[:,4]
plt.hist(input_max, bins='auto')
plt.title('histogram of max in input T2')
plt.savefig('histogram_input_T2_max.png')
plt.clf()

input_mean = stat[:,5]
plt.hist(input_mean, bins='auto')
plt.title('histogram of mean in input T2')
plt.savefig('histogram_input_T2_mean.png')
plt.clf()

input_max = stat[:,6]
plt.hist(input_max, bins='auto')
plt.title('histogram of max in input T2F')
plt.savefig('histogram_input_T2F_max.png')
plt.clf()

input_mean = stat[:,7]
plt.hist(input_mean, bins='auto')
plt.title('histogram of mean in input T2F')
plt.savefig('histogram_input_T2F_mean.png')
plt.clf()

output_max = stat[:,8]
plt.hist(output_max, bins='auto')
plt.title('histogram of max in output')
plt.savefig('histogram_output_max.png')
plt.clf()

output_mean = stat[:,9]
plt.hist(output_mean, bins='auto')
plt.title('histogram of mean in output')
plt.savefig('histogram_output_mean.png')
plt.clf()
