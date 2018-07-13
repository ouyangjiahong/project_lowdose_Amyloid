import argparse
from scipy import io as sio
import numpy as np
import os
import dicom
import nibabel as nib
import sys
import pdb
from data_preparation_tools import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--set', dest='set', default='train', help='train, test')
parser.add_argument('--dir_data_ori', dest='dir_data_ori', default='/home/data/', help='folder of original data')
parser.add_argument('--dir_data_dst', dest='dir_data_dst', default='data/Amyloid_norm/', help='folder of dst npz data')
parser.add_argument('--norm', dest='norm', action='store_false', help='use Frobenius norm or nor')
parser.set_defaults(norm=True)
parser.add_argument('--dimension', dest='dimension', default='2', help='2, 2.5, 3')
parser.add_argument('--block', dest='block', type=int, default=4, help='2.5D data will be 2*block+1')

args = parser.parse_args()
set = args.set
dir_data_ori = args.dir_data_ori
dir_data_dst = args.dir_data_dst
norm = args.norm
dimension = args.dimension
block = args.block


# stanford machine
# dir_data_ori = '/data3/Amyloid/'
# dir_data_dst = '/home/jiahong/data/Amyloid/'

# IBM machine
# dir_data_ori = '~/data/Amyloid/'
# dir_data_dst = '~/project_lowdose/data/'

# cmu machine
# dir_data_ori = '/home/jihang/Jiahong/data/'
# dir_data_dst = '/data/Amyloid_npz/'
# set = 'test'
# norm = True


dir_data_dst = dir_data_dst + set + '/'
if not os.path.exists(dir_data_dst):
    os.makedirs(dir_data_dst)

'''
dataset
'''
filename_init = ''
# all_set = [1350, 1355, 1375, 1726, 1732, 1750, 1758, 1762, 1771, 1785, 1789,
#             1791, 1816, 1827, 1838, 1905, 1907, 1923, 1947, 1961, 1965, 1978,
#             2014, 2016, 2063, 2152, 2157, 2185, 2214, 2287, 2304, 2314, 2317,
#             2376, 2414, 2416, 2425, 2427, 2482, 2511, 2516, 4501, 4739, 50767]
# train_set = [1350, 1355, 1375, 1726, 1732, 1750, 1758, 1762, 1771, 1785, 1789,
#             1791, 1816, 1827, 1838, 1905, 1907, 1923, 1947, 1961, 1965, 1978,
#             2014, 2016, 2063, 2152, 2157, 2185, 2214, 2287, 2304, 2314, 2317]
# test_set = [2376, 2414, 2416, 2425, 2427, 2482, 2511, 2516, 4501, 4739, 50767]
# use the same training and testing dataset as Kevin
train_set = [1350, 1726, 1750, 1758, 1762, 1785, 1791, 1827, 1838, 1905, 1907,
            1978, 2014, 2016, 2157, 2214, 2304, 2317, 2376, 2427, 2414, 1961,
            2185, 2152, 1375, 1789, 1816, 1965, 2314, 2511, 2416, 2425]
test_set = [1355, 1732, 1947, 2482, 2516, 2063, 1923, 50767]

if set == 'test':
    using_set = test_set
else:
    using_set = train_set

list_subject = [str(x) for x in using_set if os.path.isdir(os.path.join(dir_data_ori, str(x)))]
print('generating subject list:')
print(list_subject)

list_dataset_train = []

if dimension == '2':
    filename_lowPET = 'pet_nifti/501_.nii.gz'
    filename_PET = 'pet_nifti/500_.nii.gz'
    filename_T1 = 'mr_nifti/T1_nifti_inv.nii'
    filename_T2 = 'mr_nifti/T2_nifti_inv.nii'
    filename_T2FLAIR = 'mr_nifti/T2_FLAIR_nifti_inv.nii'

    for subject_id in list_subject:
        dir_subject = os.path.join(dir_data_ori, subject_id)
        list_dataset_train.append({'input':[os.path.join(dir_subject, filename_lowPET),
                                       os.path.join(dir_subject, filename_T1),
                                       os.path.join(dir_subject, filename_T2),
                                       os.path.join(dir_subject, filename_T2FLAIR)],
                             'gt':os.path.join(dir_subject, filename_PET)}
                            )
elif dimension == '2.5':
    filename_lowPET = 'pet_nifti/501_.nii.gz'
    filename_PET = 'pet_nifti/500_.nii.gz'

    for subject_id in list_subject:
        dir_subject = os.path.join(dir_data_ori, subject_id)
        list_dataset_train.append({'input':[os.path.join(dir_subject, filename_lowPET)],
                             'gt':os.path.join(dir_subject, filename_PET)}
                            )

num_dataset_train = len(list_dataset_train)
print('process {0} data description'.format(num_dataset_train))

'''
augmentation
'''
list_augments = []
if set == 'train':
    num_augment_flipxy = 2
    num_augment_flipx = 2
    num_augment_flipy = 2
    num_augment_shiftx = 1
    num_augment_shifty = 1
else:
    num_augment_flipxy = 1
    num_augment_flipx = 1
    num_augment_flipy = 1
    num_augment_shiftx = 1
    num_augment_shifty = 1

for flipxy in range(num_augment_flipxy):
    for flipx in range(num_augment_flipx):
        for flipy in range(num_augment_flipy):
            for shiftx in range(num_augment_shiftx):
                for shifty in range(num_augment_shifty):
                    augment={'flipxy':flipxy,'flipx':flipx,'flipy':flipy,'shiftx':shiftx,'shifty':shifty}
                    list_augments.append(augment)
num_augment=len(list_augments)
print('will augment data with {0} augmentations'.format(num_augment))

'''
file loading related
'''
ext_dicom = 'MRDC'
key_sort = lambda x: int(x.split('.')[-1])
scale_method = lambda x:np.mean(np.abs(x))
scale_by_mean = False
scale_factor = 1/32768.
ext_data = 'npz'

'''
generate train data
'''
list_train_input = []
list_train_gt = []
index_sample_total = 0
for index_data in range(num_dataset_train):
    # directory
    list_data_train_input = []
    for path_train_input in list_dataset_train[index_data]['input']:
        # load data
        data_train_input = prepare_data_from_nifti(path_train_input, list_augments, scale_by_norm=norm)
        list_data_train_input.append(data_train_input)
    data_train_input = np.concatenate(list_data_train_input, axis=-1)


    # load data ground truth
    path_train_gt = list_dataset_train[index_data]['gt']
    data_train_gt = prepare_data_from_nifti(path_train_gt, list_augments, scale_by_norm=norm)
    # append
    # list_train_input.append(data_train_input)
    # list_train_gt.append(data_train_gt)

    # export
    index_sample_total = export_data_to_npz(data_train_input,
                                            data_train_gt,
                                            dir_data_dst,
                                            index_sample_total,
                                            ext_data, dimension, block)
