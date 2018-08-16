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
parser.add_argument('--is_F_norm', dest='is_F_norm', action='store_true', help='use Frobenius norm or not')
parser.set_defaults(is_F_norm=False)
parser.add_argument('--is_mean_norm', dest='is_mean_norm', action='store_true', help='use mean norm or not')
parser.set_defaults(is_mean_norm=False)
parser.add_argument('--dimension', dest='dimension', default='2', help='2, 2.5, 3')
parser.add_argument('--block', dest='block', type=int, default=4, help='2.5D data will be 2*block+1')
parser.add_argument('--task', dest='task', default='petonly', help='lowdose, petonly, zerodose, only useful when its 2.5D')

args = parser.parse_args()
set = args.set
dir_data_ori = args.dir_data_ori
dir_data_dst = args.dir_data_dst
is_F_norm = args.is_F_norm
is_mean_norm = args.is_mean_norm
dimension = args.dimension
block = args.block
task = args.task

crop = 20       # remove first and last 20 slices

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
# train_set = [1350, 1726, 1750, 1758, 1762, 1785, 1791, 1827, 1838, 1905, 1907,
#             1978, 2014, 2016, 2157, 2214, 2304, 2317, 2376, 2427, 2414, 1961,
#             2185, 2152, 1375, 1789, 1816, 1965, 2314, 2511, 2416, 2425]
# test_set = [1355, 1732, 1947, 2482, 2516, 2063, 1923, 50767]

# test on pos/neg
# train_set = [1355, 1732, 1762, 1785, 1791, 1827, 1838, 1905, 1978, 1947,
#             2014, 2016, 2157, 2214, 2304, 2317, 2376, 2427, 2482, 1961,
#             2185, 2152, 1375, 1816, 1965, 2314, 2511, 2416, 2516, 2063, 50767]
# test_set = [1350, 1375, 1726, 1750, 1758, 1789, 1907, 1923, 2414, 2425]
train_set = [1350, 1726, 1750, 1762, 1785, 1791, 1827, 1838, 1905, 1907,
            1978, 2014, 2016, 2157, 2214, 2304, 2317, 2376, 2427, 2414, 1961,
            2185, 2152, 1789, 1816, 1965, 2314, 2511, 2416, 2482]
test_set = [1355, 1732, 1947, 2516, 2063, 50767, 1375, 1758, 1923, 2425]

if set == 'test':
    using_set = test_set
else:
    using_set = train_set

list_subject = [str(x) for x in using_set if os.path.isdir(os.path.join(dir_data_ori, str(x)))]
print('generating subject list:')
print(list_subject)

list_dataset_train = []


filename_lowPET = 'pet_nifti/501_.nii.gz'
filename_PET = 'pet_nifti/500_.nii.gz'
filename_T1 = 'mr_nifti/T1_nifti_inv.nii'
filename_T2 = 'mr_nifti/T2_nifti_inv.nii'
filename_T2FLAIR = 'mr_nifti/T2_FLAIR_nifti_inv.nii'
if dimension == '2':
    for subject_id in list_subject:
        dir_subject = os.path.join(dir_data_ori, subject_id)
        list_dataset_train.append({'input':[os.path.join(dir_subject, filename_lowPET),
                                       os.path.join(dir_subject, filename_T1),
                                       os.path.join(dir_subject, filename_T2),
                                       os.path.join(dir_subject, filename_T2FLAIR)],
                             'gt':os.path.join(dir_subject, filename_PET)})

elif dimension == '2.5':
    if task == 'petonly':
        for subject_id in list_subject:
            dir_subject = os.path.join(dir_data_ori, subject_id)
            list_dataset_train.append({'input':[os.path.join(dir_subject, filename_lowPET)],
                                 'gt':os.path.join(dir_subject, filename_PET)})

    elif task == 'lowdose':
        for subject_id in list_subject:
            dir_subject = os.path.join(dir_data_ori, subject_id)
            list_dataset_train.append({'input':[os.path.join(dir_subject, filename_lowPET),
                                            os.path.join(dir_subject, filename_T1),
                                            os.path.join(dir_subject, filename_T2),
                                            os.path.join(dir_subject, filename_T2FLAIR)],
                                 'gt':os.path.join(dir_subject, filename_PET)})

    else:           # zerodose
        for subject_id in list_subject:
            dir_subject = os.path.join(dir_data_ori, subject_id)
            list_dataset_train.append({'input':[os.path.join(dir_subject, filename_T1),
                                            os.path.join(dir_subject, filename_T2),
                                            os.path.join(dir_subject, filename_T2FLAIR)],
                                 'gt':os.path.join(dir_subject, filename_PET)})

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
ext_data = 'npz'

'''
generate train data
'''
flatten = lambda l: [item for sublist in l for item in sublist]
list_train_input = []
list_train_gt = []
index_sample_total = 0
list_subject_sample = []
for index_data in range(num_dataset_train):
    # directory
    list_data_train_input = []
    lowdose_norm = []
    fulldose_norm = 0
    for path_train_input in list_dataset_train[index_data]['input']:
        # load data
        data_train_input, f_norm = prepare_data_from_nifti(path_train_input,
                                    list_augments, scale_by_F_norm=is_F_norm,
                                    scale_by_mean_norm=is_mean_norm, crop=crop, set=set)
        list_data_train_input.append(data_train_input)
        lowdose_norm.append(f_norm)
    data_train_input = np.concatenate(list_data_train_input, axis=-1)


    # load data ground truth
    path_train_gt = list_dataset_train[index_data]['gt']
    data_train_gt, fulldose_norm = prepare_data_from_nifti(path_train_gt,
                                    list_augments, scale_by_F_norm=is_F_norm,
                                    scale_by_mean_norm=is_mean_norm, crop=crop, set=set)

    start_idx = index_sample_total
    # export
    if set == 'test':
        slices = 89
    else:
        slices = 89-2*crop-1
    index_sample_total = export_data_to_npz(data_train_input,
                                            data_train_gt,
                                            dir_data_dst,
                                            index_sample_total,
                                            ext_data, dimension,
                                            slices=slices, block=block)

    for i in range(start_idx, index_sample_total):
        info = flatten([[using_set[index_data]], lowdose_norm, [fulldose_norm]])
        list_subject_sample.append(info)

dict = np.array(list_subject_sample)
subject_list = np.array(using_set)
# pdb.set_trace()
norm = ''
if is_F_norm == True:
    norm = 'F_norm_'
if is_mean_norm == True:
    norm = 'mean_norm_'
if dimension == '2.5':
    npz_name = task + '_' + norm + str(2*block+1)+'block_'+set+'_subject_sample.npz'
elif dimension == '2':
    npz_name = norm + '2D_'+set+'_subject_sample.npz'
np.savez_compressed(npz_name, dict=dict, subject_list=subject_list)
# pdb.set_trace()
