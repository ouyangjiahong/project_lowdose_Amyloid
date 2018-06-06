from scipy import io as sio
import numpy as np
import os
import dicom
import nibabel as nib
import sys
from data_preparation_tools import *


dir_home = '/data3/Amyloid/'
dir_data1 = '/data3/Amyloid/ADNI_PROJECTS/ADNI_LOW_DOSE/TRAIN_JPR/'
dir_samples = dir_home+'ADNI_temp/data_sample/'

'''
dataset
'''
filename_init = ''
list_subject = [x for x in os.listdir(dir_data1) if os.path.isdir(os.path.join(dir_data1,x))]

list_dataset_train = []
ext_nifti='.nii'
key_T1 = 'T1_nifti_inv_resample'
key_T2FLAIR = 'T2_FLAIR_nifti_inv_resample'
key_PET = 'PET_resample'
key_lowPET = 'PET_quarter_resample'
for subject_id in list_subject:
    dir_subject = os.path.join(dir_data1, subject_id)
    list_filename_nifti = sorted([x for x in os.listdir(dir_subject) if x.find(ext_nifti)>0])
    filename_PET = [x for x in list_filename_nifti if x.lower().find(key_PET.lower())>=0][0]
    filename_lowPET = [x for x in list_filename_nifti if x.lower().find(key_lowPET.lower())>=0][0]
    filename_T1 = [x for x in list_filename_nifti if x.lower().find(key_T1.lower())>=0][0]
    filename_T2FLAIR = [x for x in list_filename_nifti if x.lower().find(key_T2FLAIR.lower())>=0][0]
    list_dataset_train.append({'input':[os.path.join(dir_subject, filename_lowPET),
    	                           os.path.join(dir_subject, filename_T1),
                                   os.path.join(dir_subject, filename_T2FLAIR)],
                         'gt':os.path.join(dir_subject, filename_PET)}
                        )
dir_train_history = dir_home+'ADNI_ckpt/'
num_dataset_train = len(list_dataset_train)
print('process {0} data description'.format(num_dataset_train))

'''
augmentation
'''
list_augments = []
num_augment_flipxy = 2
num_augment_flipx = 2
num_augment_flipy = 2
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
		data_train_input = prepare_data_from_nifti(path_train_input, list_augments)
		list_data_train_input.append(data_train_input)
	data_train_input = np.concatenate(list_data_train_input, axis=-1)


	# load data ground truth
	path_train_gt = list_dataset_train[index_data]['gt']
	data_train_gt = prepare_data_from_nifti(path_train_gt, list_augments)
	# append
	# list_train_input.append(data_train_input)
	# list_train_gt.append(data_train_gt)

	# export
	index_sample_total = export_data_to_npz(data_train_input,
											data_train_gt,
											dir_samples,
											index_sample_total,
											ext_data)
