import argparse
from scipy import io as sio
import numpy as np
import os
import dicom
import nibabel as nib
import openpyxl as px
import sys
import pdb
from data_preparation_classification_tools import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--set', dest='set', default='train', help='train, test')
parser.add_argument('--dir_label', dest='dir_label', default='diagnosis_status.xlsx', help='path of the label xlsx')
parser.add_argument('--dir_data_ori', dest='dir_data_ori', default='/home/data/', help='folder of original data')
parser.add_argument('--dir_data_dst', dest='dir_data_dst', default='data/classification/', help='folder of jpg classification data')
parser.add_argument('--norm', dest='norm', action='store_false', help='divided by forbenius norm than ax value, transform to [-128,128] (similar to value of image-vgg_mean)')
parser.set_defaults(norm=True)

args = parser.parse_args()
set = args.set
dir_label = args.dir_label
dir_data_ori = args.dir_data_ori
dir_data_dst = args.dir_data_dst
norm = args.norm

dir_data_dst = dir_data_dst + set + '/'
if not os.path.exists(dir_data_dst):
    os.makedirs(dir_data_dst)
path = dir_data_dst + 'N'
if not os.path.exists(path):
    os.makedirs(path)
path = dir_data_dst + 'P'
if not os.path.exists(path):
    os.makedirs(path)

'''
labels
'''
label_xls = px.load_workbook(dir_label)
sheet = label_xls.get_sheet_by_name(name='Sheet1')
label_dict = {}
num = 1
for row in sheet.iter_rows():
    if num == 1:
        num += 1
        continue
    # pdb.set_trace()
    label = str(sheet.cell(column=5, row=num).value)
    number = str(sheet.cell(column=6, row=num).value)
    label_dict[number] = label
    num += 1
print(label_dict)

'''
dataset
'''
filename_init = ''
# all_set = [1350, 1355, 1375, 1726, 1732, 1750, 1758, 1762, 1785, 1789,
#             1791, 1816, 1827, 1838, 1905, 1907, 1923, 1947, 1961, 1965, 1978,
#             2014, 2016, 2063, 2152, 2157, 2185, 2214, 2304, 2314, 2317,
#             2376, 2414, 2416, 2425, 2427, 2482, 2511, 2516, 50767]

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
list_dataset_label = []
filename_PET = 'pet_nifti/500_.nii.gz'

for subject_id in list_subject:
    dir_subject = os.path.join(dir_data_ori, subject_id)
    list_dataset_train.append({'gt':os.path.join(dir_subject, filename_PET)})
    list_dataset_label.append(label_dict[subject_id])

num_dataset_train = len(list_dataset_train)
print('process {0} data description'.format(num_dataset_train))
print(list_dataset_label)
# pdb.set_trace()

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
ext_data = 'jpg'

'''
generate train data
'''
list_train_input = []
list_train_gt = []
index_sample_total = 0
for index_data in range(num_dataset_train):
    # directory
    # load data ground truth
    path_train_gt = list_dataset_train[index_data]['gt']
    label = list_dataset_label[index_data]
    data_train_gt = prepare_data_from_nifti(path_train_gt, list_augments, scale_by_norm=norm)

    # export
    index_sample_total = export_data_to_jpg(data_train_gt, dir_data_dst+label+'/', index_sample_total, ext_data)
