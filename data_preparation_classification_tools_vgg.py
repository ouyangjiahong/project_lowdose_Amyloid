# moduls
import os
import numpy as np
import scipy as sci
import dicom
import nibabel as nib
import pdb

def augment_data(data_xy, axis_xy=[1,2], augment={'flipxy':0,'flipx':0,'flipy':0}):
    if 'flipxy' in augment and augment['flipxy']:
        data_xy = np.swapaxes(data_xy, axis_xy[0], axis_xy[1])
    if 'flipx' in augment and augment['flipx']:
        if axis_xy[0] == 0:
            data_xy = data_xy[::-1,...]
        if axis_xy[0] == 1:
            data_xy = data_xy[:, ::-1,...]
    if 'flipy' in augment and augment['flipy']:
        if axis_xy[1] == 1:
            data_xy = data_xy[:, ::-1,...]
        if axis_xy[1] == 2:
            data_xy = data_xy[:, :, ::-1,...]
    if 'shiftx' in augment and augment['shiftx']>0:
        if axis_xy[0] == 0:
            data_xy[:-augment['shiftx'],...] = data_xy[augment['shiftx']:,...]
        if axis_xy[0] == 1:
            data_xy[:,:-augment['shiftx'],...] = data_xy[:,augment['shiftx']:,...]
    if 'shifty' in augment and augment['shifty']>0:
        if axis_xy[1] == 1:
            data_xy[:,:-augment['shifty'],...] = data_xy[:,augment['shifty']:,...]
        if axis_xy[1] == 2:
            data_xy[:,:,:-augment['shifty'],...] = data_xy[:,:,augment['shifty']:,...]
    return data_xy

def prepare_data_from_nifti(path_load, list_augments=[], scale_by_norm=True):
    # get nifti
    nib_load = nib.load(path_load)
    # print(nib_load.header)
    # get data
    data_load = nib_load.get_data()
    # transpose to [x,y,slices] -> [slice, x, y, channel]
    data_load = np.transpose(data_load[:,:,:,np.newaxis], [2,0,1,3])

    # scale
    if scale_by_norm:
        data_load = data_load / np.linalg.norm(data_load.flatten())
        # remove the skull part to decrease noise data
        data_load = data_load[10:-10]
        for i in range(data_load.shape[0]):
            img = data_load[i,:,:,:]
            max_value = np.amax(img)
            if max_value == 0:
                max_value = 1
            data_load[i,:,:,:] = 256 * img / max_value - 1

    # finish loading data
    print('loaded from {0}, data size {1} (sample, x, y, channel)'.format(path_load, data_load.shape))

    # augmentation
    if len(list_augments)>0:
        print('data augmentation')
        list_data = []
        for augment in list_augments:
            print(augment)
            data_augmented = augment_data(data_load, axis_xy = [1,2], augment = augment)
            list_data.append(data_augmented.reshape(data_load.shape))
        data_load = np.concatenate(list_data, axis = 0)

    return data_load #, norm_factor # KC 20171018


def export_data_to_jpg(data_train_gt, dir_numpy_compressed, \
                        index_sample_total=0, ext_data = 'jpg'):
    index_sample_accumuated = index_sample_total
    num_sample_in_data = data_train_gt.shape[0]
    if not os.path.exists(dir_numpy_compressed):
        os.mkdir(dir_numpy_compressed)
        print('create directory {0}'.format(dir_numpy_compressed))
    print('start to export data dimension {0} to {1} for index {2}',
          data_train_gt.shape, dir_numpy_compressed,
          index_sample_total)

    for i in xrange(num_sample_in_data):
        im_output = np.tile(data_train_gt[i,:], (1,1,3))
        filepath_jpg = os.path.join(dir_numpy_compressed,'{0}.{1}'.format(index_sample_accumuated, ext_data))
        sci.misc.imsave(filepath_jpg, im_output)
        index_sample_accumuated+=1

    print('exported data dimension {0}to {1} for index {2}',
          data_train_gt.shape, dir_numpy_compressed,
          [index_sample_total,index_sample_accumuated])
    return index_sample_accumuated
