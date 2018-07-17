"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import dicom
import scipy.misc
import numpy as np
import pdb
import os
from glob import glob
from skimage.measure import compare_mse, compare_nrmse, compare_psnr, compare_ssim
from time import gmtime, strftime

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for pix2pix

def load_data(image_path, is_test=False, data_type='npz', task='lowdose', dimension=2):
    img_A, img_B, max_value_A, max_value_B = load_image(image_path, data_type, task, dimension)
    # img_A, img_B = preprocess_A_and_B(img_A, img_B, flip=flip, is_test=is_test)

    if data_type != 'npz':
        # print('not npz, use normalization')
        img_A = img_A/127.5 - 1.
        img_B = img_B/127.5 - 1.

    img_AB = np.concatenate((img_A, img_B), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB, [max_value_A, max_value_B]

def load_image(image_path, data_type, task, dimension):
    if data_type == 'npz':
        input_img = np.load(image_path)
        # print(image_path)
        img_A = input_img['input']
        img_B = input_img['output']
        # transfer data to [-1, 1]
        # using data only after F-norm
        if task == 'zerodose':
            img_A = img_A[:,:,1:]
        elif task == 'petonly':
            if dimension == 2:
                img_A = img_A[:,:,0]
                img_A = np.expand_dims(img_A, 2)

        # normalize by the max value of each slice (move to data preparation)
        if dimension == 2.5:
            idx = img_A.shape[2] // 2
        else:
            idx = 0
        for i in range(img_A.shape[2]):
            max_value = np.amax(img_A[:,:,i])
            if max_value == 0:
                max_value = 1
            img_A[:,:,i] = 2 * img_A[:,:,i] / max_value - 1
            if i == idx:
                max_value_A = max_value

        max_value_B = np.amax(img_B)
        img_B = 2 * img_B / max_value_B - 1

    else:                   # .jpg .png input and output concatenate
        input_img = imread(image_path)
        w = int(input_img.shape[1])
        w2 = int(w/2)
        img_B = input_img[:, 0:w2]
        img_A = input_img[:, w2:w]
    return img_A, img_B, max_value_A, max_value_B

def preprocess_A_and_B(img_A, img_B, flip=True, is_test=False):
    if not is_test:
        if flip and np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)
    return img_A, img_B

# -----------------------------

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, reals, size, image_path, data_type, is_stat, is_dicom=False, max_value=None):
    return merge(inverse_transform(images, data_type),
                    inverse_transform(reals, data_type), size, data_type, image_path, is_stat, is_dicom, max_value)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge(images, reals, size, data_type, path, is_stat=False, is_dicom=False, max_value=None):
    if data_type == 'npz':
        h, w = images.shape[1], images.shape[2]
        output_stat_list = []
        input_stat_list = []
        output_stat_rev_list = []
        input_stat_rev_list = []
        for idx, image in enumerate(images):
            # save generatove image
            lowdose = reals[idx][:,:,0]
            image = np.squeeze(image, axis=2)
            img_path = path[:-4] + '_' + str(idx) + '_output' + path[-4:]
            image_img = image
            image_img = scipy.misc.bytescale(image*255, low=int(np.amin(image)*255.0), high=int(np.amax(image)*255.0))
            scipy.misc.imsave(img_path, image_img)
            # npz_path = path[:-4] + '_' + str(idx) + '_output.npy'
            # np.save(npz_path, image)

            # save input and target
            real = reals[idx]
            tmp = real[:,:,0]
            if reals.shape[3] - 1 > 5:      # 2.5D input
                tmp = real[:,:,reals.shape[3]//2-1]
            for i in range(1, reals.shape[3]-1):
                tmp = np.concatenate((tmp, real[:,:,i]), axis=1)
            img_path = path[:-4] + '_' + str(idx) + '_input' + path[-4:]
            tmp_img = tmp
            # print(np.amin(tmp))
            # print(np.amax(tmp))
            tmp_img = scipy.misc.bytescale(tmp*255, low=int(np.amin(tmp)*255.0), high=int(np.amax(tmp)*255.0))
            scipy.misc.imsave(img_path, tmp_img)
            target = real[:,:,-1]
            img_path = path[:-4] + '_' + str(idx) + '_target' + path[-4:]
            target_img = target
            target_img = scipy.misc.bytescale(target*255, low=int(np.amin(target)*255.0), high=int(np.amax(target)*255.0))
            scipy.misc.imsave(img_path, target_img)

            # save diff
            diff = abs(image - target)
            img_path = path[:-4] + '_' + str(idx) + '_diff' + path[-4:]
            diff_img = diff
            diff_img = scipy.misc.bytescale(diff*255, low=int(np.amin(diff)*255.0), high=int(np.amax(diff)*255.0))
            scipy.misc.imsave(img_path, diff_img)
            # npz_path = path[:-4] + '_' + str(idx) + '_diff.npy'
            # np.save(npz_path, diff)

            if is_stat == True:
                input = real[:,:,0]
                output_stat = compare_stat(image, target, max_value[idx])
                output_stat_list.append(output_stat)
                input_stat = compare_stat(input, target, max_value[idx])
                input_stat_list.append(input_stat)

            # multiply by max_value, save npy first
            if is_dicom == True:
                # pdb.set_trace()
                npz_path = path[:-4] + '_' + str(idx) + '_target.npy'
                np.save(npz_path, max_value[idx][1]*target)
                npz_path = path[:-4] + '_' + str(idx) + '_output.npy'
                np.save(npz_path, max_value[idx][0]*image)

        return input_stat_list, output_stat_list
    else:
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1], 3))
        for idx, image in enumerate(images):
            img_path = path[:-4] + '_' + str(idx) + path[-4:]
            scipy.misc.imsave(img_path, image)

def compare_stat(im_pred, im_gt, max_value):
    im_pred = np.array(im_pred).astype(np.float).flatten()
    im_gt = np.array(im_gt).astype(np.float).flatten()
    mask=np.abs(im_gt.flatten())>0

    # check dimension
    assert(im_pred.flatten().shape==im_gt.flatten().shape)

    # NRMSE
    try:
        rmse_pred = compare_nrmse(im_gt, im_pred)
    except:
        rmse_pred = float('nan')

    # PSNR
    try:
        psnr_pred = compare_psnr(im_gt, im_pred)
    except:
        psnr_pred = float('nan')

    # ssim
    try:
        ssim_pred = compare_ssim(im_gt, im_pred)
        score_ismrm = sum((np.abs(im_gt.flatten()-im_pred.flatten())<0.1)*mask)/(sum(mask)+0.0)*10000
    except:
        ssim_pred = float('nan')
        score_ismrm = float('nan')

    return {'rmse':rmse_pred,'psnr':psnr_pred,'ssim':ssim_pred,'score_ismrm':score_ismrm}
    # return {'rmse':10*max_value[1]*rmse_pred,'psnr':10*np.log10(100*max_value[1])+psnr_pred,'ssim':ssim_pred,'score_ismrm':score_ismrm}


def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images, data_type):
    if data_type == 'npz':
        images = (images+1.)/2.
        return images
    else:
        return (images+1.)/2.

def save_dicom(dicom_path, dict_path, ori_path, dst_path, header_path, set='output', block=4):
    if not os.path.exists(dicom_path):
        os.makedirs(dicom_path)

    '''
    Read in dicoms for header info and sort
    '''

    # dcmlist = [f for f in os.listdir(result_path+subj_id+'/Full_Dose_40') if os.path.isfile(os.path.join(result_path+subj_id+'/Full_Dose_40', f))]
    #
    # import re
    # r = re.compile(r'\_sl(\d+)\.s')
    # def key_func(m):
    #     return int(r.search(m).group(1))
    #
    # dcmlist_sorted = sorted(dcmlist, key=key_func)

    sample_files = glob('{}/test/*.npz'.format(ori_path))
    num = [int(i) for i in map(lambda x: x.split('/')[-1].split('.npz')[0], sample_files)]
    sample_files = [x for (y, x) in sorted(zip(num, sample_files))]

    output_files = glob('{}/*{}.npy'.format(dst_path, set))
    output_files = sorted(output_files)
    if len(sample_files) != len(output_files):
        sample_files = sample_files[:len(output_files)]

    dict_sample_output = {sample_files[i]:output_files[i] for i in range(len(output_files))}
    list_subject_sample = np.load(dict_path)
    subject_list = list_subject_sample['subject_list']
    subject_sample_list = list_subject_sample['dict']   # sub_id, lowdose norm, fulldose norm
    idx = 0

    subj_num = len(subject_list)
    for id in range(subj_num):
        subj_id = subject_sample_list[idx][0]
        norm = subject_sample_list[idx][1]
        slices_list = []
        for i in range(idx, len(output_files)):
            if subject_sample_list[i][0] != subj_id:       # check str or int
                continue
            else:
                output = np.load(output_files[idx])
                slices_list.append(output)
                idx = idx + 1
        volume = np.stack(slices_list, axis=2)   # check shape
        # pdb.set_trace()
        volume = volume * norm
        # pdb.set_trace()

        # read header and save dicom
        subj_id = str(int(subj_id))
        if not os.path.exists(dicom_path+subj_id):
            os.makedirs(dicom_path+subj_id)
        if not os.path.exists(dicom_path+subj_id+'/'+set):
            os.makedirs(dicom_path+subj_id+'/'+set)
        for i in range(volume.shape[2]):
            im_pred = np.squeeze(volume[:,:,i]).T
            im_pred_rot = zip(*im_pred[::-1])
            im_pred = zip(*im_pred_rot[::-1])
            im_pred = np.asarray(im_pred)
            im_pred_flip = np.flip(im_pred, 1)
            header_name = header_path+subj_id+'/Full_Dose_40/_bin1_sl'+str(i+block+1)+'.sdcopen'
            testdcm = dicom.read_file(header_name)
            im_pred_fullrange = 100 * im_pred_flip / testdcm.RescaleSlope   # 100 for lowdose
            im_pred_fullrange[im_pred_fullrange < 0] = 0
            im_pred_fullrange[im_pred_fullrange > 32767] = 32767
            testdcm.PixelData = im_pred_fullrange.astype(np.int16).tostring()
            testdcm.save_as(dicom_path+subj_id+'/'+set+'/_bin1_sl'+str(i+block+1)+'.sdcopen')
            print(dicom_path+subj_id+'/'+set+'/_bin1_sl'+str(i+block+1)+'.sdcopen')
