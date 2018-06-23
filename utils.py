"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for pix2pix

def load_data(image_path, flip=True, is_test=False, data_type='npz'):
    img_A, img_B = load_image(image_path, data_type)
    img_A, img_B = preprocess_A_and_B(img_A, img_B, flip=flip, is_test=is_test)

    if data_type != 'npz':
        # print('not npz, use normalization')
        img_A = img_A/127.5 - 1.
        img_B = img_B/127.5 - 1.
    # else:   # 16 unit, [0, 32768]
    #     img_A = img_A/16384. - 1.
    #     img_B = img_B/16384. - 1.

    img_AB = np.concatenate((img_A, img_B), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB

def load_image(image_path, data_type):
    if data_type == 'npz':
        input_img = np.load(image_path)
        # print(image_path)
        img_A = input_img['input']
        # img_A = img_A[:,:,1:]
        img_B = input_img['output']
        img_A = 2 * img_A / np.amax(img_A) - 1
        img_B = 2 * img_B / np.amax(img_B) - 1
        # if img_A.shape[0] < 256:
            # img_A = scipy.misc.imresize(scipy.misc.toimage(img_A), [128, 128])
            # img_B = scipy.misc.imresize(scipy.misc.toimage(np.squeeze(img_B, axis=2)), [128, 128])
            # img_B = scipy.misc.imresize(scipy.misc.toimage(np.repmat(img_B, [1,1,3])), [256, 256])
            # img_A = img_A/127.5 - 1.
            # img_B = img_B/127.5 - 1.
            # img_B = np.reshape(img_B, (128,128,1))
            # img_B = np.reshape(img_B[:,:,0], [256, 256, 1])/127.5 - 1.
    else:                   # .jpg .png input and output concatenate
        input_img = imread(image_path)
        w = int(input_img.shape[1])
        w2 = int(w/2)
        img_B = input_img[:, 0:w2]
        img_A = input_img[:, w2:w]
    return img_A, img_B

def preprocess_A_and_B(img_A, img_B, flip=True, is_test=False):
    if not is_test:
        if flip and np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)

    return img_A, img_B

# -----------------------------

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, reals, size, image_path, data_type):
    return imsave(inverse_transform(images, data_type),
                    inverse_transform(reals, data_type), size, image_path, data_type)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, reals, size, data_type, path):
    images = inverse_transform(images, data_type)
    reals = inverse_transform(reals, data_type)
    if data_type == 'npz':
        h, w = images.shape[1], images.shape[2]
        for idx, image in enumerate(images):
            # save generatove image
            image = np.squeeze(image, axis=2)
            img_path = path[:-4] + '_' + str(idx) + '_output' + path[-4:]
            image_img = image
            # image_img = scipy.misc.bytescale(image*255, low=int(np.amin(image)*255.0), high=int(np.amax(image)*255.0))
            scipy.misc.imsave(img_path, image_img)
            # npz_path = path[:-4] + '_' + str(idx) + '_output.npy'
            # np.save(npz_path, image)

            # save input and target
            real = reals[idx]
            tmp = real[:,:,0]
            for i in range(1, reals.shape[3]-1):
                tmp = np.concatenate((tmp, real[:,:,i]), axis=1)
            img_path = path[:-4] + '_' + str(idx) + '_input' + path[-4:]
            tmp_img = tmp
            # tmp_img = scipy.misc.bytescale(tmp*255, low=int(np.amin(tmp)*255.0), high=int(np.amax(tmp)*255.0))
            scipy.misc.imsave(img_path, tmp_img)
            target = real[:,:,-1]
            img_path = path[:-4] + '_' + str(idx) + '_target' + path[-4:]
            target_img = target
            # target_img = scipy.misc.bytescale(target*255, low=int(np.amin(target)*255.0), high=int(np.amax(target)*255.0))
            scipy.misc.imsave(img_path, target_img)
            # npz_path = path[:-4] + '_' + str(idx) + '_target.npy'
            # np.save(npz_path, target)

            # save diff
            diff = abs(image - target)
            img_path = path[:-4] + '_' + str(idx) + '_diff' + path[-4:]
            diff_img = diff
            # diff_img = scipy.misc.bytescale(diff*255, low=int(np.amin(diff)*255.0), high=int(np.amax(diff)*255.0))
            scipy.misc.imsave(img_path, diff_img)
            # npz_path = path[:-4] + '_' + str(idx) + '_diff.npy'
            # np.save(npz_path, diff)
    else:
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1], 3))
        for idx, image in enumerate(images):
            img_path = path[:-4] + '_' + str(idx) + path[-4:]
            scipy.misc.imsave(img_path, image)

def imsave(images, reals, size, path, data_type):
    return merge(images, reals, size, data_type, path)

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
