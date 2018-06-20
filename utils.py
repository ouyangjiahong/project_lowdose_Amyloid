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
        print('not npz, use normalization')
        img_A = img_A/127.5 - 1.
        img_B = img_B/127.5 - 1.

    img_AB = np.concatenate((img_A, img_B), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB

def load_image(image_path, data_type):
    if data_type == 'npz':
        input_img = np.load(image_path)
        # print(image_path)
        img_A = input_img['input']
        img_B = input_img['output']
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
    if data_type == 'npz':
        h, w = images.shape[1], images.shape[2]
        for idx, image in enumerate(images):
            # save generatove image
            image = np.squeeze(image, axis=2)
            img_path = path[:-4] + '_' + str(idx) + path[-4:]
            scipy.misc.imsave(img_path, image)
            npz_path = path[:-4] + '_' + str(idx) + '.npy'
            np.save(npz_path, image)

            # save input and target
            real = reals[idx]
            tmp = real[:,:,0]
            for i in range(1, reals.shape[3]):
                tmp = np.concatenate((tmp, real[:,:,i]), axis=1)
            img_path = path[:-4] + '_' + str(idx) + '_input' + path[-4:]
            scipy.misc.imsave(img_path, tmp)
            npz_path = path[:-4] + '_' + str(idx) + '_input.npy'
            np.save(npz_path, tmp)
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
        return images
    else:
        return (images+1.)/2.
