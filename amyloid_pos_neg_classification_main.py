from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
import pdb
from six.moves import xrange
import argparse
from ops import *
from amyloid_pos_neg_classification import amyloid_pos_neg_classifier

parser = argparse.ArgumentParser(description='')
parser.add_argument('--gpu', dest='gpu', default='0', help='0,1,2,3')
parser.add_argument('--phase', dest='phase', default='train', help='train, test, test_subj, visualize')
parser.add_argument('--subj_id', dest='subj_id', default='1355')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='data/classification', help='name of the dataset')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint_classification', help='models are saved here')
parser.add_argument('--log_dir', dest='log_dir', default='./log_classification_new', help='logs are saved here')
parser.add_argument('--epochs', dest='epochs', type=int, default=50, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64, help='# images in batch')
parser.add_argument('--validation_split', dest='validation_split', type=float, default=0.1, help='random split validation set from training dataset')
parser.add_argument('--crop_size', dest='crop_size', type=int, default=224, help='crop image size')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=10, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', action="store_true", help='if continue training, load the latest model')
parser.set_defaults(continue_train=False)

args = parser.parse_args()

def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True


    with tf.Session(config=config) as sess:
        model = amyloid_pos_neg_classifier(sess, epochs=args.epochs, validation_split=args.validation_split,
                        dataset_dir=args.dataset_dir, log_dir=args.log_dir, checkpoint_dir=args.checkpoint_dir,
                        batch_size=args.batch_size, crop_size=args.crop_size, lr=args.lr, beta1=args.beta1,
                        print_freq=args.print_freq, continue_train=args.continue_train, phase=args.phase, subj_id=args.subj_id)

        if args.phase == 'train':
            model.train()
        elif args.phase == 'test':
            model.test()
        else:
            model.test_subj()

if __name__ == '__main__':
    tf.app.run()
