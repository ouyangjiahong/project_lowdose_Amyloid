import argparse
import os
import sys
import scipy.misc
import numpy as np

from model import pix2pix
import tensorflow as tf

parser = argparse.ArgumentParser(description='')
parser.add_argument('--gpu', dest='gpu', default='0', help='0,1,2,3')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--task', dest='task', default='lowdose', help='lowdose, zerodose, petonly')
parser.add_argument('--is_gan', dest='is_gan', action='store_true', help='use GAN or not, default:false')
parser.set_defaults(gan=False)
parser.add_argument('--is_l1', dest='is_l1', action='store_true', help='use L1 loss or not, default:false')
parser.set_defaults(is_l1=False)
parser.add_argument('--is_lc', dest='is_lc', action='store_true', help='use content loss or not, default:false')
parser.set_defaults(is_lc=False)
parser.add_argument('--is_ls', dest='is_ls', action='store_true', help='use style loss or not, default:false')
parser.set_defaults(is_ls=False)
parser.add_argument('--is_finetune', dest='is_finetune', action='store_true', help='use fine-tune VGG16 on amyloid updake, default:false')
parser.set_defaults(is_finetune=False)
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=100.0, help='weight on L1 term in objective')
parser.add_argument('--c_lambda', dest='c_lambda', type=float, default=1.0, help='weight on content loss term in objective')
parser.add_argument('--s_lambda', dest='s_lambda', type=float, default=1.0, help='weight on style loss term in objective')
parser.add_argument('--feat_match', dest='feat_match', action='store_true', help='use feature matching or not, default:false')
parser.set_defaults(feat_match=False)
parser.add_argument('--dimension', dest='dimension', type=float, default=2, help='2, 2.5, 3')
parser.add_argument('--block', dest='block', type=int, default=4, help='the input data contain 2*block+1 slices')
parser.add_argument("--residual", dest='residual', action="store_true", help="add residual learning or not, default:false")
parser.set_defaults(residual=False)
parser.add_argument('--dataset_dir', dest='dataset_dir', default='../data', help='name of the dataset')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--log_dir', dest='log_dir', default='./log', help='logs are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--epochs', dest='epochs', type=int, default=100, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=4, help='# images in batch')
parser.add_argument('--validation_split', dest='validation_split', type=float, default=0.1, help='random split validation set from training dataset')
parser.add_argument('--input_size', dest='input_size', type=int, default=256, help='input image size')
parser.add_argument('--output_size', dest='output_size', type=int, default=256, help='output image size')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=1, help='# of output image channels')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
# parser.add_argument('--niter', dest='niter', type=int, default=200, help='# of iter at starting learning rate')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--save_epoch_freq', dest='save_epoch_freq', type=int, default=20, help='save a model every save_epoch_freq epochs (does not overwrite previously saved models)')
parser.add_argument("--save_best", dest="save_best", action="store_true", help="save only the best model, overwrite the previous models")
parser.set_defaults(save_best=False)
parser.add_argument('--print_freq', dest='print_freq', type=int, default=100, help='print the debug information every print_freq iterations')
parser.add_argument('--sample_freq', dest='sample_freq', type=int, default=1000, help='test and save the sample result every sample_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', action="store_true", help='if continue training, load the latest model')
parser.set_defaults(continue_train=False)
parser.add_argument('--data_type', dest='data_type', default='npz', help='type of data, jpg or png or npz')
parser.add_argument('--g_times', dest='g_times', type=int, default=1, help='train how many times of G for training D once')
parser.add_argument('--is_dicom', dest='is_dicom', action="store_true", help='whether save dicom in testing phase')
parser.set_defaults(is_dicom=False)
args = parser.parse_args()

def main(_):
    # checking exceptions
    if args.phase == 'train' and not (args.is_gan or args.is_l1 or args.is_lc or args.is_ls):
        raise ValueError('Need to choose at least one loss objective')
    if args.feat_match and not args.is_gan:
        raise ValueError('Only can use feature matching when using GAN loss')

    # see README, need to download first
    sys.path.append("../models/research/slim")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True


    with tf.Session(config=config) as sess:
        model = pix2pix(sess, phase=args.phase, task=args.task, residual=args.residual,
                        is_gan=args.is_gan, is_l1=args.is_l1, is_lc=args.is_lc, is_ls=args.is_ls, is_finetune=args.is_finetune,
                        dataset_dir=args.dataset_dir, validation_split=args.validation_split, log_dir=args.log_dir,
                        checkpoint_dir=args.checkpoint_dir, sample_dir=args.sample_dir, feat_match=args.feat_match,
                        test_dir=args.test_dir, epochs=args.epochs, batch_size=args.batch_size, block=args.block,
                        dimension=args.dimension, input_size=args.input_size, output_size=args.output_size,
                        input_c_dim=args.input_nc, output_c_dim=args.output_nc, gf_dim=args.ngf, g_times=args.g_times,
                        df_dim=args.ndf, lr=args.lr, beta1=args.beta1, save_epoch_freq=args.save_epoch_freq,
                        save_best=args.save_best, print_freq=args.print_freq, sample_freq=args.sample_freq,
                        continue_train=args.continue_train, L1_lamb=args.L1_lambda, c_lamb=args.c_lambda, s_lamb=args.s_lambda,
                        data_type=args.data_type, is_dicom=args.is_dicom)

        if args.phase == 'train':
            model.train()
        else:
            model.test()

if __name__ == '__main__':
    tf.app.run()
