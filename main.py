import argparse
import os
import scipy.misc
import numpy as np

from model import pix2pix
import tensorflow as tf

parser = argparse.ArgumentParser(description='')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='../data', help='name of the dataset')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--epochs', dest='epochs', type=int, default=200, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--sample_size', dest='sample_size', type=int, default=1, help='# sample images')
parser.add_argument('--input_size', dest='input_size', type=int, default=256, help='input image size')
parser.add_argument('--output_size', dest='output_size', type=int, default=256, help='output image size')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=4, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=1, help='# of output image channels')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
# parser.add_argument('--niter', dest='niter', type=int, default=200, help='# of iter at starting learning rate')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
# parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
# parser.set_defaults(flip=False)
# parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--save_epoch_freq', dest='save_epoch_freq', type=int, default=50, help='save a model every save_epoch_freq epochs (does not overwrite previously saved models)')
parser.add_argument("--save_best", dest="save_best", action="store_true", help="save only the best model, overwrite the previous models")
parser.set_defaults(save_best=False)
parser.add_argument('--print_freq', dest='print_freq', type=int, default=50, help='print the debug information every print_freq iterations')
parser.add_argument('--sample_freq', dest='sample_freq', type=int, default=100, help='test and save the sample result every sample_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', action="store_true", help='if continue training, load the latest model')
parser.set_defaults(continue_train=False)
# parser.add_argument('--serial_batches', dest='serial_batches', type=bool, default=False, help='f 1, takes images in order to make batches, otherwise takes them randomly')
# parser.add_argument('--serial_batch_iter', dest='serial_batch_iter', type=bool, default=True, help='iter into serial image list')
parser.add_argument('--data_type', dest='data_type', default='npz', help='type of data, jpg or png or npz')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=100.0, help='weight on L1 term in objective')

args = parser.parse_args()

def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    with tf.Session() as sess:
        model = pix2pix(sess, phase=args.phase, dataset_dir=args.dataset_dir,
                        checkpoint_dir=args.checkpoint_dir, sample_dir=args.sample_dir,
                        test_dir=args.test_dir, epochs=args.epochs, batch_size=args.batch_size,
                        sample_size=args.sample_size, input_size=args.input_size, output_size=args.output_size,
                        input_c_dim=args.input_nc, output_c_dim=args.output_nc, gf_dim=args.ngf,
                        df_dim=args.ndf, lr=args.lr, beta1=args.beta1, save_epoch_freq=args.save_epoch_freq,
                        save_best=args.save_best, print_freq=args.print_freq, sample_freq=args.sample_freq,
                        continue_train=args.continue_train, L1_lamb=args.L1_lambda, data_type=args.data_type)

        if args.phase == 'train':
            model.train()
        else:
            model.test()

if __name__ == '__main__':
    tf.app.run()
