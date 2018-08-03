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

parser = argparse.ArgumentParser(description='')
parser.add_argument('--gpu', dest='gpu', default='0', help='0,1,2,3')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='data/classification', help='name of the dataset')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint_classification', help='models are saved here')
parser.add_argument('--log_dir', dest='log_dir', default='./log_classification', help='logs are saved here')
parser.add_argument('--epochs', dest='epochs', type=int, default=100, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64, help='# images in batch')
parser.add_argument('--validation_split', dest='validation_split', type=float, default=0.1, help='random split validation set from training dataset')
parser.add_argument('--crop_size', dest='crop_size', type=int, default=224, help='crop image size')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=20, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', action="store_true", help='if continue training, load the latest model')
parser.set_defaults(continue_train=False)

args = parser.parse_args()

class amyloid_pos_neg_classifier(object):
    def __init__(self, sess, dataset_dir, log_dir, checkpoint_dir, epochs=100, validation_split=0.1,
                    batch_size=64, crop_size=224, lr=0.0002, beta1=0.5,
                    print_freq=100, continue_train=False):
        self.sess = sess
        self.batch_size = batch_size
        self.epochs = epochs
        self.dataset_dir = dataset_dir
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.validation_split = validation_split
        self.crop_size = crop_size
        self.lr = lr
        self.beta1 = beta1
        self.print_freq = print_freq
        self.continue_train = continue_train

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.build_model()

    def build_model(self):
        # image and label
        self.input = tf.placeholder(tf.float32, [self.batch_size, self.crop_size, self.crop_size, 1],
                                    name='input')
        self.label = tf.placeholder(tf.float32)

        # loss
        self.output, self.output_logits, self.layer_feature = self.classifier(self.input)
        self.loss = tf.nn.l2_loss(self.output - self.label)

        # summary
        self.loss_sum = tf.summary.scalar("training loss", self.loss)

        # variables
        self.t_vars = tf.trainable_variables()

        # save model
        self.saver = tf.train.Saver(max_to_keep=3)

    def train(self):
        # optimizer
        optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1) \
                          .minimize(self.loss, var_list=self.t_vars)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # summary writer
        file_path = self.log_dir
        self.writer = tf.summary.FileWriter(file_path, self.sess.graph)

        # load model
        if self.continue_train == True and self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # load all data
        data = glob('{}/train/*.{}'.format(self.dataset_dir, 'npz'))
        np.random.shuffle(data)
        training_data_num = int((1 - self.validation_split) * len(data))
        training_data = data[:training_data_num]
        validation_data = data[training_data_num:]
        batch_idxs = len(training_data) // self.batch_size


        counter = 0
        start_time = time.time()
        loss_best = 10000
        # self.validate(validation_data)
        for epoch in xrange(self.epochs):
            np.random.shuffle(training_data)
            for idx in xrange(0, batch_idxs):
                # load data
                batch_files = training_data[idx*self.batch_size:(idx+1)*self.batch_size]
                batch = [self.load_data(batch_file) for batch_file in batch_files]
                images = [b[0] for b in batch]
                labels = [b[1] for b in batch]
                images = np.array(images).astype(np.float32)
                labels = np.reshape(np.array(labels), [self.batch_size, 1])

                _, summary_str = self.sess.run([optim, self.loss_sum],
                                        feed_dict={self.input:images, self.label:labels})

                counter += 1
                if counter % self.print_freq == 1:
                    loss = self.loss.eval(feed_dict={self.input:images, self.label:labels})
                    # output = self.output.eval(feed_dict={self.input:images, self.label:labels})
                    # pdb.set_trace()

                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f" % (epoch, idx, batch_idxs,
                            time.time() - start_time, loss))
                    self.writer.add_summary(summary_str, counter)

            # validate at the end of each epoch
            loss_avg = self.validate(validation_data)
            if loss_best > loss_avg: # getting better model, save
                print("save best model!")
                self.save(self.checkpoint_dir, counter, is_best=True)
                loss_best = loss_avg


    def classifier(self, image):
        filter_num = [64, 64, 128, 256, 512]
        kernel_size = [7, 3, 3, 3, 3]
        stride = [2, 1, 2, 2, 2]
        with tf.variable_scope("amyloid_classifier") as scope:
            assert tf.get_variable_scope().reuse == False

            # conv1, [224,224,1]->[56,56,64]
            with tf.variable_scope('conv1'):
                x = conv2d(image, filter_num[0], k_h=kernel_size[0], k_w=kernel_size[0],
                            d_h=stride[0], d_w=stride[0], name='conv_1')
                x = bn(x, name='bn_1')
                x = lrelu(x, name='lrelu_1')
                x = tf.nn.max_pool(x, [1,3,3,1], [1,2,2,1], 'SAME')
                conv1 = x

            # conv2, [56,56,64]->[56,56,64]
            with tf.variable_scope('conv2_1'):
                x = residual_block(x)
            with tf.variable_scope('conv2_2'):
                x = residual_block(x)
                conv2 = x

            # conv3, [56,56,64]->[28,28,128]
            with tf.variable_scope('conv3_1'):
                x = residual_block(x, output_dim=filter_num[2], stride=stride[2], is_first=True)
            with tf.variable_scope('conv3_2'):
                x = residual_block(x)
            conv3 = x

            # conv4, [28,28,128]->[14,14,256]
            with tf.variable_scope('conv4_1'):
                x = residual_block(x, output_dim=filter_num[3], stride=stride[3], is_first=True)
            with tf.variable_scope('conv4_2'):
                x = residual_block(x)
            conv4 = x

            # conv3, [14,14,256]->[7,7,512]
            with tf.variable_scope('conv5_1'):
                x = residual_block(x, output_dim=filter_num[4], stride=stride[4], is_first=True)
            with tf.variable_scope('conv5_2'):
                x = residual_block(x)
            conv5 = x

            # logits
            with tf.variable_scope('logits'):
                x = tf.reduce_mean(x, [1, 2])
                x = linear(x, 1)

            return tf.nn.sigmoid(x), x, [conv1, conv2, conv3, conv4, conv5]


    def validate(self, sample_files):
        # load validation input
        print("Loading validation images ...")
        sample = [self.load_data(sample_file) for sample_file in sample_files]
        labels = [b[1] for b in sample]
        sample = [b[0] for b in sample]
        sample = np.array(sample).astype(np.float32)

        sample_images = [sample[i:i+self.batch_size]
                         for i in xrange(0, len(sample), self.batch_size)]
        sample_images = np.array(sample_images)
        sample_labels = [labels[i:i+self.batch_size]
                         for i in xrange(0, len(sample), self.batch_size)]

        loss_counter = 0
        for i, sample_image in enumerate(sample_images):
            if sample_image.shape[0] < self.batch_size:
                break
            idx = i+1
            loss = self.sess.run(self.loss, feed_dict={self.input:sample_image, self.label:sample_labels[i]})
            loss_counter = loss + loss_counter
        loss_avg = loss_counter / idx
        print('average MSE Loss: ', loss_avg)
        return loss_avg

    def load_data(self, filename):
        data = np.load(filename)
        image = data['image']
        edge = (image.shape[0] - self.crop_size) // 2
        image = image[edge:edge+self.crop_size, edge:edge+self.crop_size, :]
        label = data['label'].tolist()
        return image, label

    def save(self, checkpoint_dir, step, is_best=False):
        model_name = "amyloid_resnet.model"
        if is_best == True:
            model_name = 'best.' + model_name
        print("save model")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True


    with tf.Session(config=config) as sess:
        model = amyloid_pos_neg_classifier(sess, epochs=args.epochs, validation_split=args.validation_split,
                        dataset_dir=args.dataset_dir, log_dir=args.log_dir, checkpoint_dir=args.checkpoint_dir,
                        batch_size=args.batch_size, crop_size=args.crop_size, lr=args.lr, beta1=args.beta1,
                        print_freq=args.print_freq, continue_train=args.continue_train)

        if args.phase == 'train':
            model.train()
        else:
            model.test()

if __name__ == '__main__':
    tf.app.run()
