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


class amyloid_pos_neg_classifier(object):
    def __init__(self, sess, dataset_dir='data/classification/', log_dir='log_classification/',
                    checkpoint_dir='checkpoint_classification/', epochs=100, validation_split=0.1,
                    batch_size=64, crop_size=224, lr=0.0002, beta1=0.5,
                    print_freq=100, continue_train=False, phase='train'):
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
        self.phase = phase

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.build_model()

    def build_model(self):
        # image and label
        self.input = tf.placeholder(tf.float32, [None, self.crop_size, self.crop_size, 1], name='input')
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
        print('finish building amyloid classifier')

    def train(self):
        # optimizer
        # self.phase = 'train'

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


    def classifier(self, image, is_extract=False):
        filter_num = [32, 32, 64, 128, 256]
        # filter_num = [64, 64, 128, 256, 512]
        kernel_size = [7, 3, 3, 3, 3]
        stride = [2, 1, 2, 2, 2]

        if self.phase == 'train':
            is_training = True
        else:
            is_training = False

        with tf.variable_scope("amyloid_classifier") as scope:
            # assert tf.get_variable_scope().reuse == False
            if is_extract == True:
                scope.reuse_variables()
            # conv1, [224,224,1]->[56,56,64]
            with tf.variable_scope('conv1'):
                x = conv2d(image, filter_num[0], k_h=kernel_size[0], k_w=kernel_size[0],
                            d_h=stride[0], d_w=stride[0], name='conv_1')
                x = bn(x, name='bn_1', is_training=is_training)
                x = lrelu(x, name='lrelu_1')
                x = tf.nn.max_pool(x, [1,3,3,1], [1,2,2,1], 'SAME')
                conv1 = x

            # conv2, [56,56,64]->[56,56,64]
            with tf.variable_scope('conv2_1'):
                x = residual_block(x, is_training=is_training)
            with tf.variable_scope('conv2_2'):
                x = residual_block(x, is_training=is_training)
                conv2 = x

            # conv3, [56,56,64]->[28,28,128]
            with tf.variable_scope('conv3_1'):
                x = residual_block(x, output_dim=filter_num[2], stride=stride[2], is_first=True, is_training=is_training)
            with tf.variable_scope('conv3_2'):
                x = residual_block(x, is_training=is_training)
            conv3 = x

            # conv4, [28,28,128]->[14,14,256]
            with tf.variable_scope('conv4_1'):
                x = residual_block(x, output_dim=filter_num[3], stride=stride[3], is_first=True, is_training=is_training)
            with tf.variable_scope('conv4_2'):
                x = residual_block(x, is_training=is_training)
            conv4 = x

            # conv5, [14,14,256]->[7,7,512]
            with tf.variable_scope('conv5_1'):
                x = residual_block(x, output_dim=filter_num[4], stride=stride[4], is_first=True, is_training=is_training)
            with tf.variable_scope('conv5_2'):
                x = residual_block(x, is_training=is_training)
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
        # pdb.set_trace()
        for i, sample_image in enumerate(sample_images):
            labels = sample_labels[i]
            idx = i+1
            # if sample_image.shape[0] < self.batch_size:
            labels = np.reshape(np.array(labels), [-1, 1])
            loss = self.sess.run(self.loss, feed_dict={self.input:sample_image, self.label:labels})
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

    def test(self):
        """Test classifier"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # load ckpt
        if self.load(self.checkpoint_dir):
            print(" [*] Load amyloid classifier SUCCESS")
        else:
            print(" [!] Load amyloid classifier failed...")

        # load data
        sample_files = glob('{}/test/*.{}'.format(self.dataset_dir, 'npz'))
        n = [int(i) for i in map(lambda x: x.split('/')[-1].split('.npz')[0], sample_files)]
        sample_files = [x for (y, x) in sorted(zip(n, sample_files))]

        print("Loading testing images ...")
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
        label_list = []
        output_list = []
        for i, sample_image in enumerate(sample_images):
            # if sample_image.shape[0] < self.batch_size:
            #     break
            idx = i+1
            labels = sample_labels[i]
            labels = np.reshape(np.array(labels), [-1, 1])
            loss, output = self.sess.run([self.loss, self.output], feed_dict={self.input:sample_image, self.label:labels})
            loss_counter = loss + loss_counter
            label_list.append(labels)
            output_list.append(output)
        loss_avg = loss_counter / idx
        print('average MSE Loss: ', loss_avg)

        # save result
        label_list = np.concatenate(label_list, axis=0)
        output_list = np.concatenate(output_list, axis=0)
        with open('classification_test.npz','w') as file_input:
            np.savez_compressed(file_input, label=label_list, output=output_list)

        res = np.concatenate([label_list, output_list, abs(label_list - output_list)], axis=1)
        np.savetxt('classification_test.txt', res, fmt='%3f', delimiter='   ')
        print('average MAE Loss: ', np.mean(abs(label_list - output_list)))

    def feature_extract(self, images):
        """feature extractor"""
        # TODO: check shape
        output, _, layer_feature = self.classifier(images, is_extract=True)

        return output, layer_feature
