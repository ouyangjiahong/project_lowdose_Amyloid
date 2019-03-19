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
import skimage
import skimage.io
import skimage.transform
import skimage.color
import scipy as sci
import matplotlib as mpl
import nibabel as nib


class amyloid_pos_neg_classifier(object):
    def __init__(self, sess, dataset_dir='data/classification/', log_dir='log_classification/',
                    checkpoint_dir='checkpoint_classification/', epochs=100, validation_split=0.1,
                    batch_size=64, crop_size=224, lr=0.0002, beta1=0.5,
                    print_freq=100, continue_train=False, phase='train', subj_id='1355'):
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
        self.subj_id = subj_id

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.build_model()

    def build_model(self):
        # image and label
        self.input = tf.placeholder(tf.float32, [None, self.crop_size, self.crop_size, 1], name='input')
        self.label = tf.placeholder(tf.float32, [None, 1], name='label')

        # loss
        self.output, self.output_logits, self.layer_feature = self.classifier(self.input)
        self.grad_list = self.compute_gradient()

        # pdb.set_trace()
        # TODO: binary cross entropy
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.output_logits))
        # old MSE loss for regression
        # self.loss = tf.nn.l2_loss(self.output - self.label)

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
                # pdb.set_trace()

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
        # pdb.set_trace()
        if label > 0:
            label = 1
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
        # TODO only test on small set
        # sample_files = sample_files[:50]

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
        heatmap_list = np.empty([0, 4, self.crop_size, self.crop_size])
        for i, sample_image in enumerate(sample_images):
            # if sample_image.shape[0] < self.batch_size:
            #     break
            idx = i+1
            labels = sample_labels[i]
            labels = np.reshape(np.array(labels), [-1, 1])
            # TODO: check here
            loss, output, grad_list, feat_list = self.sess.run([self.loss, self.output, self.grad_list, self.layer_feature], feed_dict={self.input:sample_image, self.label:labels})
            loss_counter = loss + loss_counter
            # pdb.set_trace()
            label_list.append(labels)
            output_list.append(output)
            heatmap = self.compute_heatmap(grad_list, feat_list)
            heatmap_list = np.concatenate([heatmap_list, heatmap], axis=0)
        # pdb.set_trace()
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

        self.visualization(sample, label_list, output_list, heatmap_list)

    def test_subj(self):
        """Test classifier"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # load ckpt
        if self.load(self.checkpoint_dir):
            print(" [*] Load amyloid classifier SUCCESS")
        else:
            print(" [!] Load amyloid classifier failed...")

        # load data
        sample_files = glob('{}/test_subj/{}/*.{}'.format(self.dataset_dir, self.subj_id, 'npz'))
        n = [int(i) for i in map(lambda x: x.split('/')[-1].split('.npz')[0], sample_files)]
        sample_files = [x for (y, x) in sorted(zip(n, sample_files))]
        # TODO only test on small set
        # sample_files = sample_files[:50]

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

        # pdb.set_trace()

        loss_counter = 0
        label_list = []
        output_list = []
        heatmap_list = np.empty([0, 4, self.crop_size, self.crop_size])

        for i, sample_image in enumerate(sample_images):
            # if sample_image.shape[0] < self.batch_size:
            #     break
            idx = i+1
            labels = sample_labels[i]
            labels = np.reshape(np.array(labels), [-1, 1])
            # TODO: check here
            loss, output, grad_list, feat_list = self.sess.run([self.loss, self.output, self.grad_list, self.layer_feature], feed_dict={self.input:sample_image, self.label:labels})
            loss_counter = loss + loss_counter
            # pdb.set_trace()
            label_list.append(labels)
            output_list.append(output)
            heatmap = self.compute_heatmap(grad_list, feat_list)
            heatmap_list = np.concatenate([heatmap_list, heatmap], axis=0)
        # pdb.set_trace()
        loss_avg = loss_counter / idx
        print('average MSE Loss: ', loss_avg)

        # save result
        # pdb.set_trace()
        # sci.misc.imsave('tmp_input.jpg', sample_image[0,:])
        label_list = np.concatenate(label_list, axis=0)
        output_list = np.concatenate(output_list, axis=0)
        # with open('classification_test.npz','w') as file_input:
        #     np.savez_compressed(file_input, label=label_list, output=output_list)
        #
        # res = np.concatenate([label_list, output_list, abs(label_list - output_list)], axis=1)
        # np.savetxt('classification_test.txt', res, fmt='%3f', delimiter='   ')
        # print('average MAE Loss: ', np.mean(abs(label_list - output_list)))

        # self.visualization(sample, label_list, output_list, heatmap_list)
        # pdb.set_trace()
        # heatmap_list = np.rot90(heatmap_list, axes=(2,3))
        # heatmap_list = heatmap_list / np.max(heatmap_list)
        heatmap_pad1 = np.zeros([25, 4, self.crop_size, self.crop_size])
        heatmap_pad2 = np.zeros([24, 4, self.crop_size, self.crop_size])
        heatmap = np.concatenate([heatmap_pad1, heatmap_list, heatmap_pad2], axis=0)
        heatmap = np.transpose(heatmap, [1,2,3,0])
        heatmap = np.pad(heatmap, ((0,0),(16,16),(16,16),(0,0)), 'constant', constant_values=0)
        heatmap4 = heatmap[3,:,:,:]
        # heatmap4 = np.rot90(heatmap4, axes=(0,1))
        heatmap3 = heatmap[2,:,:,:]
        # heatmap3 = np.rot90(heatmap3, axes=(0,1))
        heatmap2 = heatmap[1,:,:,:]
        # heatmap2 = np.rot90(heatmap2, axes=(0,1))
        # pdb.set_trace()
        # sci.misc.imsave('tmp_heatmap.jpg', heatmap4[:,:,-25])

        # save nifti
        nifti_path = 'visualization_subj/'
        if not os.path.exists(nifti_path):
            os.makedirs(nifti_path)
        img = nib.Nifti1Image(heatmap2, np.eye(4))
        nib.save(img, nifti_path+self.subj_id+'_2.nii')
        img = nib.Nifti1Image(heatmap3, np.eye(4))
        nib.save(img, nifti_path+self.subj_id+'_3.nii')
        img = nib.Nifti1Image(heatmap4, np.eye(4))
        nib.save(img, nifti_path+self.subj_id+'_4.nii')
        print('save visualization!')

    def compute_gradient(self):
        # positive
        yc = self.output_logits
        grad_list = []
        for feature in self.layer_feature:
            grad = tf.gradients(yc, feature)[0]
            print('grad: ', grad)
            grad_list.append(grad)
        # pdb.set_trace()
        return grad_list

    def compute_heatmap(self, grad_list, feat_list):
        # in numpy, return [batch_size, selected_layer, 224, 224]
        layer_select = [1, 2, 3, 4] # conv4 and conv5
        heatmaps_list = []
        for i in layer_select:
            grad = grad_list[i]
            feat = feat_list[i]
            batch_size = grad.shape[0]
            alpha = np.mean(grad, axis=(1,2))
            heatmaps = np.array([alpha[j] * feat[j,:,:,:] for j in range(batch_size)])
            # pdb.set_trace()
            heatmaps = np.sum(heatmaps, axis=3)
            heatmaps = np.maximum(heatmaps, 0)
            max_value = np.max(heatmaps, axis=(1,2))
            heatmaps = np.array([heatmaps[j,:,:] / max_value[j] for j in range(batch_size)])
            heatmaps = skimage.transform.resize(heatmaps, (batch_size, self.crop_size, self.crop_size), preserve_range=True)
            # pad_size = (256 - self.crop_size) // 2
            # heatmaps_pad = np.pad(heatmaps, ((0,0),(pad_size, pad_size), (pad_size, pad_size)), 'constant', constant_values=0)
            heatmaps_list.append(heatmaps)
        # pdb.set_trace()
        heatmaps_list = np.array(heatmaps_list)
        heatmaps_list = np.swapaxes(heatmaps_list, 0, 1)
        return heatmaps_list


    def feature_extract(self, images):
        """feature extractor"""
        # TODO: check shape
        output, _, layer_feature = self.classifier(images, is_extract=True)

        return output, layer_feature

    def visualization(self, input_list, label_list, output_list, heatmap_list):
        vis_path = 'visualization/'
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)

        cm_jet = mpl.cm.get_cmap('jet')
        map_num = 4

        for i in range(len(input_list)):
            img = input_list[i].squeeze()
            img = np.rot90(img, axes=(0,1))
            img = img / np.max(img)
            label = label_list[i]
            output = output_list[i]
            heatmap = heatmap_list[i]
            heatmap = np.rot90(heatmap, axes=(1,2))
            heatmap = np.concatenate([heatmap[j,:,:] for j in range(map_num)], axis=1)

            # path = vis_path + str(i).zfill(4) + '_input_' + str(label) + '_' + str(output) + '.jpg'
            # sci.misc.imsave(path, img)
            path = vis_path + str(i).zfill(4) + '_heatmap_' + str(label) + '_' + str(output) + '.jpg'
            # pdb.set_trace()
            img = np.concatenate([img for j in range(map_num)], axis=1)
            img_hsv = skimage.color.rgb2hsv(np.dstack((img, img, img)))
            heatmap_rgba = cm_jet(heatmap)
            heatmap_hsv = skimage.color.rgb2hsv(heatmap_rgba[:,:,:3])
            img_hsv[..., 0] = heatmap_hsv[..., 0]
            img_hsv[..., 1] = heatmap_hsv[..., 1] * 0.5
            img_heatmap = skimage.color.hsv2rgb(img_hsv)
            sci.misc.imsave(path, img_heatmap)
