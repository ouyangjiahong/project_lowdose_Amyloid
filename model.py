from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
from tensorflow.contrib.slim.nets import vgg
import numpy as np
import pdb
from six.moves import xrange
from operator import mul
from functools import reduce

from ops import *
from utils import *

EPS = 1e-12

class pix2pix(object):

    def __init__(self, sess, phase, dataset_dir, validation_split=0.1,
                    task='lowdose', residual=False, is_gan=False, is_l1=False,
                    is_lc=False, is_ls=False, is_finetune=False, feat_match_dynamic=False,
                    checkpoint_dir=None, sample_dir=None, log_dir=None,
                    test_dir=None, epochs=200, batch_size=1, feat_match=False,
                    dimension=2, block=4, input_size=256, output_size=256,
                    input_c_dim=3, output_c_dim=1, gf_dim=64, g_times=1,
                    df_dim=64, lr=0.0002, beta1=0.5, save_epoch_freq=50,
                    save_best=False, print_freq=50, sample_freq=100, is_dicom=False,
                    continue_train=False, L1_lamb=100, c_lamb=100, s_lamb=100, data_type='npz'):

        """
        Args:
            sess: TensorFlow session
        """
        self.sess = sess
        self.task = task
        self.is_gan = is_gan
        self.is_l1 = is_l1
        self.is_lc = is_lc
        self.is_ls = is_ls
        self.feat_match = feat_match
        self.feat_match_dynamic = feat_match_dynamic
        self.is_finetune = is_finetune

        self.mode = ''
        if is_gan:
            self.mode += 'gan+'
        if is_l1:
            self.mode += 'l1+'
        if is_lc:
            self.mode += 'lc+'
        if is_ls:
            self.mode += 'ls+'
        if feat_match:
            self.mode += 'feat+'
        if feat_match_dynamic:
            self.mode += 'dynamic'
        if self.mode[-1] == '+':
            self.mode = self.mode[:-1]

        self.residual = residual
        self.dimension = dimension
        self.block = block

        # self.is_grayscale = False           # TODO: check whether for input or output
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size      # current code only support same size of in and out?

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        # self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim
        if task == 'lowdose':
            self.input_c_dim = 4
        elif task == 'zerodose':
            self.input_c_dim = input_c_dim
        else:       # petonly
            if self.dimension == 2.5:
                self.input_c_dim = 2*self.block + 1
            else:
                self.input_c_dim = 1


        self.epochs = epochs
        self.lr = lr
        self.beta1 = beta1
        self.L1_lamb = L1_lamb
        self.c_lamb = c_lamb
        self.s_lamb = s_lamb
        self.g_times = g_times
        self.validation_split = validation_split

        self.save_epoch_freq = save_epoch_freq
        self.save_best = save_best
        self.print_freq = print_freq
        self.sample_freq = sample_freq
        self.continue_train = continue_train
        self.data_type = data_type
        self.is_dicom = is_dicom

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.d_bn4 = batch_norm(name='d_bn4')

        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')

        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        self.g_bn_d6 = batch_norm(name='g_bn_d6')
        self.g_bn_d7 = batch_norm(name='g_bn_d7')

        self.dataset_dir = dataset_dir
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.test_dir = test_dir
        self.log_dir = log_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        path_tmp = os.path.join(self.sample_dir, self.task + '_' + self.mode)
        if not os.path.exists(path_tmp):
            os.makedirs(path_tmp)
        path_tmp = os.path.join(self.test_dir, self.task + '_' + self.mode)
        if not os.path.exists(path_tmp):
            os.makedirs(path_tmp)

        self.vgg = vgg.vgg_16
        self.build_model()

    def calculator(self):
        # calculate the perceptual loss using pre-trained VGG16 net
        if not(self.is_lc or self.is_ls):
            return tf.constant(0.0), tf.constant(0.0)

        # resize from 256 to 224 by central cropping
        real_B_224 = 128*tf.image.resize_image_with_crop_or_pad(self.real_B, 224, 224)
        real_B_224 = tf.tile(real_B_224, [1,1,1,3])
        fake_B_224 = 128*tf.image.resize_image_with_crop_or_pad(self.fake_B, 224, 224)
        fake_B_224 = tf.tile(fake_B_224, [1,1,1,3])

        # calculate the output of each layers for both real and fake image
        content_layer_dict = ['conv1/conv1_2', 'conv2/conv2_2', 'conv3/conv3_3', 'conv4/conv4_3']
        # content_weight = [1e7, 1e7, 1e10, 1e10]
        content_weight = [1, 1, 1, 1]
        style_layer_dict = ['conv1/conv1_2', 'conv2/conv2_2', 'conv3/conv3_3', 'conv4/conv4_3']
        style_weight = [1, 1, 1, 1]

        if self.is_finetune:
            pretrained_model_path = 'vgg_16_amyloid_ckpt/'
            num_cls = 2
        else:
            pretrained_model_path = 'vgg_16.ckpt'
            num_cls = 1000
        with tf.variable_scope("calculator") as scope:
            with tf.contrib.slim.arg_scope(vgg.vgg_arg_scope()):
                # load pretrained vgg16 models
                logits, layers_real = self.vgg(real_B_224, is_training=False, num_classes=num_cls, scope='real/vgg_16')
                tf.contrib.framework.init_from_checkpoint(pretrained_model_path, {'vgg_16/':'calculator/real/vgg_16/'})

                logits, layers_fake = self.vgg(fake_B_224, is_training=False, num_classes=num_cls, scope='fake/vgg_16')
                tf.contrib.framework.init_from_checkpoint(pretrained_model_path, {'vgg_16/':'calculator/fake/vgg_16/'})
                print('load VGG16 pretrained model')

                # content loss
                content_loss = tf.constant(0.0)
                style_loss = tf.constant(0.0)
                if self.is_lc:
                    for i, layer_name in enumerate(content_layer_dict):
                        layer_real = layers_real['calculator/real/vgg_16/'+layer_name]
                        layer_fake = layers_fake['calculator/fake/vgg_16/'+layer_name]
                        bs, h, w, c = map(lambda i: i.value, layer_real.get_shape())
                        size = bs * h * w * c
                        content_loss += content_weight[i] * tf.nn.l2_loss((layer_real - layer_fake) / float(size))

                # style Loss
                if self.is_ls:
                    # print(layers_fake)
                    for i, layer_name in enumerate(content_layer_dict):
                        layer_real = layers_real['calculator/real/vgg_16/'+layer_name]
                        bs, h, w, c = map(lambda i: i.value, layer_real.get_shape())
                        size = bs * h * w * c
                        layer_real = tf.reshape(layer_real, [bs, h * w, c])
                        gram_real = tf.matmul(tf.transpose(layer_real, perm=[0,2,1]), layer_real) / size
                        layer_fake = layers_fake['calculator/fake/vgg_16/'+layer_name]
                        layer_fake = tf.reshape(layer_fake, [bs, h * w, c])
                        gram_fake = tf.matmul(tf.transpose(layer_fake, perm=[0,2,1]), layer_fake) / size
                        style_loss += style_weight[i] * tf.nn.l2_loss(gram_real - gram_fake) / (4*c*c)

        return content_loss, style_loss

    def feature_matching(self):
        self.feat_match_loss = []
        for i in range(len(self.D_h_all)):
            self.D_hi_diff = self.D_h_all[i] - self.D_h_all_[i]
            # print(self.D_hi_diff)
            size = reduce(mul, (d.value for d in self.D_hi_diff.get_shape()), 1)
            self.feat_match_loss.append(tf.nn.l2_loss(self.D_hi_diff) / size)
        return self.feat_match_loss

    def build_model(self):
        self.real_data = tf.placeholder(tf.float32,
                                        [self.batch_size, self.input_size, self.input_size,
                                         self.input_c_dim + self.output_c_dim],
                                        name='real_A_and_B_images')

        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        self.fake_B = self.generator(self.real_A)

        self.real_AB = tf.concat([self.real_A, self.real_B], 3)
        self.fake_AB = tf.concat([self.real_A, self.fake_B], 3)
        self.D, self.D_logits, self.D_h_all = self.discriminator(self.real_AB, reuse=False)
        self.D_, self.D_logits_, self.D_h_all_ = self.discriminator(self.fake_AB, reuse=True)

        # self.fake_B_sample = self.sampler(self.real_A)
        self.fake_B_sample = self.generator(self.real_A, is_sampler=True)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.fake_B_sum = tf.summary.image("fake_B", self.fake_B)
        self.real_B_sum = tf.summary.image("real_B", self.real_B)

        # self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        # self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.d_loss_real = tf.reduce_mean(-tf.log(self.D + EPS))
        self.d_loss_fake = tf.reduce_mean(-tf.log(1 - self.D_ + EPS))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        # add feature matching here
        self.feature_matching()
        self.feat_match_flag_holder = tf.placeholder(tf.float32, [len(self.feat_match_loss)], name='feat_match_flag')

        if self.feat_match == False:
            # self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))
            self.g_loss = tf.reduce_mean(-tf.log(self.D_ + EPS))
        else:
            # self.g_loss = sum([self.feat_match_loss[3]])
            self.g_loss = sum([self.feat_match_flag_holder[i]*self.feat_match_loss[i] for i in range(len(self.feat_match_loss))])

        self.L1_loss = tf.reduce_mean(tf.abs(self.real_B - self.fake_B))
        self.content_loss, self.style_loss = self.calculator()

        # combination of each loss
        self.l1_lambda_holder = tf.placeholder(tf.float32)
        self.lc_lambda_holder = tf.placeholder(tf.float32)
        self.ls_lambda_holder = tf.placeholder(tf.float32)
        self.g_loss_all = 0
        if self.is_gan:
            self.g_loss_all += self.g_loss
        if self.is_l1:
            # self.g_loss_all += self.L1_lamb * self.L1_loss
            self.g_loss_all += self.l1_lambda_holder * self.L1_loss
        if self.is_lc:
            self.g_loss_all += self.lc_lambda_holder * self.content_loss
        if self.is_ls:
            self.g_loss_all += self.ls_lambda_holder * self.style_loss

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.g_loss_all_sum = tf.summary.scalar("g_loss_all", self.g_loss_all)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.L1_loss_sum = tf.summary.scalar("L1_loss", self.L1_loss)
        self.content_loss_sum = tf.summary.scalar("content_loss", self.content_loss)
        self.style_loss_sum = tf.summary.scalar("style_loss", self.style_loss)

        self.feat_match_loss_sum_in = tf.summary.scalar("feat/input", self.L1_loss)
        self.feat_match_loss_sum_h1 = tf.summary.scalar("feat/h1", self.feat_match_loss[0])
        self.feat_match_loss_sum_h2 = tf.summary.scalar("feat/h2", self.feat_match_loss[1])
        self.feat_match_loss_sum_h3 = tf.summary.scalar("feat/h3", self.feat_match_loss[2])
        self.feat_match_loss_sum_h4 = tf.summary.scalar("feat/h4", self.feat_match_loss[3])
        self.feat_match_loss_sum_num = tf.summary.scalar("feat/num", tf.reduce_sum(self.feat_match_flag_holder))

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=5)


    def load_random_samples(self):
        # data = np.random.choice(glob('./datasets/{}/val/*.jpg'.format(self.dataset_name)), self.batch_size)
        data = np.random.choice(glob('{}/test/*.{}'.format(self.dataset_dir, self.data_type)), self.batch_size)
        sample = [load_data(sample_file, data_type=self.data_type, task=self.task, \
                            dimension=self.dimension) for sample_file in data]
        sample = [b[0] for b in sample]
        sample_images = np.array(sample).astype(np.float32)
        return sample_images

    def sample_model(self, sample_dir, epoch, idx):
        sample_images = self.load_random_samples()
        samples, d_loss, g_loss_all, g_loss, L1_loss, content_loss, style_loss = self.sess.run(
            [self.fake_B_sample, self.d_loss, self.g_loss_all, self.g_loss, self.L1_loss, self.content_loss, self.style_loss],
            feed_dict={self.real_data: sample_images, self.l1_lambda_holder: self.l1_lamb_cur,
            self.lc_lambda_holder: self.lc_lamb_cur, self.ls_lambda_holder: self.ls_lamb_cur,
            self.feat_match_flag_holder:self.feat_match_flag}
        )
        save_images(samples, sample_images, [self.batch_size, 1],
                    './{}/{}_{}/train_{:02d}_{:04d}.jpg'.format(sample_dir, self.task, self.mode, epoch, idx),
                    data_type=self.data_type, is_stat=False)
        print("[Sample] d_loss: {:.8f}, g_loss_all: {:.8f}, g_loss: {:.8f}, L1_loss: {:.8f}, \
                content_loss: {:.8f}, style_loss: {:.8f}".format(d_loss, g_loss_all, g_loss, L1_loss, content_loss, style_loss))

    def train(self):
        """Train pix2pix"""
        d_optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1) \
                          .minimize(self.g_loss_all, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.g_sum = tf.summary.merge([self.d__sum, self.L1_loss_sum, self.content_loss_sum, self.style_loss_sum,
            self.fake_B_sum, self.real_B_sum, self.d_loss_fake_sum, self.g_loss_sum, self.g_loss_all_sum])
        self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.feat_match_sum = tf.summary.merge([self.feat_match_loss_sum_num, self.feat_match_loss_sum_in,
                                        self.feat_match_loss_sum_h1, self.feat_match_loss_sum_h2,
                                        self.feat_match_loss_sum_h3, self.feat_match_loss_sum_h4])
        file_path = self.log_dir + '/' + self.task + '_' + self.mode
        self.writer = tf.summary.FileWriter(file_path, self.sess.graph)

        counter = 0
        start_time = time.time()

        if self.continue_train == True and self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        L1_loss_best = 100
        self.l1_lamb_cur = self.L1_lamb
        self.lc_lamb_cur = 0
        self.ls_lamb_cur = 0
        if self.feat_match_dynamic:
            self.feat_match_flag = [0.0,0.0,0.0,0.0]
        else:
            self.feat_match_flag = [1.0,1.0,1.0,1.0]


        for epoch in xrange(self.epochs):
            # data = glob('./datasets/{}/train/*.jpg'.format(self.dataset_name))
            data = glob('{}/train/*.{}'.format(self.dataset_dir, self.data_type))
            np.random.shuffle(data)
            training_data_num = int((1 - self.validation_split) * len(data))
            training_data = data[:training_data_num]
            validation_data = data[training_data_num:]
            batch_idxs = len(training_data) // self.batch_size

            # if epoch > 0:
            #     self.lc_lamb_cur = self.c_lamb
            #     self.ls_lamb_cur = self.s_lamb

            for idx in xrange(0, batch_idxs):
                batch_files = training_data[idx*self.batch_size:(idx+1)*self.batch_size]
                batch = [load_data(batch_file, data_type=self.data_type, task=self.task, dimension=self.dimension) for batch_file in batch_files]
                batch = [b[0] for b in batch]
                batch_images = np.array(batch).astype(np.float32)

                # self.feat_match_flag = [1.0,1.0,1.0,1.0]
                self.feed_dict = {self.real_data:batch_images, self.l1_lambda_holder:self.l1_lamb_cur,
                self.lc_lambda_holder:self.lc_lamb_cur, self.ls_lambda_holder:self.ls_lamb_cur,
                self.feat_match_flag_holder:self.feat_match_flag}
                # Update D network
                _, summary_str_d = self.sess.run([d_optim, self.d_sum], feed_dict=self.feed_dict)

                # Update G network
                # Run g_optim g_times to make sure that d_loss does not go to zero
                for g_t in range(self.g_times):
                    _, summary_str_g, summary_str_feat = self.sess.run([g_optim, self.g_sum, self.feat_match_sum],
                                                feed_dict=self.feed_dict)

                counter += 1
                if counter % self.print_freq == 1:
                    errD_fake = self.d_loss_fake.eval(self.feed_dict)
                    errD_real = self.d_loss_real.eval(self.feed_dict)
                    errG = self.g_loss.eval(self.feed_dict)
                    errG_all = self.g_loss_all.eval(self.feed_dict)
                    errL1 = self.L1_loss.eval(self.feed_dict)
                    errC = self.content_loss.eval(self.feed_dict)
                    errS = self.style_loss.eval(self.feed_dict)

                    # pdb.set_trace()
                    if errC < 0.01:
                        if self.lc_lamb_cur == self.c_lamb / 10:
                            self.lc_lamb_cur = self.c_lamb
                    elif errC < 0.1:
                        self.lc_lamb_cur = self.c_lamb / 10
                    if errS < 1:
                        if self.ls_lamb_cur == self.s_lamb / 10:
                            self.ls_lamb_cur = self.s_lamb
                    elif errS < 10:
                        if self.ls_lamb_cur == self.s_lamb / 100:
                            self.ls_lamb_cur = self.s_lamb / 10
                    elif errS < 100:
                        if self.ls_lamb_cur == self.s_lamb / 1000:
                            self.ls_lamb_cur = self.s_lamb / 100
                    elif errS < 1000:
                        self.ls_lamb_cur = self.s_lamb / 1000
                    # print(self.lc_lamb_cur)
                    # print(self.ls_lamb_cur)


                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss_all: %.8f, \
                        g_loss: %.8f, L1_loss: %.8f, content_loss: %.8f, style_loss: %.8f" % (epoch, idx, batch_idxs,
                            time.time() - start_time, errD_fake+errD_real, errG_all, errG, errL1, errC, errS))

                    self.writer.add_summary(summary_str_d, counter)
                    self.writer.add_summary(summary_str_g, counter)
                    self.writer.add_summary(summary_str_feat, counter)

                if counter % self.sample_freq == 1:
                    self.sample_model(self.sample_dir, epoch, idx)

            # validate at the end of each epoch
            L1_loss_avg = self.validate(validation_data)
            if L1_loss_best > L1_loss_avg: # getting better model, save
                print("save best model!")
                self.save(self.checkpoint_dir, counter, is_best=True)
                L1_loss_best = L1_loss_avg

            elif epoch % self.save_epoch_freq == 0:
                self.save(self.checkpoint_dir, counter)

            # change self.feat_match_flag based on L1 loss average
            if self.feat_match_dynamic:
                if L1_loss_avg < 0.015:
                    print('add h4 layer')
                    self.feat_match_flag = [1.0,1.0,1.0,1.0]
                elif L1_loss_avg < 0.02:
                    print('add h3 layer')
                    self.feat_match_flag = [1.0,1.0,1.0,0.0]
                elif L1_loss_avg < 0.025:
                    print('add h2 layer')
                    self.feat_match_flag = [1.0,1.0,0.0,0.0]
                elif L1_loss_avg < 0.03:
                    print('add h1 layer')
                    self.feat_match_flag = [1.0,0.0,0.0,0.0]

    def discriminator(self, image, y=None, reuse=False):

        with tf.variable_scope("discriminator") as scope:

            # image is 256 x 256 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            # h2 is (32x 32 x self.df_dim*4)
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv')))
            # h3 is (32x 32 x self.df_dim*8)
            h4 = conv2d(h3, 1, d_h=1, d_w=1, name='d_h4_conv')
            # h4 is (32 x 32 x 1)
            h4_bn = lrelu(self.d_bn4(h4))
            # h4_bn = h4

            return tf.nn.sigmoid(h4), h4, [h1,h2,h3,h4_bn]

    def generator(self, image, y=None, is_sampler=False):
        with tf.variable_scope("generator") as scope:
            if is_sampler == True:          # use it instead of the sampler function
                scope.reuse_variables()

            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            # image is (256 x 256 x input_c_dim)
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            # e1 is (128 x 128 x self.gf_dim)
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
            # e2 is (64 x 64 x self.gf_dim*2)
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
            # e3 is (32 x 32 x self.gf_dim*4)
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
            # e4 is (16 x 16 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
            # e5 is (8 x 8 x self.gf_dim*8)
            e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
            # e6 is (4 x 4 x self.gf_dim*8)
            e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
            # e7 is (2 x 2 x self.gf_dim*8)
            e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv', center=True))
            # e8 is (1 x 1 x self.gf_dim*8)

            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                [self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True)
            # d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = self.g_bn_d1(self.d1)
            d1 = tf.concat([d1, e7], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
            # d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = self.g_bn_d2(self.d2)
            d2 = tf.concat([d2, e6], 3)
            # d2 is (4 x 4 x self.gf_dim*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
            # d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = self.g_bn_d3(self.d3)
            d3 = tf.concat([d3, e5], 3)
            # d3 is (8 x 8 x self.gf_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)
            # d6 is (64 x 64 x self.gf_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)
            # d7 is (128 x 128 x self.gf_dim*1*2)

            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                [self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True)
            # d8 is (256 x 256 x output_c_dim)

            if self.residual == True:
                return tf.nn.tanh(self.d8 + tf.expand_dims(image[:,:,:,0], 3))
            else:
                return tf.nn.tanh(self.d8)


    def save(self, checkpoint_dir, step, is_best=False):
        model_name = "pix2pix.model"
        if is_best == True:
            model_name = 'best.' + model_name
        model_dir = "%s_%s" % (self.task, self.mode)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        print("save model")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.task, self.mode)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def validate(self, sample_files):
        # load validation input
        print("Loading validation images ...")
        sample = [load_data(sample_file, is_test=True, data_type=self.data_type, \
                            task=self.task, dimension=self.dimension) for sample_file in sample_files]
        sample = [b[0] for b in sample]

        sample_images = np.array(sample).astype(np.float32)

        sample_images = [sample_images[i:i+self.batch_size]
                         for i in xrange(0, len(sample_images), self.batch_size)]
        sample_images = np.array(sample_images)

        L1_loss_counter = 0
        for i, sample_image in enumerate(sample_images):
            if sample_image.shape[0] < self.batch_size:
                break
            idx = i+1
            samples, L1_loss = self.sess.run(
                [self.fake_B_sample, self.L1_loss],
                feed_dict={self.real_data: sample_image, self.l1_lambda_holder: self.l1_lamb_cur,
                self.lc_lambda_holder: self.lc_lamb_cur, self.ls_lambda_holder: self.ls_lamb_cur,
                self.feat_match_flag_holder:self.feat_match_flag}
            )
            L1_loss_counter = L1_loss_counter + L1_loss
        L1_loss_avg = L1_loss_counter / idx
        print('average L1 Loss: ', L1_loss_avg)
        return L1_loss_avg


    def test(self):
        """Test pix2pix"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # sample_files = glob('./datasets/{}/val/*.jpg'.format(self.dataset_name))
        sample_files = glob('{}/test/*.{}'.format(self.dataset_dir, self.data_type))

        # sort testing input
        n = [int(i) for i in map(lambda x: x.split('/')[-1].split('.'+self.data_type)[0], sample_files)]
        sample_files = [x for (y, x) in sorted(zip(n, sample_files))]

        # load testing input
        print("Loading testing images ...")
        sample = [load_data(sample_file, is_test=True, data_type=self.data_type, \
                            task=self.task, dimension=self.dimension) for sample_file in sample_files]
        max_value = [b[1] for b in sample]
        sample = [b[0] for b in sample]

        sample_images = np.array(sample).astype(np.float32)

        sample_images = [sample_images[i:i+self.batch_size]
                         for i in xrange(0, len(sample_images), self.batch_size)]
        sample_images = np.array(sample_images)
        print(sample_images.shape)

        start_time = time.time()
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        input_stat_all = []
        output_stat_all = []
        for i, sample_image in enumerate(sample_images):
            if sample_image.shape[0] < self.batch_size:
                break
            idx = i+1
            print("sampling image ", idx)
            samples = self.sess.run(
                self.fake_B_sample,
                feed_dict={self.real_data: sample_image, self.l1_lambda_holder: self.L1_lamb,
                self.lc_lambda_holder: self.c_lamb, self.ls_lambda_holder: self.s_lamb,
                self.feat_match_flag_holder:[1.0,1.0,1.0,1.0]}
            )
            input_stat_list, output_stat_list= save_images(samples, sample_image, [self.batch_size, 1],
                        './{}/{}_{}/test_{:04d}.jpg'.format(self.test_dir, self.task, self.mode, idx),
                        data_type=self.data_type, is_stat=True, is_dicom=self.is_dicom,
                        max_value=max_value[self.batch_size*i:self.batch_size*(i+1)])
            input_stat_all.append(input_stat_list)
            output_stat_all.append(output_stat_list)
        input_stat_all = [y for x in input_stat_all for y in x]
        output_stat_all = [y for x in output_stat_all for y in x]
        print('input average metrics:', {k:np.nanmean([x[k] for x in input_stat_all]) for k in input_stat_all[0].keys()})
        print('prediction average metrics:', {k:np.nanmean([x[k] for x in output_stat_all]) for k in output_stat_all[0].keys()})
        print('input variance metrics:', {k:np.nanvar([x[k] for x in input_stat_all]) for k in input_stat_all[0].keys()})
        print('prediction variance metrics:', {k:np.nanvar([x[k] for x in output_stat_all]) for k in output_stat_all[0].keys()})

        if self.is_dicom:
            if self.dimension == 2.5:
                series_name = self.task + '_' + str(2*self.block+1) + 'block_' +self.mode
                dicom_path = 'dicom/'+series_name+'/'
                dict_path = str(2*self.block+1)+'block_test_subject_sample.npz'

            else:
                series_name = self.task + '_' +self.mode
                dicom_path = 'dicom/'+series_name+'/'
                dict_path = '2D_test_subject_sample.npz'
            header_path = '/data3/Amyloid/temp/'

            save_dicom(series_name, dicom_path, dict_path, self.dataset_dir, './{}/{}_{}/'.format(self.test_dir, self.task, self.mode), header_path, set='output', block=self.block)
            save_dicom(series_name, dicom_path, dict_path, self.dataset_dir, './{}/{}_{}/'.format(self.test_dir, self.task, self.mode), header_path, set='output', block=self.block)
