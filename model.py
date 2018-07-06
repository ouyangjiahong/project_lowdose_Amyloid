from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
import pdb
from six.moves import xrange

from ops import *
from utils import *

EPS = 1e-12

class pix2pix(object):

    def __init__(self, sess, phase, dataset_dir, validation_split=0.1,
                    task='lowdose', mode='mix', residual=False,
                    checkpoint_dir=None, sample_dir=None,
                    test_dir=None, epochs=200, batch_size=1,
                    dimension=2, block=4, input_size=256, output_size=256,
                    input_c_dim=3, output_c_dim=1, gf_dim=64,
                    df_dim=64, lr=0.0002, beta1=0.5, save_epoch_freq=50,
                    save_best=False, print_freq=50, sample_freq=100,
                    continue_train=False, L1_lamb=100, data_type='npz'):

        """
        Args:
            sess: TensorFlow session
        """
        self.sess = sess
        self.task = task
        self.mode = mode
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
        # pdb.set_trace()
        if task == 'lowdose':
            self.input_c_dim = 4
        elif task == 'zerodose':
            self.input_c_dim = input_c_dim
        else:       # petonly
            if self.dimension == 2.5:
                self.input_c_dim = 2*self.block + 1
                print(self.input_c_dim)
                print("___________________________________")
            else:
                self.input_c_dim = 1


        self.epochs = epochs
        self.lr = lr
        self.beta1 = beta1
        self.L1_lamb = L1_lamb

        self.save_epoch_freq = save_epoch_freq
        self.save_best = save_best
        self.print_freq = print_freq
        self.sample_freq = sample_freq
        self.continue_train = continue_train
        self.data_type = data_type

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

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
        # self.dataset_name = dataset_dir
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.test_dir = test_dir
        self.validation_split = validation_split
        self.build_model()

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
        self.D, self.D_logits = self.discriminator(self.real_AB, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.fake_AB, reuse=True)

        # self.fake_B_sample = self.sampler(self.real_A)
        self.fake_B_sample = self.generator(self.real_A, is_sampler=True)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.fake_B_sum = tf.summary.image("fake_B", self.fake_B)
        self.real_B_sum = tf.summary.image("real_B", self.real_B)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))
        # self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_))) \
        #                 + self.L1_lamb * self.L1_loss
        # self.d_loss_real = tf.reduce_mean(-tf.log(self.D + EPS))
        # self.d_loss_fake = tf.reduce_mean(-tf.log(1 - self.D_ + EPS))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        # self.g_loss = tf.reduce_mean(-tf.log(self.D_ + EPS))
        self.L1_loss = tf.reduce_mean(tf.abs(self.real_B - self.fake_B))
        if self.mode == 'mix':
            self.g_loss_all = self.g_loss + self.L1_lamb * self.L1_loss
        else:
            self.g_loss_all = self.L1_loss

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.g_loss_all_sum = tf.summary.scalar("g_loss_all", self.g_loss_all)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.L1_loss_sum = tf.summary.scalar("L1_loss", self.L1_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=5)


    def load_random_samples(self):
        # data = np.random.choice(glob('./datasets/{}/val/*.jpg'.format(self.dataset_name)), self.batch_size)
        data = np.random.choice(glob('{}/test/*.{}'.format(self.dataset_dir, self.data_type)), self.batch_size)
        sample = [load_data(sample_file, data_type=self.data_type, task=self.task, \
                            dimension=self.dimension) for sample_file in data]
        sample_images = np.array(sample).astype(np.float32)
        return sample_images

    def sample_model(self, sample_dir, epoch, idx):
        sample_images = self.load_random_samples()
        samples, d_loss, g_loss_all, L1_loss = self.sess.run(
            [self.fake_B_sample, self.d_loss, self.g_loss_all, self.L1_loss],
            feed_dict={self.real_data: sample_images}
        )
        save_images(samples, sample_images, [self.batch_size, 1],
                    './{}/{}_{}/train_{:02d}_{:04d}.jpg'.format(sample_dir, self.task, self.mode, epoch, idx),
                    data_type=self.data_type, is_stat=False)
        print("[Sample] d_loss: {:.8f}, g_loss_all: {:.8f}, L1_loss: {:.8f}".format(d_loss, g_loss_all, L1_loss))

    def train(self):
        """Train pix2pix"""
        d_optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1) \
                          .minimize(self.g_loss_all, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.g_sum = tf.summary.merge([self.d__sum, self.L1_loss_sum,
            self.fake_B_sum, self.real_B_sum, self.d_loss_fake_sum, self.g_loss_sum, self.g_loss_all_sum])
        self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        file_path = './logs/' + self.task + '_' + self.mode
        self.writer = tf.summary.FileWriter(file_path, self.sess.graph)

        counter = 0
        start_time = time.time()

        if self.continue_train == True and self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        L1_loss_best = 100

        for epoch in xrange(self.epochs):
            # data = glob('./datasets/{}/train/*.jpg'.format(self.dataset_name))
            data = glob('{}/train/*.{}'.format(self.dataset_dir, self.data_type))
            np.random.shuffle(data)
            training_data_num = int((1 - self.validation_split) * len(data))
            training_data = data[:training_data_num]
            validation_data = data[training_data_num:]
            batch_idxs = len(training_data) // self.batch_size

            for idx in xrange(0, batch_idxs):
                batch_files = training_data[idx*self.batch_size:(idx+1)*self.batch_size]
                batch = [load_data(batch_file, data_type=self.data_type, task=self.task, dimension=self.dimension) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)

                # Update D network
                if self.mode == 'mix':
                    _, summary_str_d = self.sess.run([d_optim, self.d_sum],
                                                   feed_dict={ self.real_data: batch_images })

                # Update G network
                _, summary_str_g = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={ self.real_data: batch_images })

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                # _, summary_str_g = self.sess.run([g_optim, self.g_sum],
                #                                feed_dict={ self.real_data: batch_images })
                # self.writer.add_summary(summary_str, counter)

                counter += 1
                if counter % self.print_freq == 1:
                    errD_fake = self.d_loss_fake.eval({self.real_data: batch_images})
                    errD_real = self.d_loss_real.eval({self.real_data: batch_images})
                    errG = self.g_loss_all.eval({self.real_data: batch_images})
                    errL1 = self.L1_loss.eval({self.real_data: batch_images})

                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss_all: %.8f, L1_loss: %.8f" \
                        % (epoch, idx, batch_idxs,
                            time.time() - start_time, errD_fake+errD_real, errG, errL1))

                    if self.mode == 'mix':
                        self.writer.add_summary(summary_str_d, counter)
                    self.writer.add_summary(summary_str_g, counter)

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
            # h3 is (16 x 16 x self.df_dim*8)
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4

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
                feed_dict={self.real_data: sample_image}
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
                feed_dict={self.real_data: sample_image}
            )
            input_stat_list, output_stat_list = save_images(samples, sample_image, [self.batch_size, 1],
                        './{}/{}_{}/test_{:04d}.jpg'.format(self.test_dir, self.task, self.mode, idx),
                        data_type=self.data_type, is_stat=True)
            input_stat_all.append(input_stat_list)
            output_stat_all.append(output_stat_list)
        input_stat_all = [y for x in input_stat_all for y in x]
        output_stat_all = [y for x in output_stat_all for y in x]
        print('input average metrics:', {k:np.nanmean([x[k] for x in input_stat_all]) for k in input_stat_all[0].keys()})
        print('prediction average metrics:', {k:np.nanmean([x[k] for x in output_stat_all]) for k in output_stat_all[0].keys()})
        print('input variance metrics:', {k:np.nanvar([x[k] for x in input_stat_all]) for k in input_stat_all[0].keys()})
        print('prediction variance metrics:', {k:np.nanvar([x[k] for x in output_stat_all]) for k in output_stat_all[0].keys()})
