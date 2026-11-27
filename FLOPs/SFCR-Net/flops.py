import time
import cv2 as cv

from operation import ConvBlock, ResBlock, Up, Compute_z
from tensorflow.keras import layers, models
from ops import *
from loss.loss import *
import tools.rgb_lab_formulation as Conv_img

class T_CNN(object):
    def __init__(self, sess, image_height, image_width, label_shape,
                 batch_size, c_dim, data, labels_dir, checkpoint_dir, sample_dir, test_image_name, test_wb_name, test_ce_name, test_gc_name, id):
        self.sess = sess
        self.image_height = image_height
        self.image_width = image_width
        self.label_shape = label_shape
        self.batch_size = batch_size
        self.c_dim = c_dim
        self.data = data
        self.labels_dir = labels_dir
        self.checkpoint_dir = checkpoint_dir

        self.sample_dir = sample_dir
        self.test_image_name = test_image_name
        self.test_wb_name = test_wb_name
        self.test_ce_name = test_ce_name
        self.test_gc_name = test_gc_name
        self.id = id

        self.build_model()

    def build_model(self):
        self.images = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images')
        self.labels_image = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='labels_image')
        self.labels_gray = tf.image.rgb_to_grayscale(self.labels_image)

        self.pred_h1, self.gray_out= self.model()
        '''
        loss
        '''
        self.loss_vgg = loss_vgg(self.labels_image, self.pred_h1)
        self.loss_l1 = loss_l1(self.labels_image, self.pred_h1)
        self.loss_ssim = ssim_loss(self.labels_image, self.pred_h1)

        self.loss_ssim_gray = ssim_loss(self.labels_gray, self.gray_out)
        self.loss_edge = edge_loss(self.labels_gray, self.gray_out)

        self.loss = 0.05 * self.loss_vgg + self.loss_l1 + 3 * self.loss_ssim + self.loss_ssim_gray + self.loss_edge

        #获取所有变量
        t_vars = tf.compat.v1.trainable_variables()
        # 计算参数数量
        total_params = 0
        for var in t_vars:
            shape = var.get_shape()
            param_count = 1
            for dim in shape:
                param_count *= dim
            total_params += param_count
        print(f"Total Parameters: {total_params / 1000000}")

        self.saver = tf.compat.v1.train.Saver(max_to_keep=0)

    def train(self, config):
        if config.is_train:
            print(">>>开始训练")
        else:
            print(">>>开始测试")

        '''
        设置优化器
        初始化变量
        '''
        self.train_op = tf.compat.v1.train.AdamOptimizer(config.learning_rate,0.9).minimize(self.loss)
        tf.compat.v1.global_variables_initializer().run()

        counter = 0
        yc = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
            print(" >>>加载模型成功，继续训练")
        else:
            print(" [!] Load FAILED")
            print(" >>>加载模型失败，重新训练")

        #训练training
        if config.is_train:
            print("Training...")
            loss = np.ones(config.epoch)

            for ep in range(config.epoch):
                data_train_list = prepare_data(self.data, dataset="input_train")
                image_train_list = prepare_data(self.data, dataset="gt_train")

                seed = 500 + ep
                np.random.seed(seed)
                np.random.shuffle(data_train_list)
                np.random.seed(seed)
                np.random.shuffle(image_train_list)

                batch_idxs = len(data_train_list) // config.batch_size
                loss_train = np.ones(batch_idxs)
                loss_train1 = np.ones(batch_idxs)
                loss_train2 = np.ones(batch_idxs)
                loss_train3 = np.ones(batch_idxs)
                loss_train4 = np.ones(batch_idxs)
                loss_train5 = np.ones(batch_idxs)

                print("Epoch:",yc)

                for idx in range(0, batch_idxs):
                    batch_files = data_train_list[idx*config.batch_size:(idx+1)*config.batch_size]
                    batch_image_files = image_train_list[idx*config.batch_size : (idx+1)*config.batch_size]

                    batch_input = [get_image(batch_file, self.image_height, self.image_width) for batch_file in batch_files]
                    batch_image_input = [get_image(batch_image_file, self.image_height, self.image_width) for batch_image_file in batch_image_files]

                    counter += 1
                    _, err, err1, err2, err3, err4, err5 = self.sess.run([self.train_op, self.loss, self.loss_ssim, self.loss_l1, self.loss_vgg, self.loss_ssim_gray, self.loss_edge],
                                                             feed_dict={self.images: batch_input,
                                                                        self.labels_image:batch_image_input})

                    progress = (counter) / batch_idxs * 100
                    print(f'\rProgress: [{"=" * int(counter / 2)}{" " * int((batch_idxs - (counter + 1)) / 2)}] {progress:.2f}%', end='',
                          flush=True)
                    loss_train[idx] = err
                    loss_train1[idx] = err1
                    loss_train2[idx] = err2
                    loss_train3[idx] = err3
                    loss_train4[idx] = err4
                    loss_train5[idx] = err5
                loss1 = np.mean(loss_train)
                loss2 = np.mean(loss_train1)
                loss3 = np.mean(loss_train2)
                loss4 = np.mean(loss_train3)
                loss5 = np.mean(loss_train4)
                loss6 = np.mean(loss_train5)
                print()
                print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]，loss_ssim: [%.8f]，loss_h1: [%.8f]， loss_p1: [%.8f], loss_gray_ssim: [%.8f], loss_edge: [%.8f]" \
                    % ((ep + 1), counter, time.time() - start_time, loss1, loss2, loss3, loss4, loss5, loss6))
                loss[ep]=np.mean(loss_train)
                counter = 0
                yc += 1
                self.save(config.checkpoint_dir, yc-1)
                '''
                #labels
                if yc % 10 == 0:
                    label_train_list = prepare_data(self.labels_dir, dataset="input_labels")
                    label_gt_train_list = prepare_data(self.labels_dir, dataset="input_gt_labels")

                    batch_labels_idxs = len(label_train_list) // config.batch_size
                    loss_label0 = np.ones(batch_labels_idxs)
                    loss_label1 = np.ones(batch_labels_idxs)
                    loss_label2 = np.ones(batch_labels_idxs)
                    loss_label3 = np.ones(batch_labels_idxs)

                    for idx in range(0, batch_labels_idxs):
                        batch_files = label_train_list[idx * config.batch_size:(idx + 1) * config.batch_size]
                        batch_image_files = label_gt_train_list[idx * config.batch_size: (idx + 1) * config.batch_size]

                        batch_input = [get_image(batch_file, self.image_height, self.image_width) for batch_file in
                                       batch_files]
                        batch_image_input = [get_image(batch_image_file, self.image_height, self.image_width) for
                                             batch_image_file in batch_image_files]

                        err, err1, err2, err3 = self.sess.run(
                            [self.loss, self.loss_ssim, self.loss_l1, self.loss_vgg],
                            feed_dict={self.images: batch_input,
                                       self.labels_image: batch_image_input})

                        loss_label0[idx] = err
                        loss_label1[idx] = err1
                        loss_label2[idx] = err2
                        loss_label3[idx] = err3
                    label0 = np.mean(loss_label0)
                    label1 = np.mean(loss_label1)
                    label2 = np.mean(loss_label2)
                    label3 = np.mean(loss_label3)
                    print(
                        "Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]，loss_ssim: [%.8f]，loss_h1: [%.8f]， loss_p1: [%.8f]" \
                        % ((ep + 1), counter, time.time() - start_time, label0, label1, label2, label3))
                    '''
            print(loss)
            self.count_flops()
        else:
            print("Testing...")
            shape0 = (self.image_height, self.image_width, self.c_dim)
            image_testm = cv.imread(self.test_image_name, cv.IMREAD_COLOR)
            image_testm = cv.resize(image_testm, (shape0[0], shape0[1]))
            image_test = cv.cvtColor(image_testm, cv.COLOR_BGR2RGB) / 255
            expand_test = image_test[np.newaxis, :, :, :]
            expand_zero = np.zeros([self.batch_size - 1, shape0[0], shape0[1], shape0[2]])
            batch_test_image = np.append(expand_test, expand_zero, axis=0)

            start_time = time.time()
            output_name2 = os.path.basename(self.test_image_name)
            print(output_name2)
            result_h = self.sess.run(self.pred_h1,
                                     feed_dict={self.images: batch_test_image})
            all_time = time.time()
            final_time = all_time - start_time
            print(final_time)
            self.count_flops()

            _, h, w, c = result_h.shape
            for id in range(0, 1):
                result_h0 = result_h[id, :, :, :].reshape(h, w, 3)
                result_h0 = result_h0.squeeze()
                image_path0 = os.path.join(os.getcwd(), "enhancement")
                output_name = os.path.basename(self.test_image_name)
                print(image_path0, output_name)
                image_path = os.path.join(image_path0, output_name)
                #imsave_lable(result_h0, image_path)
                image_save(result_h0,image_path)
                print(image_path)



    def model(self):

        with tf.compat.v1.variable_scope("main_branch") as scope3:
            gray_image = tf.image.rgb_to_grayscale(self.images)
            print(gray_image.shape)
            #encoder
            conv1_gray = tf.nn.relu(conv2d(gray_image, 1, 64, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_gray'))
            conv2_gray = tf.nn.relu(conv2d(conv1_gray, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_gray'))
            conv3_gray = conv2d(conv2_gray, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_gray')
            conv4_gray = tf.add(conv1_gray, conv3_gray)
            print(conv4_gray.shape)
            graypooling = max_pool_2x2(conv4_gray)
            conv5_gray = tf.nn.relu(conv2d(graypooling, 64, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='conv5_gray'))
            conv6_gray = tf.nn.relu(conv2d(conv5_gray, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='conv6_gray'))
            conv7_gray = conv2d(conv6_gray, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='conv7_gray')
            conv8_gray = tf.add(conv5_gray, conv7_gray)
            print(conv8_gray.shape)
            graypooling1 = max_pool_2x2(conv8_gray)
            conv9_gray = tf.nn.relu(conv2d(graypooling1, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='conv9_gray'))
            conv10_gray = tf.nn.relu(conv2d(conv9_gray, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='conv10_gray'))
            conv11_gray = conv2d(conv10_gray, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='conv11_gray')
            conv12_gray = tf.add(conv9_gray, conv11_gray)
            print(conv12_gray.shape)
            #decoder
            conv13_gray = tf.nn.relu(conv2d(conv12_gray, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='conv13_gray'))
            conv14_gray = tf.nn.relu(conv2d(conv13_gray, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='conv14_gray'))
            conv15_gray = conv2d(conv14_gray, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='conv15_gray')
            conv16_gray = tf.add(conv13_gray, conv15_gray)
            decoder1 = tf.image.resize(conv16_gray, [graypooling.get_shape().as_list()[1],
                                                           graypooling.get_shape().as_list()[2]],
                                             method=tf.image.ResizeMethod.BILINEAR)
            print(decoder1.shape)
            conv17_gray = tf.nn.relu(conv2d(decoder1, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='conv17_gray'))
            conv18_gray = tf.nn.relu(conv2d(conv17_gray, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='conv18_gray'))
            conv19_gray = conv2d(conv18_gray, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='conv19_gray')
            conv20_gray = tf.add(conv17_gray, conv19_gray)
            decoder2 = tf.image.resize(conv20_gray, [gray_image.get_shape().as_list()[1],
                                                     gray_image.get_shape().as_list()[2]],
                                       method=tf.image.ResizeMethod.BILINEAR)
            print(decoder2.shape)
            conv21_gray = tf.nn.relu(conv2d(decoder2, 128, 64, k_h=3, k_w=3, d_h=1, d_w=1, name='conv21_gray'))
            conv22_gray = tf.nn.relu(conv2d(conv21_gray, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1, name='conv22_gray'))
            conv23_gray = conv2d(conv22_gray, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1, name='conv23_gray')
            conv24_gray = tf.add(conv21_gray, conv23_gray)
            gray_out = tf.nn.relu(conv2d(conv24_gray, 64, 1, k_h=3, k_w=3, d_w=1, d_h=1, name='gray_out'))
            print(gray_out.shape)

            conv1_rgb = tf.nn.relu(conv2d(self.images, 3, 64, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_rgb'))
            conv2_rgb = tf.nn.relu(conv2d(conv1_rgb, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_rgb'))
            conv3_rgb = tf.nn.relu(conv2d(conv2_rgb, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_rgb'))
            conv4_rgb = tf.add(conv1_rgb, conv3_rgb)
            conv5_rgb = tf.nn.relu(conv2d(conv4_rgb, 64, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='conv5_rgb'))
            conv6_rgb = tf.nn.relu(conv2d(conv5_rgb, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='conv6_rgb'))
            conv7_rgb = tf.nn.relu(conv2d(conv6_rgb, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='conv7_rgb'))
            conv8_rgb = tf.add(conv5_rgb, conv7_rgb)

            #attention
            rgb_out = Squeeze_excitation_layer(conv8_rgb, 128, 8, 'Squeeze_excitation_layer1')
            print(rgb_out.shape)

            out = tf.multiply(rgb_out, gray_out)
            print(out.shape)

            conv1_out = tf.nn.relu(conv2d(out, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_out'))
            conv2_out = tf.nn.relu(conv2d(conv1_out, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_out'))
            conv3_out = tf.nn.relu(conv2d(conv2_out, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_out'))
            conv4_out = tf.add(conv1_out, conv3_out)
            conv5_out = tf.nn.relu(conv2d(conv4_out, 128, 64, k_h=3, k_w=3, d_h=1, d_w=1, name='conv5_out'))
            conv6_out = tf.nn.relu(conv2d(conv5_out, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1, name='conv6_out'))
            conv7_out = tf.nn.relu(conv2d(conv6_out, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1, name='conv7_out'))
            conv8_out = tf.nn.relu(conv2d(conv7_out, 64, 3, k_h= 3, k_w=3, d_h=1, d_w=1, name='conv8_out'))
            print(conv8_out.shape)

            output = conv8_out
        return output, gray_out


    def save(self, checkpoint_dir, step):
        model_name = "coarse.model"
        model_dir = "%s_%s" % ("coarse", self.image_height)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)


    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s" % ("coarse", self.label_shape)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def count_flops(self):
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(
            graph=self.sess.graph,
            run_meta=run_meta,
            cmd='op',
            options=opts
        )
        print(f"Total FLOPs: {flops.total_float_ops / 1000000000}")
