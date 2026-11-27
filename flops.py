import time
import cv2 as cv
from ops import *
from loss.loss import *


class T_CNN(object):
    def __init__(self, sess, image_height, image_width, label_shape,
                 batch_size, c_dim, data, labels_dir, checkpoint_dir, sample_dir, test_image_name, test_wb_name,
                 test_ce_name, test_gc_name, id):
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
        self.images = tf.compat.v1.placeholder(tf.float32,
                                               [self.batch_size, self.image_height, self.image_width, self.c_dim],
                                               name='images')
        self.images_wb = tf.compat.v1.placeholder(tf.float32,
                                                  [self.batch_size, self.image_height, self.image_width, self.c_dim],
                                                  name='images_wb')
        self.images_ce = tf.compat.v1.placeholder(tf.float32,
                                                  [self.batch_size, self.image_height, self.image_width, self.c_dim],
                                                  name='images_ce')
        self.images_gc = tf.compat.v1.placeholder(tf.float32,
                                                  [self.batch_size, self.image_height, self.image_width, self.c_dim],
                                                  name='images_gc')
        self.labels_image = tf.compat.v1.placeholder(tf.float32,
                                                     [self.batch_size, self.image_height, self.image_width, self.c_dim],
                                                     name='labels_image')

        self.pred_h1 = self.model()
        '''
        loss
        '''
        self.loss_vgg = loss_vgg(self.labels_image, self.pred_h1)
        self.loss_l1 = loss_l1(self.labels_image, self.pred_h1)
        self.loss_ssim = ssim_loss(self.labels_image, self.pred_h1)
        self.loss = 0.05 * self.loss_vgg + self.loss_l1

        # 获取所有变量
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
        self.train_op = tf.compat.v1.train.AdamOptimizer(config.learning_rate, 0.9).minimize(self.loss)
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

        # 训练training
        if config.is_train:
            print("Training...")
            loss = np.ones(config.epoch)

            for ep in range(config.epoch):
                data_train_list = prepare_data(self.data, dataset="input_train")
                data_wb_train_list = prepare_data(self.data, dataset="input_wb_train")
                data_ce_train_list = prepare_data(self.data, dataset="input_ce_train")
                data_gc_train_list = prepare_data(self.data, dataset="input_gc_train")
                image_train_list = prepare_data(self.data, dataset="input_gt_train")

                seed = 500 + ep
                np.random.seed(seed)
                np.random.shuffle(data_train_list)
                np.random.seed(seed)
                np.random.shuffle(data_wb_train_list)
                np.random.seed(seed)
                np.random.shuffle(data_ce_train_list)
                np.random.seed(seed)
                np.random.shuffle(data_gc_train_list)
                np.random.seed(seed)
                np.random.shuffle(image_train_list)

                batch_idxs = len(data_train_list) // config.batch_size
                loss_train = np.ones(batch_idxs)
                loss_train1 = np.ones(batch_idxs)
                loss_train2 = np.ones(batch_idxs)
                loss_train3 = np.ones(batch_idxs)

                print("Epoch:", yc)

                for idx in range(0, batch_idxs):
                    batch_files = data_train_list[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_files_wb = data_wb_train_list[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_files_ce = data_ce_train_list[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_files_gc = data_gc_train_list[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_image_files = image_train_list[idx * config.batch_size: (idx + 1) * config.batch_size]

                    batch_input = [get_image(batch_file, self.image_height, self.image_width) for batch_file in
                                   batch_files]
                    batch_wb_input = [get_image(batch_wb_file, self.image_height, self.image_width) for batch_wb_file in
                                      batch_files_wb]
                    batch_ce_input = [get_image(batch_ce_file, self.image_height, self.image_width) for batch_ce_file in
                                      batch_files_ce]
                    batch_gc_input = [get_image(batch_gc_file, self.image_height, self.image_width) for batch_gc_file in
                                      batch_files_gc]
                    batch_image_input = [get_image(batch_image_file, self.image_height, self.image_width) for
                                         batch_image_file in batch_image_files]

                    counter += 1
                    _, err, err1, err2, err3 = self.sess.run(
                        [self.train_op, self.loss, self.loss_ssim, self.loss_l1, self.loss_vgg],
                        feed_dict={self.images: batch_input,
                                   self.images_wb: batch_wb_input,
                                   self.images_ce: batch_ce_input,
                                   self.images_gc: batch_gc_input,
                                   self.labels_image: batch_image_input})

                    progress = (counter) / batch_idxs * 100
                    print(f'\rProgress: [{"=" * (counter)}{" " * (batch_idxs - (counter + 1))}] {progress:.2f}%',
                          end='',
                          flush=True)
                    loss_train[idx] = err
                    loss_train1[idx] = err1
                    loss_train2[idx] = err2
                    loss_train3[idx] = err3
                loss1 = np.mean(loss_train)
                loss2 = np.mean(loss_train1)
                loss3 = np.mean(loss_train2)
                loss4 = np.mean(loss_train3)
                print()
                print(
                    "Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]，loss_ssim: [%.8f]，loss_h1: [%.8f]， loss_p1: [%.8f]" \
                    % ((ep + 1), counter, time.time() - start_time, loss1, loss2, loss3, loss4))
                loss[ep] = np.mean(loss_train)
                counter = 0
                yc += 1
                self.save(config.checkpoint_dir, yc - 1)

                # labels
                if yc % 10 == 0:
                    label_train_list = prepare_data(self.labels_dir, dataset="input_labels")
                    label_wb_train_list = prepare_data(self.labels_dir, dataset="input_wb_labels")
                    label_ce_train_list = prepare_data(self.labels_dir, dataset="input_ce_labels")
                    label_gc_train_list = prepare_data(self.labels_dir, dataset="input_gc_labels")
                    label_gt_train_list = prepare_data(self.labels_dir, dataset="input_gt_labels")

                    batch_labels_idxs = len(label_train_list) // config.batch_size
                    loss_label0 = np.ones(batch_labels_idxs)
                    loss_label1 = np.ones(batch_labels_idxs)
                    loss_label2 = np.ones(batch_labels_idxs)
                    loss_label3 = np.ones(batch_labels_idxs)

                    for idx in range(0, batch_labels_idxs):
                        batch_files = label_train_list[idx * config.batch_size:(idx + 1) * config.batch_size]
                        batch_files_wb = label_wb_train_list[idx * config.batch_size:(idx + 1) * config.batch_size]
                        batch_files_ce = label_ce_train_list[idx * config.batch_size:(idx + 1) * config.batch_size]
                        batch_files_gc = label_gc_train_list[idx * config.batch_size:(idx + 1) * config.batch_size]
                        batch_image_files = label_gt_train_list[idx * config.batch_size: (idx + 1) * config.batch_size]

                        batch_input = [get_image(batch_file, self.image_height, self.image_width) for batch_file in
                                       batch_files]
                        batch_wb_input = [get_image(batch_wb_file, self.image_height, self.image_width) for
                                          batch_wb_file in
                                          batch_files_wb]
                        batch_ce_input = [get_image(batch_ce_file, self.image_height, self.image_width) for
                                          batch_ce_file in
                                          batch_files_ce]
                        batch_gc_input = [get_image(batch_gc_file, self.image_height, self.image_width) for
                                          batch_gc_file in
                                          batch_files_gc]
                        batch_image_input = [get_image(batch_image_file, self.image_height, self.image_width) for
                                             batch_image_file in batch_image_files]

                        err, err1, err2, err3 = self.sess.run(
                            [self.loss, self.loss_ssim, self.loss_l1, self.loss_vgg],
                            feed_dict={self.images: batch_input,
                                       self.images_wb: batch_wb_input,
                                       self.images_ce: batch_ce_input,
                                       self.images_gc: batch_gc_input,
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

            image_testwb = cv.imread(self.test_wb_name, cv.IMREAD_COLOR)
            image_testwb = cv.resize(image_testwb, (shape0[0], shape0[1]))
            image_wb_test = cv.cvtColor(image_testwb, cv.COLOR_BGR2RGB) / 255
            expand_wb_test = image_wb_test[np.newaxis, :, :, :]
            expand_wb_zero = np.zeros([self.batch_size - 1, shape0[0], shape0[1], shape0[2]])
            batch_test_wb = np.append(expand_wb_test, expand_wb_zero, axis=0)

            image_testce = cv.imread(self.test_ce_name, cv.IMREAD_COLOR)
            image_testce = cv.resize(image_testce, (shape0[0], shape0[1]))
            image_ce_test = cv.cvtColor(image_testce, cv.COLOR_BGR2RGB) / 255
            expand_ce_test = image_ce_test[np.newaxis, :, :, :]
            expand_ce_zero = np.zeros([self.batch_size - 1, shape0[0], shape0[1], shape0[2]])
            batch_test_ce = np.append(expand_ce_test, expand_ce_zero, axis=0)

            image_testgc = cv.imread(self.test_gc_name, cv.IMREAD_COLOR)
            image_testgc = cv.resize(image_testgc, (shape0[0], shape0[1]))
            image_gc_test = cv.cvtColor(image_testgc, cv.COLOR_BGR2RGB) / 255
            expand_gc_test = image_gc_test[np.newaxis, :, :, :]
            expand_gc_zero = np.zeros([self.batch_size - 1, shape0[0], shape0[1], shape0[2]])
            batch_test_gc = np.append(expand_gc_test, expand_gc_zero, axis=0)

            start_time = time.time()
            output_name2 = os.path.basename(self.test_image_name)
            print(output_name2)
            result_h = self.sess.run(self.pred_h1,
                                     feed_dict={self.images: batch_test_image, self.images_wb: batch_test_wb,
                                                self.images_ce: batch_test_ce, self.images_gc: batch_test_gc})
            all_time = time.time()
            final_time = all_time - start_time
            print(final_time)
            self.count_flops()

    def model(self):
        with tf.compat.v1.variable_scope("main_branch") as scope3:
            # Feature Extraction
            conv_E1 = tf.nn.relu(conv2d(self.images, 3, 64, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_E1"))
            conv_E2 = tf.nn.relu(conv2d(conv_E1, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_E2"))
            conv_E5 = tf.nn.relu(conv2d(conv_E2, 64, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_E5"))
            conv_E6 = tf.nn.relu(conv2d(conv_E5, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_E6"))
            conv_E8 = tf.nn.relu(conv2d(conv_E6, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_E8"))

            # Preconditioning
            conv_P1 = tf.nn.relu(conv2d(self.images_wb, 3, 64, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_P1"))
            conv_P2 = tf.nn.relu(conv2d(conv_P1, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_P2"))
            conv_P4 = tf.nn.relu(conv2d(conv_P2, 64, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_P4"))

            conv_P11 = tf.nn.relu(conv2d(self.images_ce, 3, 64, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_P11"))
            conv_P22 = tf.nn.relu(conv2d(conv_P11, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_P22"))
            conv_P44 = tf.nn.relu(conv2d(conv_P22, 64, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_P44"))

            conv_P111 = tf.nn.relu(conv2d(self.images_gc, 3, 64, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_P111"))
            conv_P222 = tf.nn.relu(conv2d(conv_P111, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_P222"))
            conv_P444 = tf.nn.relu(conv2d(conv_P222, 64, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_P444"))

            # Channel attention mechanism
            conv_CA1 = conv2d(conv_E8, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_CA1")
            conv_CA2 = conv2d(conv_E8, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_CA2")
            conv_CA3 = conv2d(conv_E8, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_CA3")

            # channel = CBAM_attention_layer(conv_E8, out_dim=128, ratio=16, layer_name='channle_weight')
            '''

            '''
            channel_weight = Squeeze_excitation_layer(conv_CA1, out_dim=128, ratio=8, layer_name='channle_weight')
            channel_attention = Channel_attention(conv_CA2, layer_name='channel_attention')
            channel1 = tf.multiply(conv_CA3, channel_weight)
            channel2 = tf.multiply(conv_CA3, channel_attention)
            channel = tf.add(channel1, channel2)

            wb_out = CBAM_attention_layer(conv_P4, out_dim=128, ratio=16, layer_name='channle_weight_wb')
            ce_out = CBAM_attention_layer(conv_P44, out_dim=128, ratio=16, layer_name='channle_weight_ce')
            gc_out = CBAM_attention_layer(conv_P444, out_dim=128, ratio=16, layer_name='channle_weight_gc')
            '''
            # shenjing
            avagepool_wb = Global_Average_Pooling(conv_P4)
            ava_wb = tf.reduce_mean(conv_P4, axis=-1, keepdims=True)
            fc_wb = Fully_connected(avagepool_wb, units=153, layer_name='wb_fully_connected')
            R, G, B = tf.split(fc_wb, 3, axis=1)
            R_in, G_in, B_in = self.images_wb[:, :, :, 0:1], self.images_wb[:, :, :, 1:2], self.images_wb[:, :, :, 2:3]
            R_out = piece_function_org(R_in, R, ava_wb, 51)
            G_out = piece_function_org(G_in, G, ava_wb, 51)
            B_out = piece_function_org(B_in, B, ava_wb, 51)

            zero_lab = tf.zeros_like(R_out)
            one_lab = tf.ones_like(R_out)
            r_t = tf.where(R_out < 0, zero_lab, R_out)
            R_out = tf.where(r_t > 1, one_lab, r_t)
            zero_lab = tf.zeros_like(G_out)
            one_lab = tf.ones_like(G_out)
            g_t = tf.where(G_out < 0, zero_lab, G_out)
            G_out = tf.where(g_t > 1, one_lab, g_t)
            zero_lab = tf.zeros_like(B_out)
            one_lab = tf.ones_like(B_out)
            b_t = tf.where(B_out < 0, zero_lab, B_out)
            B_out = tf.where(b_t > 1, one_lab, b_t)

            wb_out = tf.concat([R_out, G_out, B_out], axis=-1)
            wb_out = conv2d(wb_out, 3, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="wb_out")

            avagepool_ce = Global_Average_Pooling(conv_P44)
            ava_ce = tf.reduce_mean(conv_P44, axis=-1, keepdims=True)
            fc_ce = Fully_connected(avagepool_ce, units=153, layer_name='ce_fully_connected')
            R1, G1, B1 = tf.split(fc_ce, 3, axis=1)
            R_in1, G_in1, B_in1 = self.images_ce[:, :, :, 0:1], self.images_ce[:, :, :, 1:2], self.images_ce[:, :, :,
                                                                                              2:3]
            R_out1 = piece_function_org(R_in1, R1, ava_ce, 51)
            G_out1 = piece_function_org(G_in1, G1, ava_ce, 51)
            B_out1 = piece_function_org(B_in1, B1, ava_ce, 51)

            zero_lab1 = tf.zeros_like(R_out1)
            one_lab1 = tf.ones_like(R_out1)
            r_t1 = tf.where(R_out1 < 0, zero_lab1, R_out1)
            R_out1 = tf.where(r_t1 > 1, one_lab1, r_t1)
            zero_lab1 = tf.zeros_like(G_out1)
            one_lab1 = tf.ones_like(G_out1)
            g_t1 = tf.where(G_out1 < 0, zero_lab1, G_out1)
            G_out1 = tf.where(g_t1 > 1, one_lab1, g_t1)
            zero_lab1 = tf.zeros_like(B_out1)
            one_lab1 = tf.ones_like(B_out1)
            b_t1 = tf.where(B_out1 < 0, zero_lab1, B_out1)
            B_out1 = tf.where(b_t1 > 1, one_lab1, b_t1)

            ce_out = tf.concat([R_out1, G_out1, B_out1], axis=-1)
            ce_out = conv2d(ce_out, 3, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="ce_out")

            avagepool_gc = Global_Average_Pooling(conv_P444)
            ava_gc = tf.reduce_mean(conv_P444, axis=-1, keepdims=True)
            fc_gc = Fully_connected(avagepool_gc, units=153, layer_name='gc_fully_connected')
            R2, G2, B2 = tf.split(fc_gc, 3, axis=1)
            R_in2, G_in2, B_in2 = self.images_gc[:, :, :, 0:1], self.images_gc[:, :, :, 1:2], self.images_gc[:, :, :,
                                                                                              2:3]
            R_out2 = piece_function_org(R_in2, R2, ava_gc, 51)
            G_out2 = piece_function_org(G_in2, G2, ava_gc, 51)
            B_out2 = piece_function_org(B_in2, B2, ava_gc, 51)

            zero_lab2 = tf.zeros_like(R_out2)
            one_lab2 = tf.ones_like(R_out2)
            r_t2 = tf.where(R_out2 < 0, zero_lab2, R_out2)
            R_out2 = tf.where(r_t2 > 1, one_lab2, r_t2)
            zero_lab2 = tf.zeros_like(G_out2)
            one_lab2 = tf.ones_like(G_out2)
            g_t2 = tf.where(G_out2 < 0, zero_lab2, G_out2)
            G_out2 = tf.where(g_t2 > 1, one_lab2, g_t2)
            zero_lab2 = tf.zeros_like(B_out2)
            one_lab2 = tf.ones_like(B_out2)
            b_t2 = tf.where(B_out2 < 0, zero_lab2, B_out2)
            B_out2 = tf.where(b_t2 > 1, one_lab2, b_t2)

            gc_out = tf.concat([R_out2, G_out2, B_out2], axis=-1)
            gc_out = conv2d(gc_out, 3, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="gc_out")
            '''
            conv_F1 = tf.add(channel, wb_out)
            conv_F1 = tf.add(conv_F1, ce_out)
            conv_F1 = tf.add(conv_F1, gc_out)
            conv_F2 = tf.nn.relu(conv2d(conv_F1, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_F2"))
            conv_F5 = conv2d(conv_F2, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_F5")
            conv_F6 = tf.add(conv_F1, conv_F5)
            conv_F7 = tf.nn.relu(conv2d(conv_F6, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_F7"))
            conv_F10 = conv2d(conv_F7, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_F10")
            conv_F11 = tf.add(conv_F6, conv_F10)

            conv1_F1 = tf.add(conv_F11, channel)
            conv1_F2 = tf.nn.relu(conv2d(conv1_F1, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv1_F2"))
            conv1_F5 = conv2d(conv1_F2, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv1_F5")
            conv1_F6 = tf.add(conv1_F1, conv1_F5)
            conv1_F7 = tf.nn.relu(conv2d(conv1_F6, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv1_F7"))
            conv1_F10 = conv2d(conv1_F7, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv1_F10")
            conv1_F11 = tf.add(conv1_F6, conv1_F10)

            conv2_F1 = tf.add(conv1_F11, channel)
            conv2_F2 = tf.nn.relu(conv2d(conv2_F1, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_F2"))
            conv2_F5 = conv2d(conv2_F2, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_F5")
            conv2_F6 = tf.add(conv2_F1, conv2_F5)
            conv2_F7 = tf.nn.relu(conv2d(conv2_F6, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_F7"))
            conv2_F10 = conv2d(conv2_F7, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_F10")
            conv2_F11 = tf.add(conv2_F6, conv2_F10)

            conv_out1 = tf.nn.relu(conv2d(conv2_F11, 128, 64, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_out1"))
            conv_out2 = tf.nn.relu(conv2d(conv_out1, 64, 32, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_out2"))
            output = conv2d(conv_out2, 32, 3, k_h=3, k_w=3, d_h=1, d_w=1, name="output")

        return output

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