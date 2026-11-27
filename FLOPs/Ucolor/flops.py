import time
import cv2 as cv
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
        self.images_depth = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, 1], name='images_depth')
        self.labels_image = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='labels_image')

        self.pred_h1= self.model()
        '''
        loss
        '''
        self.loss_vgg = loss_vgg(self.labels_image, self.pred_h1)
        self.loss_l2 = loss_l2(self.labels_image, self.pred_h1)
        self.loss_ssim = ssim_loss(self.labels_image, self.pred_h1)
        self.loss = 0.05 * self.loss_vgg + 5 * self.loss_l2

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
                data_wb_train_list = prepare_data(self.data, dataset="depth_train")
                image_train_list = prepare_data(self.data, dataset="gt_train")

                seed = 500 + ep
                np.random.seed(seed)
                np.random.shuffle(data_train_list)
                np.random.seed(seed)
                np.random.shuffle(data_wb_train_list)
                np.random.seed(seed)
                np.random.shuffle(image_train_list)

                batch_idxs = len(data_train_list) // config.batch_size
                loss_train = np.ones(batch_idxs)
                loss_train1 = np.ones(batch_idxs)
                loss_train2 = np.ones(batch_idxs)
                loss_train3 = np.ones(batch_idxs)

                print("Epoch:",yc)

                for idx in range(0, batch_idxs):
                    batch_files = data_train_list[idx*config.batch_size:(idx+1)*config.batch_size]
                    batch_files_wb = data_wb_train_list[idx*config.batch_size:(idx+1)*config.batch_size]
                    batch_image_files = image_train_list[idx*config.batch_size : (idx+1)*config.batch_size]

                    batch_input = [get_image(batch_file, self.image_height, self.image_width) for batch_file in batch_files]
                    batch_wb_input = [get_image(batch_wb_file, self.image_height, self.image_width) for batch_wb_file in batch_files_wb]
                    batch_wb_input = np.expand_dims(batch_wb_input, axis=-1)
                    batch_image_input = [get_image(batch_image_file, self.image_height, self.image_width) for batch_image_file in batch_image_files]

                    counter += 1
                    _, err, err1, err2, err3 = self.sess.run([self.train_op, self.loss, self.loss_ssim, self.loss_l2, self.loss_vgg],
                                                             feed_dict={self.images: batch_input,
                                                                        self.images_depth: batch_wb_input,
                                                                        self.labels_image:batch_image_input})

                    progress = (counter) / batch_idxs * 100
                    print(f'\rProgress: [{"=" * ( counter )}{" " * (batch_idxs - (counter + 1))}] {progress:.2f}%', end='',
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
                print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]，loss_ssim: [%.8f]，loss_h1: [%.8f]， loss_p1: [%.8f]" \
                    % ((ep + 1), counter, time.time() - start_time, loss1, loss2, loss3, loss4))
                loss[ep]=np.mean(loss_train)
                counter = 0
                yc += 1
                self.save(config.checkpoint_dir, yc-1)
                '''
                #labels
                if yc % 10 == 0:
                    label_train_list = prepare_data(self.labels_dir, dataset="input_labels")
                    label_wb_train_list = prepare_data(self.labels_dir, dataset="input_wb_labels")
                    label_gt_train_list = prepare_data(self.labels_dir, dataset="input_gt_labels")

                    batch_labels_idxs = len(label_train_list) // config.batch_size
                    loss_label0 = np.ones(batch_labels_idxs)
                    loss_label1 = np.ones(batch_labels_idxs)
                    loss_label2 = np.ones(batch_labels_idxs)
                    loss_label3 = np.ones(batch_labels_idxs)

                    for idx in range(0, batch_labels_idxs):
                        batch_files = label_train_list[idx * config.batch_size:(idx + 1) * config.batch_size]
                        batch_files_wb = label_wb_train_list[idx * config.batch_size:(idx + 1) * config.batch_size]
                        batch_image_files = label_gt_train_list[idx * config.batch_size: (idx + 1) * config.batch_size]

                        batch_input = [get_image(batch_file, self.image_height, self.image_width) for batch_file in
                                       batch_files]
                        batch_wb_input = [get_image(batch_wb_file, self.image_height, self.image_width) for
                                          batch_wb_file in
                                          batch_files_wb]
                        batch_image_input = [get_image(batch_image_file, self.image_height, self.image_width) for
                                             batch_image_file in batch_image_files]

                        err, err1, err2, err3 = self.sess.run(
                            [self.loss, self.loss_ssim, self.loss_l2, self.loss_vgg],
                            feed_dict={self.images: batch_input,
                                       self.images_depth: batch_wb_input,
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

            image_testwb = cv.imread(self.test_wb_name, cv.IMREAD_GRAYSCALE)
            image_testwb = cv.resize(image_testwb, (shape0[0], shape0[1]))
            image_wb_test = image_testwb / 255
            image_wb_test = np.expand_dims(image_wb_test, axis=-1)
            expand_wb_test = image_wb_test[np.newaxis, :, :, :]
            expand_wb_zero = np.zeros([self.batch_size - 1, shape0[0], shape0[1], 1])
            batch_test_wb = np.append(expand_wb_test, expand_wb_zero, axis=0)

            start_time = time.time()
            output_name2 = os.path.basename(self.test_image_name)
            print(output_name2)
            result_h = self.sess.run(self.pred_h1,
                                     feed_dict={self.images: batch_test_image, self.images_depth: batch_test_wb})
            all_time = time.time()
            final_time = all_time - start_time
            print(final_time)
            self.count_flops()



    def model(self):

        with tf.compat.v1.variable_scope("main_branch") as scope3:
            #RMT
            depth_2down = max_pool_2x2(1 - self.images_depth)
            depth_4down = max_pool_2x2(depth_2down)
            depth_8down = max_pool_2x2(depth_4down)
            # first HSV encoder
            HSV = tf.image.rgb_to_hsv(self.images)
            conv2_1_HSV = tf.nn.relu(conv2d(HSV, 3, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_1_HSV"))
            conv2_cb1_1_HSV = tf.nn.relu(conv2d(conv2_1_HSV, 3, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb1_1_HSV"))
            conv2_cb1_2_HSV = tf.nn.relu(conv2d(conv2_cb1_1_HSV, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb1_2_HSV"))
            conv2_cb1_3_HSV = conv2d(conv2_cb1_2_HSV, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb1_3_HSV")

            first_add_HSV = tf.add(conv2_1_HSV, conv2_cb1_3_HSV)

            conv2_2_HSV = tf.nn.relu(conv2d(first_add_HSV, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_2_HSV"))
            conv2_cb1_4_HSV = tf.nn.relu(conv2d(conv2_2_HSV, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb1_4_HSV"))
            conv2_cb1_5_HSV = tf.nn.relu(conv2d(conv2_cb1_4_HSV, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb1_5_HSV"))
            conv2_cb1_6_HSV = conv2d(conv2_cb1_5_HSV, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb1_6_HSV")

            first_add1_HSV = tf.add(conv2_2_HSV, conv2_cb1_6_HSV)

            encoder1_down2_HSV = max_pool_2x2(first_add1_HSV)

            # first YUV encoder
            YUV = Conv_img.rgb_to_lab(self.images)
            conv2_1_YUV = tf.nn.relu(conv2d(YUV, 3, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_1_YUV"))
            conv2_cb1_1_YUV = tf.nn.relu(conv2d(conv2_1_YUV, 3, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb1_1_YUV"))
            conv2_cb1_2_YUV = tf.nn.relu(conv2d(conv2_cb1_1_YUV, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb1_2_YUV"))
            conv2_cb1_3_YUV = conv2d(conv2_cb1_2_YUV, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb1_3_YUV")

            first_add_YUV = tf.add(conv2_1_YUV, conv2_cb1_3_YUV)

            conv2_2_YUV = tf.nn.relu(conv2d(first_add_YUV, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_2_YUV"))
            conv2_cb1_4_YUV = tf.nn.relu(conv2d(conv2_2_YUV, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb1_4_YUV"))
            conv2_cb1_5_YUV = tf.nn.relu(conv2d(conv2_cb1_4_YUV, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb1_5_YUV"))
            conv2_cb1_6_YUV = conv2d(conv2_cb1_5_YUV, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb1_6_YUV")

            first_add1_YUV = tf.add(conv2_2_YUV, conv2_cb1_6_YUV)

            encoder1_down2_YUV = max_pool_2x2(first_add1_YUV)

            # first RGB encoder
            conv2_1 = tf.nn.relu(conv2d(self.images, 3, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_1"))
            conv2_cb1_1 = tf.nn.relu(conv2d(tf.concat(axis=3, values=[conv2_1, conv2_1_HSV, conv2_1_YUV]), 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb1_1"))
            conv2_cb1_2 = tf.nn.relu(conv2d(tf.concat(axis=3, values=[conv2_cb1_1, conv2_cb1_1_HSV, conv2_cb1_1_YUV]), 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb1_2"))
            conv2_cb1_3 = conv2d(tf.concat(axis=3, values=[conv2_cb1_2, conv2_cb1_3_HSV, conv2_cb1_3_YUV]), 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb1_3")
            first_add = tf.add(conv2_1, conv2_cb1_3)
            conv2_2 = tf.nn.relu(conv2d(tf.concat(axis=3, values=[first_add, first_add_HSV, first_add_YUV]), 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_2"))
            conv2_cb1_4 = tf.nn.relu(conv2d(tf.concat(axis=3, values=[conv2_2, conv2_2_HSV, conv2_2_YUV]), 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb1_4"))
            conv2_cb1_5 = tf.nn.relu(conv2d(tf.concat(axis=3, values=[conv2_cb1_4, conv2_cb1_4_HSV, conv2_cb1_4_YUV]), 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb1_5"))
            conv2_cb1_6 = conv2d(tf.concat(axis=3, values=[conv2_cb1_5, conv2_cb1_5_HSV, conv2_cb1_5_YUV]), 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb1_6")
            first_add1 = tf.add(conv2_2, conv2_cb1_6)

            encoder1_down2 = max_pool_2x2(first_add1)

            # second hsv encoder
            conv2_2_1_HSV = tf.nn.relu(conv2d(encoder1_down2_HSV, 3, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_2_1_HSV"))
            conv2_cb2_1_HSV = tf.nn.relu(conv2d(conv2_2_1_HSV, 3, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb2_1_HSV"))
            conv2_cb2_2_HSV = tf.nn.relu(conv2d(conv2_cb2_1_HSV, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb2_2_HSV"))
            conv2_cb2_3_HSV = conv2d(conv2_cb2_2_HSV, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb2_3_HSV")
            second_add_HSV = tf.add(conv2_2_1_HSV, conv2_cb2_3_HSV)
            conv2_2_2_HSV = tf.nn.relu(conv2d(second_add_HSV, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_2_2_HSV"))
            conv2_cb2_4_HSV = tf.nn.relu(conv2d(conv2_2_2_HSV, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb2_4_HSV"))
            conv2_cb2_5_HSV = tf.nn.relu(conv2d(conv2_cb2_4_HSV, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb2_5_HSV"))
            conv2_cb2_6_HSV = conv2d(conv2_cb2_5_HSV, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb2_6_HSV")
            second_add1_HSV = tf.add(conv2_2_2_HSV, conv2_cb2_6_HSV)

            encoder2_down2_HSV = max_pool_2x2(second_add1_HSV)

            # second YUV encoder
            conv2_2_1_YUV = tf.nn.relu(conv2d(encoder1_down2_YUV, 3, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_2_1_YUV"))
            conv2_cb2_1_YUV = tf.nn.relu(conv2d(conv2_2_1_YUV, 3, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb2_1_YUV"))
            conv2_cb2_2_YUV = tf.nn.relu(conv2d(conv2_cb2_1_YUV, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb2_2_YUV"))
            conv2_cb2_3_YUV = conv2d(conv2_cb2_2_YUV, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb2_3_YUV")
            second_add_YUV = tf.add(conv2_2_1_YUV, conv2_cb2_3_YUV)
            conv2_2_2_YUV = tf.nn.relu(conv2d(second_add_YUV, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_2_2_YUV"))
            conv2_cb2_4_YUV = tf.nn.relu(conv2d(conv2_2_2_YUV, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb2_4_YUV"))
            conv2_cb2_5_YUV = tf.nn.relu(conv2d(conv2_cb2_4_YUV, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb2_5_YUV"))
            conv2_cb2_6_YUV = conv2d(conv2_cb2_5_YUV, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb2_6_YUV")
            second_add1_YUV = tf.add(conv2_2_2_YUV, conv2_cb2_6_YUV)

            encoder2_down2_YUV = max_pool_2x2(second_add1_YUV)

            # second RGB encoder
            conv2_2_1 = tf.nn.relu(conv2d(encoder1_down2, 3, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_2_1"))
            conv2_cb2_1 = tf.nn.relu(conv2d(tf.concat(axis=3, values=[conv2_2_1, conv2_2_1_HSV, conv2_2_1_YUV]), 3, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb2_1"))
            conv2_cb2_2 = tf.nn.relu(conv2d(tf.concat(axis=3, values=[conv2_cb2_1, conv2_cb2_1_HSV, conv2_cb2_1_YUV]), 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb2_2"))
            conv2_cb2_3 = conv2d(tf.concat(axis=3, values=[conv2_cb2_2, conv2_cb2_3_HSV, conv2_cb2_3_YUV]), 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb2_3")
            second_add = tf.add(conv2_2_1, conv2_cb2_3)
            conv2_2_2 = tf.nn.relu(conv2d(tf.concat(axis=3, values=[second_add, second_add_HSV, second_add_YUV]), 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_2_2"))
            conv2_cb2_4 = tf.nn.relu(conv2d(tf.concat(axis=3, values=[conv2_2_2, conv2_2_2_HSV, conv2_2_2_YUV]), 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb2_4"))
            conv2_cb2_5 = tf.nn.relu(conv2d(tf.concat(axis=3, values=[conv2_cb2_4, conv2_cb2_4_HSV, conv2_cb2_4_YUV]), 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb2_5"))
            conv2_cb2_6 = conv2d(tf.concat(axis=3, values=[conv2_cb2_5, conv2_cb2_5_HSV, conv2_cb2_5_YUV]), 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb2_6")
            second_add1 = tf.add(conv2_2_2, conv2_cb2_6)

            encoder2_down2 = max_pool_2x2(second_add1)

            # third hsv encoder
            conv2_3_1_HSV = tf.nn.relu(conv2d(encoder2_down2_HSV, 3, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_3_1_HSV"))
            conv2_cb3_1_HSV = tf.nn.relu(conv2d(conv2_3_1_HSV, 3, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb3_1_HSV"))
            conv2_cb3_2_HSV = tf.nn.relu(conv2d(conv2_cb3_1_HSV, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb3_2_HSV"))
            conv2_cb3_3_HSV = conv2d(conv2_cb3_2_HSV, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb3_3_HSV")
            third_add_HSV = tf.add(conv2_3_1_HSV, conv2_cb3_3_HSV)
            conv2_3_2_HSV = tf.nn.relu(conv2d(third_add_HSV, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_3_2_HSV"))
            conv2_cb3_4_HSV = tf.nn.relu(conv2d(conv2_3_2_HSV, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb3_4_HSV"))
            conv2_cb3_5_HSV = tf.nn.relu(conv2d(conv2_cb3_4_HSV, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb3_5_HSV"))
            conv2_cb3_6_HSV = conv2d(conv2_cb3_5_HSV, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb3_6_HSV")

            third_add1_HSV = tf.add(conv2_3_2_HSV, conv2_cb3_6_HSV)

            # third YUV encoder
            conv2_3_1_YUV = tf.nn.relu(conv2d(encoder2_down2_YUV, 3, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_3_1_YUV"))
            conv2_cb3_1_YUV = tf.nn.relu(conv2d(conv2_3_1_YUV, 3, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb3_1_YUV"))
            conv2_cb3_2_YUV = tf.nn.relu(conv2d(conv2_cb3_1_YUV, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb3_2_YUV"))
            conv2_cb3_3_YUV = conv2d(conv2_cb3_2_YUV, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb3_3_YUV")
            third_add_YUV = tf.add(conv2_3_1_YUV, conv2_cb3_3_YUV)
            conv2_3_2_YUV = tf.nn.relu(conv2d(third_add_YUV, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_3_2_YUV"))
            conv2_cb3_4_YUV = tf.nn.relu(conv2d(conv2_3_2_YUV, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb3_4_YUV"))
            conv2_cb3_5_YUV = tf.nn.relu(conv2d(conv2_cb3_4_YUV, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb3_5_YUV"))
            conv2_cb3_6_YUV = conv2d(conv2_cb3_5_YUV, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb3_6_YUV")

            third_add1_YUV = tf.add(conv2_3_2_YUV, conv2_cb3_6_YUV)

            # third RGB encoder
            conv2_3_1 = tf.nn.relu(conv2d(encoder2_down2, 3, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_3_1"))
            conv2_cb3_1 = tf.nn.relu(conv2d(tf.concat(axis=3, values=[conv2_3_1, conv2_3_1_HSV, conv2_3_1_YUV]), 3, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb3_1"))
            conv2_cb3_2 = tf.nn.relu(conv2d(tf.concat(axis=3, values=[conv2_cb3_1, conv2_cb3_1_HSV, conv2_cb3_1_YUV]), 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb3_2"))
            conv2_cb3_3 = conv2d(tf.concat(axis=3, values=[conv2_cb3_2, conv2_cb3_2_HSV, conv2_cb3_2_YUV]), 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb3_3")
            third_add = tf.add(conv2_3_1, conv2_cb3_3)
            conv2_3_2 = tf.nn.relu(conv2d(tf.concat(axis=3, values=[third_add, third_add_HSV, third_add_YUV]), 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_3_2"))
            conv2_cb3_4 = tf.nn.relu(conv2d(tf.concat(axis=3, values=[conv2_3_2, conv2_3_2_HSV, conv2_3_2_YUV]), 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb3_4"))
            conv2_cb3_5 = tf.nn.relu(conv2d(tf.concat(axis=3, values=[conv2_cb3_4, conv2_cb3_4_HSV, conv2_cb3_4_YUV]), 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb3_5"))
            conv2_cb3_6 = conv2d(tf.concat(axis=3, values=[conv2_cb3_5, conv2_cb3_5_HSV, conv2_cb3_5_YUV]), 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_cb3_6")

            third_add1 = tf.add(conv2_3_2, conv2_cb3_6)

            #concate
            third_con = tf.concat(axis=3, values=[third_add1, third_add1_HSV, third_add1_YUV])
            channle_weight_third_con_temp = Squeeze_excitation_layer(third_con, out_dim=1536, ratio=16,
                                                                          layer_name='channle_weight_third_con_temp')
            third_con_ff = tf.nn.relu(conv2d(channle_weight_third_con_temp, 1536, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="third_con_ff"))
            second_con = tf.concat(axis=3, values=[second_add1, second_add1_HSV, second_add1_YUV])
            channle_weight_second_con_temp = Squeeze_excitation_layer(second_con, out_dim=768, ratio=16,
                                                                           layer_name='channle_weight_second_con_temp')
            second_con_ff = tf.nn.relu(conv2d(channle_weight_second_con_temp, 768, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="second_con_ff"))
            first_con = tf.concat(axis=3, values=[first_add1, first_add1_HSV, first_add1_YUV])
            channle_weight_first_con_temp = Squeeze_excitation_layer(first_con, out_dim=384, ratio=16,
                                                                          layer_name='channle_weight_first_con_temp')
            first_con_ff = tf.nn.relu(conv2d(channle_weight_first_con_temp, 384, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="first_con_ff"))

            # first decoder
            decoder_input = tf.add(third_con_ff, tf.multiply(third_con_ff, depth_4down))
            conv2_1_1_dc = tf.nn.relu(conv2d(decoder_input, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_1_1_dc"))
            conv2_dc1_1 = tf.nn.relu(conv2d(conv2_1_1_dc, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc1_1"))
            conv2_dc1_2 = tf.nn.relu(conv2d(conv2_dc1_1, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc1_2"))
            conv2_dc1_3 = conv2d(conv2_dc1_2, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc1_3")
            first_dadd = tf.add(conv2_1_1_dc, conv2_dc1_3)
            conv2_1_2_dc = tf.nn.relu(conv2d(first_dadd, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_1_2_dc"))
            conv2_dc1_4 = tf.nn.relu(conv2d(conv2_1_2_dc, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc1_4"))
            conv2_dc1_5 = tf.nn.relu(conv2d(conv2_dc1_4, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc1_5"))
            conv2_dc1_6 = conv2d(conv2_dc1_5, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc1_6")
            first_dadd1 = tf.add(conv2_1_2_dc, conv2_dc1_6)
            decoder1_down2 = tf.image.resize(first_dadd1, [second_con_ff.get_shape().as_list()[1],
                                                           second_con_ff.get_shape().as_list()[2]],
                                             method=tf.image.ResizeMethod.BILINEAR)

            # second decoder
            decoder_input1 = tf.add(second_con_ff, tf.multiply(second_con_ff, depth_2down))
            concate_1 = tf.concat(axis=3, values=[decoder1_down2, decoder_input1])
            conv2_1_3_dc = tf.nn.relu(conv2d(concate_1, 512, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_1_3_dc"))
            conv2_dc2_1 = tf.nn.relu(conv2d(conv2_1_3_dc, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc2_1"))
            conv2_dc2_2 = tf.nn.relu(conv2d(conv2_dc2_1, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc2_2"))
            conv2_dc2_3 = conv2d(conv2_dc2_2, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc2_3")
            second_dadd = tf.add(conv2_1_3_dc, conv2_dc2_3)
            conv2_1_4_dc = tf.nn.relu(conv2d(second_dadd, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_1_4_dc"))
            conv2_dc2_4 = tf.nn.relu(conv2d(conv2_1_4_dc, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc2_4"))
            conv2_dc2_5 = tf.nn.relu(conv2d(conv2_dc2_4, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc2_5"))
            conv2_dc2_6 = conv2d(conv2_dc2_5, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc2_6")
            second_dadd1 = tf.add(conv2_1_4_dc, conv2_dc2_6)
            decoder2_down2 = tf.image.resize(second_dadd1, [first_con_ff.get_shape().as_list()[1],
                                                                     first_con_ff.get_shape().as_list()[2]],
                                             method=tf.image.ResizeMethod.BILINEAR)

            # third decoder
            decoder_input2 = tf.add(first_con_ff, tf.multiply(first_con_ff, (1 - self.images_depth)))
            concate_2 = tf.concat(axis=3, values=[decoder2_down2, decoder_input2])
            conv2_1_5_dc = tf.nn.relu(conv2d(concate_2, 256, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_1_5_dc"))
            conv2_dc3_1 = tf.nn.relu(conv2d(conv2_1_5_dc, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc3_1"))
            conv2_dc3_2 = tf.nn.relu(conv2d(conv2_dc3_1, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc3_2"))
            conv2_dc3_3 = conv2d(conv2_dc3_2, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc3_3")
            third_dadd = tf.add(conv2_1_5_dc, conv2_dc3_3)
            conv2_1_6_dc = tf.nn.relu(conv2d(third_dadd, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_1_6_dc"))
            conv2_dc3_4 = tf.nn.relu(conv2d(conv2_1_6_dc, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc3_4"))
            conv2_dc3_5 = tf.nn.relu(conv2d(conv2_dc3_4, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc3_5"))
            conv2_dc3_6 = conv2d(conv2_dc3_5, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_dc3_6")
            third_dadd1 = tf.add(conv2_1_6_dc, conv2_dc3_6)
            conv2_refine = conv2d(third_dadd1, 128, 3, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_refine")

            output = conv2_refine


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
