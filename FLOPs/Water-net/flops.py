import time
import cv2 as cv
from ops import *
from loss.loss import *

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
        self.images_wb = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images_wb')
        self.images_ce = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images_ce')
        self.images_gc = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images_gc')
        self.labels_image = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='labels_image')

        self.pred_h1= self.model()
        '''
        loss
        '''
        self.loss_vgg = loss_vgg(self.labels_image, self.pred_h1)
        self.loss_l1 = loss_l1(self.labels_image, self.pred_h1)
        self.loss_ssim = ssim_loss(self.labels_image, self.pred_h1)
        self.loss = 0.05 * self.loss_vgg + self.loss_l1

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

                print("Epoch:",yc)

                for idx in range(0, batch_idxs):
                    batch_files = data_train_list[idx*config.batch_size:(idx+1)*config.batch_size]
                    batch_files_wb = data_wb_train_list[idx*config.batch_size:(idx+1)*config.batch_size]
                    batch_files_ce = data_ce_train_list[idx*config.batch_size:(idx+1)*config.batch_size]
                    batch_files_gc = data_gc_train_list[idx*config.batch_size:(idx+1)*config.batch_size]
                    batch_image_files = image_train_list[idx*config.batch_size : (idx+1)*config.batch_size]

                    batch_input = [get_image(batch_file, self.image_height, self.image_width) for batch_file in batch_files]
                    batch_wb_input = [get_image(batch_wb_file, self.image_height, self.image_width) for batch_wb_file in batch_files_wb]
                    batch_ce_input = [get_image(batch_ce_file, self.image_height, self.image_width) for batch_ce_file in batch_files_ce]
                    batch_gc_input = [get_image(batch_gc_file, self.image_height, self.image_width) for batch_gc_file in batch_files_gc]
                    batch_image_input = [get_image(batch_image_file, self.image_height, self.image_width) for batch_image_file in batch_image_files]

                    counter += 1
                    _, err, err1, err2, err3 = self.sess.run([self.train_op, self.loss, self.loss_ssim, self.loss_l1, self.loss_vgg],
                                                             feed_dict={self.images: batch_input,
                                                                        self.images_wb: batch_wb_input,
                                                                        self.images_ce: batch_ce_input,
                                                                        self.images_gc: batch_gc_input,
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

                #labels
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
            conb0 = tf.concat(axis=3, values=[self.images, self.images_wb, self.images_ce, self.images_gc])
            conv_wb1 = tf.nn.relu(conv2d(conb0, 16, 128, k_h=7, k_w=7, d_h=1, d_w=1, name="conv2wb_1"))
            conv_wb2 = tf.nn.relu(conv2d(conv_wb1, 128, 128, k_h=5, k_w=5, d_h=1, d_w=1, name="conv2wb_2"))
            conv_wb3 = tf.nn.relu(conv2d(conv_wb2, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2wb_3"))
            conv_wb4 = tf.nn.relu(conv2d(conv_wb3, 128, 64, k_h=1, k_w=1, d_h=1, d_w=1, name="conv2wb_4"))
            conv_wb5 = tf.nn.relu(conv2d(conv_wb4, 64, 64, k_h=7, k_w=7, d_h=1, d_w=1, name="conv2wb_5"))
            conv_wb6 = tf.nn.relu(conv2d(conv_wb5, 64, 64, k_h=5, k_w=5, d_h=1, d_w=1, name="conv2wb_6"))
            conv_wb7 = tf.nn.relu(conv2d(conv_wb6, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2wb_7"))

            conv_wb77 = tf.nn.sigmoid(conv2d(conv_wb7, 64, 3, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2wb_77"))

            conb00 = tf.concat(axis=3, values=[self.images, self.images_wb])
            conv_wb9 = tf.nn.relu(conv2d(conb00, 3, 32, k_h=7, k_w=7, d_h=1, d_w=1, name="conv2wb_9"))
            conv_wb10 = tf.nn.relu(conv2d(conv_wb9, 32, 32, k_h=5, k_w=5, d_h=1, d_w=1, name="conv2wb_10"))
            wb1 = tf.nn.relu(conv2d(conv_wb10, 32, 3, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2wb_11"))

            conb11 = tf.concat(axis=3, values=[self.images, self.images_ce])
            conv_wb99 = tf.nn.relu(conv2d(conb11, 3, 32, k_h=7, k_w=7, d_h=1, d_w=1, name="conv2wb_99"))
            conv_wb100 = tf.nn.relu(conv2d(conv_wb99, 32, 32, k_h=5, k_w=5, d_h=1, d_w=1, name="conv2wb_100"))
            ce1 = tf.nn.relu(conv2d(conv_wb100, 32, 3, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2wb_111"))

            conb111 = tf.concat(axis=3, values=[self.images, self.images_gc])
            conv_wb999 = tf.nn.relu(conv2d(conb111, 3, 32, k_h=7, k_w=7, d_h=1, d_w=1, name="conv2wb_999"))
            conv_wb1000 = tf.nn.relu(conv2d(conv_wb999, 32, 32, k_h=5, k_w=5, d_h=1, d_w=1, name="conv2wb_1000"))
            gc1 = tf.nn.relu(conv2d(conv_wb1000, 32, 3, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2wb_1111"))

            weight_wb, weight_ce, weight_gc = tf.split(conv_wb77, 3, 3)
            output = tf.add(tf.add(tf.multiply(wb1, weight_wb), tf.multiply(ce1, weight_ce)),
                             tf.multiply(gc1, weight_gc))

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