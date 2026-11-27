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
            image_testm = cv.imread(self.test_image_name, cv.IMREAD_COLOR)
            image_test = cv.cvtColor(image_testm, cv.COLOR_BGR2RGB) / 255
            shape = image_test.shape
            expand_test = image_test[np.newaxis, :, :, :]
            expand_zero = np.zeros([self.batch_size - 1, shape[0], shape[1], shape[2]])
            batch_test_image = np.append(expand_test, expand_zero, axis=0)

            image_testwb = cv.imread(self.test_wb_name, cv.IMREAD_GRAYSCALE)
            image_wb_test = image_testwb / 255
            image_wb_test = np.expand_dims(image_wb_test, axis=-1)
            shape1 = image_wb_test.shape
            expand_wb_test = image_wb_test[np.newaxis, :, :, :]
            expand_wb_zero = np.zeros([self.batch_size - 1, shape1[0], shape1[1], shape1[2]])
            batch_test_wb = np.append(expand_wb_test, expand_wb_zero, axis=0)

            start_time = time.time()
            output_name2 = os.path.basename(self.test_image_name)
            print(output_name2)
            result_h = self.sess.run(self.pred_h1,
                                     feed_dict={self.images: batch_test_image, self.images_depth: batch_test_wb})
            all_time = time.time()
            final_time = all_time - start_time
            print(final_time)

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
        print(f"Total FLOPs: {flops.total_float_ops}")
