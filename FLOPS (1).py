import time

from flops import T_CNN
from absl import flags
import tensorflow as tf
import pprint
import os
import glob
import cv2 as cv

from utils import prepare_data

flags.DEFINE_integer("epoch", 1, "Number of epoch [120]")
flags.DEFINE_integer("batch_size", 1, "The size of batch images [128]")
flags.DEFINE_integer("image_height", 256, "The size of image to use [230]")
flags.DEFINE_integer("image_width", 256, "The size of image to use [310]")
flags.DEFINE_integer("label_shape", 100, "The size of image to use [230]")
flags.DEFINE_float("learning_rate", 0.0001, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("data", "FLOPs", "Name of data directory")
flags.DEFINE_string("labels_dir", "labels", "labels_dir")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", "enhancement", "Name of sample directory [sample]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [True]")
FLAGS = flags.FLAGS
pp = pprint.PrettyPrinter()

def main(_):

    if not os.path.exists(FLAGS.checkpoint_dir):
      os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
      os.makedirs(FLAGS.sample_dir)

    test_data_list = prepare_data(FLAGS.data, "img_flops")

    test_data_wb_list = prepare_data(FLAGS.data, "img_wb")
    test_data_ce_list = prepare_data(FLAGS.data, "img_ce")
    test_data_gc_list = prepare_data(FLAGS.data, "img_gc")

    time_start = time.time()

    for ide in range(0,len(test_data_list)):
        print(ide)
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            srcnn = T_CNN(sess,
                      image_height=FLAGS.image_height,
                      image_width=FLAGS.image_width,
                      label_shape=FLAGS.label_shape,
                      batch_size=FLAGS.batch_size,
                      c_dim=FLAGS.c_dim,
                      data=FLAGS.data,
                      labels_dir=FLAGS.labels_dir,
                      checkpoint_dir=FLAGS.checkpoint_dir,
                      sample_dir=FLAGS.sample_dir,
                      test_image_name = test_data_list[ide],
                      test_wb_name = test_data_wb_list[ide],
                      test_ce_name = test_data_ce_list[ide],
                      test_gc_name = test_data_gc_list[ide],
                      id = ide
                      )
            print("Loop1")
            srcnn.train(FLAGS)
            sess.close()
        tf.compat.v1.get_default_graph().finalize()

    time_end = time.time()
    time_final = time_end - time_start
    print("final time:", time_final / 20)



if __name__ == '__main__':
  tf.compat.v1.app.run()