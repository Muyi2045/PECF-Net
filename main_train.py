from model import T_CNN
import os
from absl import flags
import tensorflow as tf
import pprint

tf.get_logger().setLevel('ERROR')

flags.DEFINE_integer("epoch", 300, "Number of epoch [400]")
flags.DEFINE_integer("batch_size", 4, "The size of batch images [16]")
flags.DEFINE_integer("image_height", 320, "The size of image to use [230]")
flags.DEFINE_integer("image_width", 320, "The size of image to use [310]")
flags.DEFINE_integer("label_shape", 320, "The size of image to use [230]")
flags.DEFINE_float("learning_rate", 0.0001, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("data", "UIEB", "Name of data directory")
flags.DEFINE_string("labels_dir", "labels", "labels_dir")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory [sample]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")
FLAGS = flags.FLAGS
pp = pprint.PrettyPrinter()

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  
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
                  test_image_name=None,
                  test_wb_name=None,
                  test_ce_name=None,
                  test_gc_name=None,
                  id=0
                  )

    srcnn.train(FLAGS)
    
if __name__ == '__main__':
  tf.compat.v1.app.run()

