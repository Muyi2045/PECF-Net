import math
import numpy as np 
import tensorflow as tf
import keras as keras

#from tensorflow.python.framework import ops

from utils import *
'''
统一版本问题
'''
try:
  image_summary = tf.image_summary
  scalar_summary = tf.scalar_summary
  histogram_summary = tf.histogram_summary
  merge_summary = tf.merge_summary
  SummaryWriter = tf.train.SummaryWriter
except:
  image_summary = tf.compat.v1.summary.image
  scalar_summary = tf.compat.v1.summary.scalar
  histogram_summary = tf.compat.v1.summary.histogram
  merge_summary = tf.compat.v1.summary.merge
  SummaryWriter = tf.compat.v1.summary.FileWriter

'''
基础层
批量归一化层 Batch Normalization
对每个特征进行归一化处理，使得特征分布保持一致，帮助网络更快收敛，避免了深度网络在训练时可能出现的梯度消失或梯度爆炸问题
'''
class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.compat.v1.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.compat.v1.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

'''
Local Response Normalization (LRN) 层
常用于卷积神经网络中的归一化方法，作用是对局部区域进行归一化，增强模型的鲁棒性。
'''
def lrn(inputs, depth_radius=2, alpha=0.0001, beta=0.75, bias=1.0):
    return tf.nn.local_response_normalization(inputs, depth_radius=depth_radius, alpha=alpha, beta=beta, bias=bias)

'''
卷积层 Conv
增加了卷积核的初始化、偏置项添加、变量作用域管理等功能，能够更好地控制卷积层的行为
'''
def conv2d(input_, input_dim, output_dim,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="conv2d"):
  with tf.compat.v1.variable_scope(name):
    w = tf.compat.v1.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.compat.v1.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input=input_, filters=w, strides=[1, d_h, d_w, 1], padding='SAME')

    biases = tf.compat.v1.get_variable('biases', [output_dim], initializer=tf.compat.v1.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
    return conv
'''
反卷积层 Deconvolution，通常用于上采样
将低分辨率的图像通过上采样变为更高分辨率的图像
'''
def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d", with_w=False):
  with tf.compat.v1.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.compat.v1.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.compat.v1.random_normal_initializer(stddev=stddev))
    try:
      deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])
    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
      deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])
    biases = tf.compat.v1.get_variable('biases', [output_shape[-1]], initializer=tf.compat.v1.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    else:
      return deconv

'''
最大池化层
帮助减少计算量并控制过拟合
缩小一半
'''
def max_pool_2x2(x):
  return tf.nn.max_pool2d(input=x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
'''
全局平均池化层
Kears
1*1
'''
def Global_Average_Pooling(x):
    return keras.layers.GlobalAveragePooling2D()(x)
'''
全连接层
linear 函数（使用 tf.compat.v1.get_variable 和手动控制权重、偏置）
主要处理的是 2D 输入（批量大小和特征维度）

fc 函数（通过 tf.Variable 手动初始化权重）
可以处理不同形状的输入，包括 4D 输入（图像数据）

Fully_connected 函数（使用 Keras API）
适用于 TensorFlow 2.x 主要处理的是 2D 输入（批量大小和特征维度）
'''
def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()

  with tf.compat.v1.variable_scope(scope or "Linear"):
    matrix = tf.compat.v1.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.compat.v1.random_normal_initializer(stddev=stddev))
    bias = tf.compat.v1.get_variable("bias", [output_size],
      initializer=tf.compat.v1.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias
def fc(inputs, output_size, init_bias=0.0, activation_func=tf.nn.relu, stddev=0.01):
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) == 4:
        fc_weights = tf.Variable(
            tf.random.normal([input_shape[1] * input_shape[2] * input_shape[3], output_size], dtype=tf.float32,
                             stddev=stddev),
            name='weights')
        inputs = tf.reshape(inputs, [-1, fc_weights.get_shape().as_list()[0]])
    else:
        fc_weights = tf.Variable(tf.random.normal([input_shape[-1], output_size], dtype=tf.float32, stddev=stddev),
                                 name='weights')

    fc_biases = tf.Variable(tf.constant(init_bias, shape=[output_size], dtype=tf.float32), name='biases')
    fc_layer = tf.matmul(inputs, fc_weights)
    fc_layer = tf.nn.bias_add(fc_layer, fc_biases)
    if activation_func:
        fc_layer = activation_func(fc_layer)
    return fc_layer

def Fully_connected(x, units=512, layer_name='fully_connected'):
    with tf.name_scope(layer_name):
        return tf.keras.layers.Dense(units=units, use_bias=False)(x)

'''
ReLU激活函数
Kears
'''
def Relu(x):
    return tf.nn.relu(x)
'''
Sigmoid激活函数
Kears
'''
def Sigmoid(x) :
    return tf.nn.sigmoid(x)
'''
Leaky ReLU 激活函数
'''
def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)
'''
PReLU（Parametric ReLU）激活函数
带有可学习参数的 ReLU 激活函数，它允许神经网络在负输入时具有不同的斜率
'''
def prelu(_x, scope=None):
    """parametric ReLU activation"""
    with tf.compat.v1.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.compat.v1.get_variable("prelu", shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.compat.v1.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)

'''
模块区域
'''
'''
Squeeze-and-Excitation (SE) 模块
>通道注意力模块
>引用 应用于TCCA第一层
'''
#SE层
def Squeeze_excitation_layer(input_x, out_dim, ratio, layer_name):
    with tf.name_scope(layer_name):
        squeeze = Global_Average_Pooling(input_x)

        excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name + '_fully_connected1')
        excitation = Relu(excitation)
        excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name + '_fully_connected2')
        excitation = Sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])

        scale = input_x * excitation

        return scale
'''
Channel_attention 模块
>空间注意力模块
>自制 应用于TCCA第二层
'''
def Channel_attention(input_yc, layer_name):
    with tf.name_scope(layer_name):
        GAP = tf.reduce_mean(input_yc, axis=-1, keepdims=True)
        GMP = tf.reduce_max(input_yc, axis=-1, keepdims=True)
        GP = tf.concat([GAP, GMP], axis=-1)  # 沿着通道维度拼接
        output = tf.keras.layers.Conv2D(1, (3, 3), strides=(1, 1), padding='same', name="conv_ca1")(GP)
        output = Sigmoid(output)
        return output
'''
Convolutional Block Attention Module (CBAM) 模块
>通道和空间注意力模块
'''
def CBAM_attention_layer(input_x, out_dim, ratio, layer_name):
    with tf.name_scope(layer_name):
        avg_pool = tf.reduce_mean(input_x, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(input_x, axis=[1, 2], keepdims=True)

        excitation1 = Fully_connected(avg_pool, units=out_dim / ratio, layer_name=layer_name + '_fully_connected1')
        excitation1 = Relu(excitation1)
        excitation1 = Fully_connected(excitation1, units=out_dim, layer_name=layer_name + '_fully_connected2')
        excitation2 = Fully_connected(max_pool, units=out_dim / ratio, layer_name=layer_name + '_fully_connected3')
        excitation2 = Relu(excitation2)
        excitation2 = Fully_connected(excitation2, units=out_dim, layer_name=layer_name + '_fully_connected4')

        channel_attention = Sigmoid(excitation1 + excitation2)
        excitation = tf.reshape(channel_attention, [-1, 1, 1, out_dim])

        scale1 = input_x * excitation

        avg_pool2 = tf.reduce_mean(scale1, axis=3, keepdims=True)
        max_pool2 = tf.reduce_max(scale1, axis=3, keepdims=True)

        spatial_attention_input = tf.concat([avg_pool2, max_pool2], axis=3)

        spatial_attention = conv2d(spatial_attention_input, 128, 128, k_h=7, k_w=7, d_h=1, d_w=1,
                                      name=layer_name + '_spatial_conv')
        spatial_attention = Sigmoid(spatial_attention)

        # Apply the spatial attention
        scale2 = scale1 * spatial_attention

        return scale2

'''
Neuron Difference Layer NDL模块
>预处理增强模块
>引用自改 应用于NDL后半段
'''
def sgn_m(x):
    # 创建与 x 形状相同的张量
    zero_lab = tf.zeros_like(x)  # 形状与 x 相同的零张量
    s_t = tf.where(x < 0, zero_lab, x)  # 小于 0 的值设置为 0
    one_lab = tf.ones_like(x)  # 形状与 x 相同的单位张量
    s = tf.where(s_t > 1, one_lab, s_t)  # 大于 1 的值设置为 1
    return s
def piece_function_org(x_m, para_m, input, M):
    shape = tf.shape(x_m)  # 获取输入张量的形状
    b = shape[0]
    c = shape[3]
    w = shape[2]
    h = shape[1]
    r_m = tf.reshape(para_m[:, 0], (b, 1, 1, c))  # 选择第一个参数并重塑
    #r_m = tf.tile(r_m, (1, h, w, 1))  # 扩展到
    r_m = tf.multiply(r_m, input)

    for i in range(M - 1):
        para = tf.reshape(para_m[:, i + 1] - para_m[:, i], (b, 1, 1, c))  # 计算分段间的差值
        #para = tf.tile(para, (1, h, w, 1))  # 扩展到
        para = tf.multiply(para ,input)
        threshold = M * x_m - i  # 计算阈值
        r_m += para * sgn_m(threshold)  # 更新 r_m

    r_m = r_m + input

    return r_m

'''
Neuron Difference Layer NDL模块
>预处理增强模块
>AI修改 未验证其正确性

def sgn_m(x):
    # 使用 tf.clip_by_value 来避免不必要的张量创建
    return tf.clip_by_value(tf.sign(x), 0, 1)


def piece_function_org(x_m, para_m, input, M):
    shape = tf.shape(x_m)  # 获取输入张量的形状
    b, h, w, c = shape[0], shape[1], shape[2], shape[3]

    r_m = tf.reshape(para_m[:, 0], (b, 1, 1, c))  # 选择第一个参数并重塑
    r_m = tf.multiply(r_m, input)  # 乘以 input

    # 使用广播机制代替 for 循环
    para_diff = para_m[:, 1:] - para_m[:, :-1]  # 计算参数差值
    thresholds = M * x_m - tf.range(M - 1, dtype=tf.float32)  # 计算阈值
    thresholds = tf.reshape(thresholds, (b, h, w, M - 1))  # 调整形状以便广播

    # 计算阈值对应的符号值
    signs = tf.cast(sgn_m(thresholds), tf.float32)

    # 通过广播机制计算最终的 r_m
    para_diff = tf.reshape(para_diff, (b, 1, 1, M - 1))  # 重新调整 para_diff 的形状
    r_m += tf.reduce_sum(para_diff * signs, axis=-2)  # 对每个 M 的部分进行求和

    r_m += input  # 最后加上 input

    return r_m
'''
'''
>空间衰减归一层
>自创
'''
def custom_transform(x):
    # 小于0.3的归0, 0.3到0.6之间的归0.5, 大于0.6的归1
    x = tf.where(x < 0.3, tf.zeros_like(x), x)  # 小于0.3的设置为0
    x = tf.where((x >= 0.3) & (x < 0.6), tf.ones_like(x) * 0.5, x)  # 0.3到0.6之间设置为0.5
    x = tf.where(x >= 0.6, tf.ones_like(x), x)  # 大于0.6的设置为1
    return x



