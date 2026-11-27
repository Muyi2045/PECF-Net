from .vgg import *
import tensorflow as tf
import math
import cv2
import numpy as np

'''
总loss设置
'''
def loss1(label_image, enhanced_image):
    loss = 0.05 * loss_vgg(label_image, enhanced_image) + loss_l1(label_image, enhanced_image)
    return loss

'''
基于VGG网络特征的感知损失
Perceptual Loss
'''
def loss_vgg(label_image, enhanced_image):
    vgg_dir = 'loss/vgg_pretrained/imagenet-vgg-verydeep-19.mat'
    CONTENT_LAYER = 'relu5_4'
    label_vgg = net(vgg_dir, preprocess(label_image * 255))
    enhanced_vgg = net(vgg_dir, preprocess(enhanced_image *255))
    loss_vgg = tf.reduce_mean(tf.square(enhanced_vgg[CONTENT_LAYER] - label_vgg[CONTENT_LAYER]))
    #print(loss_vgg)
    return loss_vgg
'''
均值绝对误差
L1 loss
'''
def loss_l1(label_image, enhanced_image):
    loss_l1 = tf.reduce_mean(tf.abs(label_image - enhanced_image))
    #print(loss_l1)
    return loss_l1
'''
均值绝对误差
L2 loss
'''
def loss_l2(label_image, enhanced_image):
    loss_l2 = tf.reduce_mean(tf.square(label_image - enhanced_image))
    #print(loss_l1)
    return loss_l2
'''
SSIM 损失函数
结构相似性指数
'''
def ssim_loss(y_true, y_pred, max_val=1.0):
    # 计算批次中每张图像的 SSIM 值，范围是 [-1, 1]。
    ssim_value = tf.image.ssim(y_true, y_pred, max_val=max_val)
    # SSIM 值在 [-1, 1] 范围内，我们希望损失值是正的，因此返回 1 - SSIM。
    ssim_loss_value = 1 - ssim_value
    # 对批次求平均损失
    return tf.reduce_mean(ssim_loss_value)

'''
基于 HSV 颜色空间的 L1 损失
'''
def hsv_loss(y_true, y_pred):
    pi = math.pi
    # RGB -> HSV 转换
    pred_hsv = tf.image.rgb_to_hsv(y_pred)
    gt_hsv = tf.image.rgb_to_hsv(y_true)
    # 提取 HSV 通道
    hi, si, vi = tf.split(pred_hsv, num_or_size_splits=3, axis=-1)
    hj, sj, vj = tf.split(gt_hsv, num_or_size_splits=3, axis=-1)
    # 色相（H）处理，乘以 2π 转为弧度
    hipi = hi * pi * 2
    hjpi = hj * pi * 2
    # 计算色相的余弦和正弦值
    coshp = tf.cos(hipi)
    sinhg = tf.sin(hjpi)
    # 计算饱和度和明度的乘积
    sv_p = si * vi
    sv_g = sj * vj
    # 预测和真实值的加权计算
    temp_pred = coshp * sv_p
    temp_gt = sinhg * sv_g
    # 计算 L1 损失
    loss = tf.reduce_mean(tf.abs(temp_pred - temp_gt))

    return loss
'''
Canny算法边缘强度
'''
def edge_loss(predicted_image, target_image):
    # 通过Canny算法提取目标边缘
    target_edges = tf.image.sobel_edges(target_image)

    # 通过Canny算法提取网络预测边缘
    predicted_edges = tf.image.sobel_edges(predicted_image)

    # 计算两张边缘图的差异（可以使用L2损失或交叉熵等方式）
    edge_loss = tf.reduce_mean(tf.square(predicted_edges - target_edges))
    return edge_loss