import os
import tensorflow as tf
import numpy as np
from PIL import Image
import glob
import imageio
'''
图像归一化处理
'''
def transform(images):
  return np.array(images)/127.5 - 1.
def inverse_transform(images):
  return (images+1)/2
'''
路径加载
'''
def prepare_data(data, dataset):
  data_dir = os.path.join(os.getcwd(), "img", "training", data, dataset)
  data = glob.glob(os.path.join(data_dir, "*.png"))
  data = data + glob.glob(os.path.join(data_dir, "*.jpg"))
  return data
def prepare_data_test(data, dataset):
  data_dir = os.path.join(os.getcwd(), "img", "testing", data, dataset)
  data = glob.glob(os.path.join(data_dir, "*.png"))
  data = data + glob.glob(os.path.join(data_dir, "*.jpg"))
  return data
'''
读取图像
'''
def imread(path, height, width):
  target_shape = (height, width)
  img = Image.open(path)
  image_resized = img.resize(target_shape)
  return np.array(image_resized).astype(np.float32)
'''
训练图像
'''
def get_image(image_path, height, width):
  image = imread(image_path, height, width)
  return image/255
'''
保存图像
'''
def imsave_lable(image, path):
  image_uint8 = (image * 255).astype(np.uint8)
  image_pil = Image.fromarray(image_uint8)
  image_pil.save(path)

def image_save(image, path):
  return imageio.imsave(path, image*255)