from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr, mean_squared_error as mse
import os
from PIL import Image
import numpy as np
import shutil

count = 0
count_mse = 0
count_psnr = 0
count_ssim = 0
scores = []
# 获取当前工作目录
current_directory = os.getcwd()
data = 'LSUI'
current_directory = os.path.join(current_directory, data)


# 两个文件夹的名称
folder1 = 'reference'

#folder2 = 'enhancement'
#folder2 = 'get images_Water-Net'
#folder2 = 'get images_Water-Net-final'
#folder2 = 'get images_SFCR-Net'
folder2 = 'get images_SFCR-Net-2'

# 获取两个文件夹的路径
folder1_path = os.path.join(current_directory, folder1)
folder2_path = os.path.join(current_directory, folder2)

# 获取文件夹中的所有文件
folder1_files = os.listdir(folder1_path)
folder2_files = os.listdir(folder2_path)

# 筛选出图像文件
image_files1 = [file for file in folder1_files if file.endswith('.png')]
image_files2 = [file for file in folder2_files if file.endswith('.png')]

# 读取图像并计算均方误差
for image_file1, image_file2 in zip(image_files1, image_files2):
    image_path1 = os.path.join(folder1_path, image_file1)
    image_path2 = os.path.join(folder2_path, image_file2)

    count += 1
    # 读取图像
    img1 = Image.open(image_path1).convert('RGB')
    img2 = Image.open(image_path2).convert('RGB')

    # 获取第一张图片的尺寸
    img1_size = img1.size  # (宽度, 高度)
    # 调整第二张图片的尺寸
    img2 = img2.resize(img1_size, Image.LANCZOS)
    img1 = np.array(img1)
    img2 = np.array(img2)

    # 计算 PSNR 和 SSIM
    psnr_value = psnr(img1, img2)
    ssim_value, _ = ssim(img1, img2, multichannel=True, full=True)
    mse_value = mse(img1, img2)

    # 计算总得分（你可以根据需要调整得分的加权）
    total_score = psnr_value + ssim_value * 30 - mse_value * 0.01 # 例如，简单加权

    # 将得分和文件名存储在列表中
    scores.append((image_file1, image_file2, psnr_value, ssim_value, mse_value, total_score))

    count_mse += mse_value
    count_psnr += psnr_value
    count_ssim += ssim_value

    print(count)
    print('PSNR：{}，SSIM：{}，MSE：{}'.format(psnr_value, ssim_value, mse_value))

get_mse = count_mse / count
get_psnr = count_psnr / count
get_ssim = count_ssim / count
print('平均 PSNR：{}，平均 SSIM：{}，平均 MSE：{}'.format(get_psnr, get_ssim, get_mse))