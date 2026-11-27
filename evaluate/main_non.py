import cv2
import numpy as np
import os
import math

count = 0
count_uciqe = 0
count_uiqm = 0

def uicm(img):
    b, r, g = cv2.split(img)
    RG = r - g
    YB = (r + g) / 2 - b

    # Flatten and sort arrays
    RG_list = np.sort(RG.flatten())
    YB_list = np.sort(YB.flatten())

    m, n = RG.shape
    K = m * n
    alpha_L = 0.1
    alpha_R = 0.1

    T_alpha_L = math.ceil(alpha_L * K)
    T_alpha_R = math.floor(alpha_R * K)

    # Compute U_RG and sigma2_RG
    RG_range = RG_list[T_alpha_L:K - T_alpha_R]
    U_RG = np.mean(RG_range)
    sigma2_RG = np.mean((RG_list - U_RG) ** 2)

    # Compute U_YB and sigma2_YB
    YB_range = YB_list[T_alpha_L:K - T_alpha_R]
    U_YB = np.mean(YB_range)
    sigma2_YB = np.mean((YB_list - U_YB) ** 2)

    Uicm = -0.0268 * np.sqrt(U_RG ** 2 + U_YB ** 2) + 0.1586 * np.sqrt(sigma2_RG + sigma2_YB)
    return Uicm


def EME(rbg, L):
    m, n = rbg.shape
    number_m = math.floor(m / L)
    number_n = math.floor(n / L)

    E = 0
    for i in range(number_m):
        for t in range(number_n):
            A1 = rbg[i * L:(i + 1) * L, t * L:(t + 1) * L]
            rbg_min = np.amin(A1)
            rbg_max = np.amax(A1)
            rbg_ratio = rbg_max / (rbg_min + 1e-5)  # Avoid division by zero
            E += np.log1p(np.maximum(rbg_ratio - 1, 1e-5))  # More stable log calculation

    E_sum = 2 * E / (number_m * number_n)
    return E_sum


def UICONM(rbg, L):
    if L <= 0:
        raise ValueError("块大小 L 必须是正整数。")

    m, n = rbg.shape[:2]
    if m < L or n < L:
        raise ValueError("块大小 L 大于图像的尺寸。")

    number_m = math.floor(m / L)
    number_n = math.floor(n / L)

    if number_m == 0 or number_n == 0:
        raise ValueError("块大小 L 对于图像尺寸来说太大。")

    logAMEE = 0
    for i in range(number_m):
        for t in range(number_n):
            A1 = rbg[i * L:(i + 1) * L, t * L:(t + 1) * L]
            rbg_min = np.amin(A1).astype(float)
            rbg_max = np.amax(A1).astype(float)

            plip_add = rbg_max + rbg_min - rbg_max * rbg_min / 1026
            if 1026 - rbg_min > 0:
                plip_del = 1026 * (rbg_max - rbg_min) / (1026 - rbg_min)
                if plip_del > 0 and plip_add > 0:
                    local_a = plip_del / plip_add
                    local_b = np.log((plip_del / plip_add) + 1)  # 1e-10 用于避免 log(0)
                    phi = local_a * local_b
                    logAMEE += phi

    ## 调试打印最终值
    #print(f"Intermediate logAMEE value: {logAMEE}")

    # 确保 logAMEE 处于合理范围
    logAMEE = max(0, logAMEE)  # 将负值截断为 0

    # 计算最终结果
    if number_n * number_m * 1026 > 0:
        logAMEE = 1026 - 1026 * ((1 - logAMEE / (number_n * number_m * 1026 / 64)) ** (1 / (number_n * number_m / 64)))
    else:
        print("error")

    return logAMEE


# 获取当前工作目录
# 获取当前工作目录
current_directory = os.getcwd()
data = 'LSUI'
current_directory = os.path.join(current_directory, data)

# 文件夹的名称
#folder = 'reference'
#folder = 'enhancement'
#folder = 'get images_Water-Net'
folder = 'get images_Water-Net-final'

# 获取两个文件夹的路径
folder_path = os.path.join(current_directory, folder)

# 获取文件夹中的所有文件
folder_files = os.listdir(folder_path)

# 筛选出图像文件
image_files = [file for file in folder_files if file.endswith('.png')]

# 读取图像并计算均方误差
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)

    count += 1
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at the path: {image_path}")

    # 转换到 HSV 色彩空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # 计算色度的标准差
    delta = np.std(H) / 180  # 色度的标准差

    # 计算饱和度的平均值
    mu = np.mean(S) / 255  # 饱和度的平均值

    # 计算所需像素的个数
    n, m = V.shape
    number = int(n * m / 100)  # 所需像素的个数

    # 计算亮度的对比度
    V_normalized = V / 255.0

    # 获取最大值和最小值的像素索引
    max_values = np.partition(V_normalized.flatten(), -number)[-number:]
    top = np.mean(max_values)

    min_values = np.partition(V_normalized.flatten(), number)[:number]
    bottom = np.mean(min_values)

    conl = top - bottom  # 对比度

    # 计算 UCiQE 指标
    uciqe = 0.4680 * delta + 0.2745 * conl + 0.2575 * mu
    count_uciqe = count_uciqe + uciqe

    r, b, g = cv2.split(image)

    Uicm = uicm(image)

    EME_r = EME(r, 8)
    EME_b = EME(b, 8)
    EME_g = EME(g, 8)
    Uism = 0.299 * EME_r + 0.114 * EME_b + 0.587 * EME_g

    img = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
    Uiconm = UICONM(img, 16)

    uiqm = 0.0282 * Uicm + 0.2953 * Uism + 3.5753 * Uiconm
    count_uiqm = count_uiqm + uiqm

    print(count)
    #print(f"Delta: {delta:.4f}")
    #print(f"Contrast: {conl:.4f}")
    #print(f"Mean Saturation: {mu:.4f}")
    print(f"UCIQE: {uciqe:.4f}")

    print(f"Uicm: {Uicm:.4f}")
    print(f"Uism: {Uism:.4f}")
    print(f"Uiconm: {Uiconm:.4f}")
    print(f"UIQM: {uiqm:.4f}")

get_uciqe = count_uciqe / count
get_uiqm = count_uiqm / count
duibi_uciqe = count_uciqe / 890
duibi_uiqm = count_uiqm / 890
print('平均 UCIQE：{}，平均 UIQM：{}'.format(get_uciqe, get_uiqm))
print('平均 UCIQE：{}，平均 UIQM：{}'.format(duibi_uciqe, duibi_uiqm))