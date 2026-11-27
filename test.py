import cv2
import numpy as np
import matplotlib.pyplot as plt
'''
10_img_.png 25_img_.png 70_img_.png 
'''
# 读取图像
image1 = cv2.imread('img/training/UIEB/input_train/25_img_.png', cv2.IMREAD_COLOR)
image2 = cv2.imread('evaluate/UIEB/get images_Water-Net/25_img_.png', cv2.IMREAD_COLOR)
image3 = cv2.imread('evaluate/UIEB/reference/25_img_.png', cv2.IMREAD_COLOR)

# 将图像从BGR转为RGB（OpenCV读取图片是BGR格式）
image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
image3_rgb = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)

# 转换为灰度图像
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
gray_image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)

# 应用Canny边缘检测
edges1 = cv2.Canny(gray_image1, 100, 200)
edges2 = cv2.Canny(gray_image2, 100, 200)
edges3 = cv2.Canny(gray_image3, 100, 200)

# 显示原图与边缘检测结果
plt.figure(figsize=(15, 10))

# 显示原图1与边缘图像1
plt.subplot(3, 3, 1)
plt.imshow(image1_rgb)
plt.title('Original Image 1')
plt.axis('off')

plt.subplot(3, 3, 2)
plt.imshow(gray_image1, cmap='gray')
plt.title('Gray Image 1')
plt.axis('off')

plt.subplot(3, 3, 3)
plt.imshow(edges1, cmap='gray')
plt.title('Canny Edges 1')
plt.axis('off')

# 显示原图2与边缘图像2
plt.subplot(3, 3, 4)
plt.imshow(image2_rgb)
plt.title('Enhanced by Water-Net')
plt.axis('off')

plt.subplot(3, 3, 5)
plt.imshow(gray_image2, cmap='gray')
plt.title('Gray Image 2')
plt.axis('off')

plt.subplot(3, 3, 6)
plt.imshow(edges2, cmap='gray')
plt.title('Canny Edges 2')
plt.axis('off')

# 显示原图3与边缘图像3
plt.subplot(3, 3, 7)
plt.imshow(image3_rgb)
plt.title('Reference Image')
plt.axis('off')

plt.subplot(3, 3, 8)
plt.imshow(gray_image3, cmap='gray')
plt.title('Gray Image 3')
plt.axis('off')

plt.subplot(3, 3, 9)
plt.imshow(edges3, cmap='gray')
plt.title('Canny Edges 3')
plt.axis('off')

# 调整布局并显示
plt.tight_layout()
plt.show()


