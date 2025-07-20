from skimage.metrics import structural_similarity as ssim
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image

# 计算 SSIM
grad_cam_1 = cv2.imread('D:/PycharmProjects/shibie projext/grad_cam_enhanced_1.jpg')
grad_cam_2 = cv2.imread('D:/PycharmProjects/shibie projext/grad_cam_original_2.jpg')

ssim_score = ssim(grad_cam_1, grad_cam_2, data_range=1.0, win_size=3,multichannel=True,channel_axis=-1)
print("SSIM Score:", ssim_score)

# 拉平 Grad-CAM 图像
grad_cam_1_flat = grad_cam_1.flatten().reshape(1, -1)
grad_cam_2_flat = grad_cam_2.flatten().reshape(1, -1)

cos_sim = cosine_similarity(grad_cam_1_flat, grad_cam_2_flat)[0, 0]
print("Cosine Similarity:", cos_sim)