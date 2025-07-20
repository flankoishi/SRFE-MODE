import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image

# **1. 加载模型**
model = models.resnet101().eval()
target_layer = model.layer4[-1]

# **2. 读取原始图像**
img_path = "C:/Users/y'l'r/Desktop/cam/1.jpg"
img = cv2.imread(img_path)

# **3. 检查图像是否正确加载**
if img is None:
    raise ValueError(f"图像加载失败，请检查路径: {img_path}")

# **4. OpenCV 读取的 BGR 转 RGB**
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# **5. 归一化，并转换为 float32**
img = img.astype(np.float32) / 255.0

# **6. 预处理为 PyTorch 需要的格式**
input_tensor = preprocess_image(img)
input_tensor = input_tensor.to(torch.float32)

# **7. 计算原始 Grad-CAM**
cam = GradCAM(model=model, target_layers=[target_layer])
grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)

cv2.imwrite("grad_cam_original_2.jpg", visualization)

# **8. 定义数据增强**
transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.05, hue=0.05)
])

# **9. 转换为 PIL 进行增强**
img_pil = Image.fromarray((img * 255).astype(np.uint8))  # 转换为 uint8
enhanced_img_pil = transform(img_pil)  # 进行增强

# **10. 转换回 numpy 格式**
enhanced_img = np.array(enhanced_img_pil).astype(np.float32) / 255.0  # 归一化

# **11. 计算增强后 Grad-CAM**
enhanced_tensor = preprocess_image(enhanced_img)
grayscale_cam_enhanced = cam(input_tensor=enhanced_tensor, targets=None)[0]
visualization_enhanced = show_cam_on_image(enhanced_img, grayscale_cam_enhanced, use_rgb=True)

cv2.imwrite("grad_cam_enhanced_2.jpg", visualization_enhanced)
