import random
from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

class SingleImageDataset(Dataset):
    def __init__(self, image_path, height=192, width=640):
        """初始化，只需要一张图片的路径"""
        self.image_path = image_path
        self.height = height  # Lite-Mono 默认输入尺寸
        self.width = width
        # 基本的图片处理：调整大小、转成张量、归一化
        self.base_transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 标准归一化
        ])
    
    def __getitem__(self, index):
        """每次调用，生成一张图片的增强版本"""
        # 加载单张图片
        img = Image.open(self.image_path).convert('RGB')
        
        # 随机增强参数
        shift_x = random.uniform(-10, 10)  # 水平随机平移 -10 到 10 像素
        shift_y = random.uniform(-5, 5)    # 垂直随机平移 -5 到 5 像素
        
        # 生成伪连续帧
        # t-1 帧：向左平移
        img_t_minus_1 = T.functional.affine(
            img, angle=0, translate=(-shift_x, shift_y), scale=1.0, shear=0
        )
        # t 帧：原图
        img_t = img
        # t+1 帧：向右平移
        img_t_plus_1 = T.functional.affine(
            img, angle=0, translate=(shift_x, shift_y), scale=1.0, shear=0
        )
        
        # 应用基本处理（调整大小、归一化等）
        img_t_minus_1 = self.base_transform(img_t_minus_1)
        img_t = self.base_transform(img_t)
        img_t_plus_1 = self.base_transform(img_t_plus_1)
        
        # 返回 Lite-Mono 期望的格式
        return {
            ('color', -1, 0): img_t_minus_1,  # 前一帧
            ('color', 0, 0): img_t,          # 当前帧
            ('color', 1, 0): img_t_plus_1    # 后一帧
        }
    
    def __len__(self):
        """返回一个较大的数字，让训练能多次迭代，每次生成不同增强"""
        return 1000  # 随意设的大数字，够训练用