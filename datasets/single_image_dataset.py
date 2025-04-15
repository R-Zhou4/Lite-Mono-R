import random
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F

class SingleImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, intrinsics, transform=None):
        self.img_list = img_list
        self.K = intrinsics  # 相机内参
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        # 原图当 target
        target = img
        # 随机几何变换参数
        angle = random.uniform(-2, 2)      # ±2°
        translate = (random.uniform(-0.05,0.05)*img.width,
                     random.uniform(-0.05,0.05)*img.height)
        scale = random.uniform(0.98,1.02)  # ±2%
        # 生成 source1, source2
        src1 = F.affine(img, angle= angle, translate=translate,
                        scale=scale, shear=0)
        src2 = F.affine(img, angle=-angle, translate=(-translate[0],-translate[1]),
                        scale=1/scale, shear=0)
        # 光度增强（可选）
        if self.transform:
            target = self.transform(target)
            src1    = self.transform(src1)
            src2    = self.transform(src2)
        return {'tgt': target, 'src1': src1, 'src2': src2, 'K': self.K}
