import random
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F

class SingleImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, intrinsics, transform=None, repeat=1000):
        self.img_path = img_path  # 只有一张图
        self.K = intrinsics
        self.transform = transform
        self.repeat = repeat  # 总共生成的样本数

    def __len__(self):
        return self.repeat  # 生成 repeat 个样本

    def __getitem__(self, idx):
        from PIL import Image
        import random
        from torchvision.transforms import functional as F

        img = Image.open(self.img_path).convert('RGB')
        target = img

        # 随机几何变换
        angle = random.uniform(-2, 2)
        translate = (random.uniform(-0.05,0.05)*img.width,
                     random.uniform(-0.05,0.05)*img.height)
        scale = random.uniform(0.98,1.02)
        src1 = F.affine(img, angle, translate, scale, shear=0)
        src2 = F.affine(img, -angle, (-translate[0], -translate[1]), 1/scale, shear=0)

        if self.transform:
            target = self.transform(target)
            src1 = self.transform(src1)
            src2 = self.transform(src2)

        return {'tgt': target, 'src1': src1, 'src2': src2, 'K': self.K}
