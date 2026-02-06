# --- data.py ---------------------------------------------------------------
import glob, os, random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch
from sr3 import *

def identity(x):
    return x

class SRFolderDataset(Dataset):
    """
    Expects a directory full of HR PNG/JPG images.
    LR is generated on-the-fly with bicubic ↓.
    """
    def __init__(self,
                 root_dir: str,
                 hr_size: int = HR_IMG_SIZE,
                 lr_size: int = LR_IMG_SIZE,
                 training: bool = True):
        super().__init__()
        self.paths = sorted(
            sum([glob.glob(os.path.join(root_dir, ext))
                 for ext in ("*.png", "*.jpg", "*.jpeg")], [])
        )
        self.training = training
        self.hr_tf = T.Compose([
            # random crops / flips only during training
            T.RandomCrop(hr_size) if training else T.CenterCrop(hr_size),
            T.RandomHorizontalFlip() if training else T.Lambda(identity),
            T.ToTensor(),                       # (0,1)
            T.Normalize(0.5, 0.5)               # (-1,1)
        ])
        self.lr_tf = T.Compose([
            T.Resize(lr_size, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img_hr = Image.open(path).convert("RGB")
        img_hr = self.hr_tf(img_hr)
        img_lr = self.lr_tf(T.ToPILImage()( (img_hr * 0.5 + 0.5).clamp(0,1) ))
        return img_lr, img_hr
