# --- utils.py --------------------------------------------------------------
import os
import torch, math
from torchvision.utils import save_image
import pytorch_ssim   # pip install pytorch-ssim
from torchmetrics.image import StructuralSimilarityIndexMeasure
from sr3 import *

def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return -10.0 * torch.log10(mse + 1e-10)

ssim_fn = StructuralSimilarityIndexMeasure(data_range=2.0).to(DEVICE)
def evaluate(model, loader, step, save_dir="samples"):
    model.eval()
    psnr_acc, ssim_acc, n = 0., 0., 0
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for i, (lr, hr) in enumerate(loader):
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)
            b = lr.size(0)
            noise_img = torch.randn_like(hr)
            t = torch.full((b,), TIMESTEPS - 1, device=DEVICE, dtype=torch.long)
            # deterministic DDPM: just run p_sample_loop once per image
            sr = p_sample_loop(model, hr.shape, lr)
            psnr_acc += psnr(sr, hr).item() * b
            ssim_acc += ssim_fn(sr.clamp(-1,1), hr.clamp(-1, 1)).item() * b
            n += b
            if i == 0:   # save visual grid
                grid = torch.cat([lr.repeat(1,1,UPSCALE_FACTOR,UPSCALE_FACTOR), sr, hr], dim=0)
                save_image(grid*0.5+0.5, f"{save_dir}/step_{step:07d}.png", nrow=b)
    return psnr_acc / n, ssim_acc / n

def save_ckpt(model, opt, step, path):
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "step": step
    }, path)
