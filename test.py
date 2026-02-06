# --- test.py ---------------------------------------------------------------
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
import requests
import warnings

# Suppress a specific warning from torchvision.utils.save_image
warnings.filterwarnings("ignore", message="The given buffer is not writable, and PyTorch does not support non-writable tensors.")


# Import from your project files
from sr3 import UNetSR3_Refined, p_sample_loop, DEVICE, HR_IMG_SIZE, LR_IMG_SIZE, UPSCALE_FACTOR, IMG_CHANNELS
from data import SRFolderDataset
from utils import psnr, ssim_fn

# --- Configuration ---
CKPT_DIR = "ckpts"
TEST_DATA_DIR = "lhq_split/test"
OUTPUT_DIR = "test_results"
BATCH_SIZE = 5  # Number of images to test and include in the grid
FSRCNN_MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_extra/main/testdata/dnn/fsrcnn_x2.pb"
FSRCNN_MODEL_PATH = "FSRCNN_x2.pb"


# --- Helper function to download models ---
def download_file_if_not_exists(url, filename, gdrive_id=None):
    """Downloads a file from a URL if it doesn't already exist."""
    if os.path.exists(filename):
        print(f"{filename} already exists. Skipping download.")
        return

    if gdrive_id:
        print(f"Downloading {filename} from Google Drive...")
        try:
            import gdown
            gdown.download(id=gdrive_id, output=filename, quiet=False)
        except Exception as e:
            raise RuntimeError(f"Could not download from Google Drive. Install gdown (`pip install gdown`) or download manually. Error: {e}")
    else:
        print(f"Downloading {filename} from {url}...")
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            exit()

# --- Model Loading Functions ---
def load_sr3_model(ckpt_path):
    """Loads the SR3 UNet model from a checkpoint."""
    # NOTE: Ensure these parameters match what was used for training!

    model = UNetSR3_Refined(

        in_channels_hr=IMG_CHANNELS,

        in_channels_lr=IMG_CHANNELS,

        out_channels=IMG_CHANNELS,

        init_dim=64,

        dim_mults=(1, 2, 2, 4),

        num_resnet_blocks_per_level=2,

        time_emb_dim_input=64,

        time_emb_dim_mlp=256,

        use_attention_at_resolutions=(32, 16),  # e.g. (16,) or (32, 16) if model is larger

        resnet_groups=8,

        attention_heads=8,

        attention_head_dim=64,

        dropout_rate=0.0,

        hr_size=HR_IMG_SIZE

    ).to(DEVICE)
    data = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(data["model"])
    model.eval()
    return model
def load_esrgan_model():
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet

    # path where you saved the weight (or leave it as the URL)
    ckpt = "RealESRGAN_x2plus.pth"        # or the full https://…/RealESRGAN_x2plus.pth

    net = RRDBNet(num_in_ch=3, num_out_ch=3,
                  num_feat=64, num_block=23,
                  num_grow_ch=32, scale=2)

    model = RealESRGANer(
        scale      = 2,
        model_path = ckpt,                # can be local file OR https:// URL
        model      = net,
        tile       = 0,                   # 128×128 fits easily → no tiling
        tile_pad   = 10,
        half       = False               # keep FP32 unless you really want FP16
    )
    return model
# def load_esrgan_model():
#     """
#     Original ESRGAN (2018) RRDB x4 model.
#     This version manually renames the keys from the checkpoint to match the model architecture.
#     """
#     from basicsr.archs.rrdbnet_arch import RRDBNet
#     import tempfile
#
#     ESRGAN_MODEL_GDRIVE_ID = "1pJ_T-V1dpb1ewoEra1TGSWl5e6H7M4NN"
#     ESRGAN_MODEL_PATH = os.path.join(tempfile.gettempdir(), "RRDB_ESRGAN_x4.pth")
#
#     download_file_if_not_exists(None, ESRGAN_MODEL_PATH, gdrive_id=ESRGAN_MODEL_GDRIVE_ID)
#
#     # Instantiate the model architecture
#     model = RRDBNet(
#         num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
#     ).to(DEVICE)
#
#     # Load the state dict from the file
#     state_from_file = torch.load(ESRGAN_MODEL_PATH, map_location=DEVICE)
#
#     # --- THIS IS THE FIX ---
#     # Create a new, empty dictionary for the corrected keys
#     corrected_state = {}
#     # Iterate over the keys in the file's state dict and rename them
#     for key, value in state_from_file.items():
#         new_key = key.replace('RRDB_trunk.', 'body.') \
#             .replace('trunk_conv.', 'conv_body.') \
#             .replace('upconv1.', 'conv_up1.') \
#             .replace('upconv2.', 'conv_up2.') \
#             .replace('HRconv.', 'conv_hr.') \
#             .replace('.RDB', '.rdb')
#         corrected_state[new_key] = value
#     # --- END OF FIX ---
#
#     # Load the corrected state dict into the model
#     model.load_state_dict(corrected_state, strict=True)
#     model.eval()
#     return model

def load_fsrcnn_model():
    """Loads the FSRCNN model using OpenCV's DNN module."""
    download_file_if_not_exists(FSRCNN_MODEL_URL, FSRCNN_MODEL_PATH)
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(FSRCNN_MODEL_PATH)
    sr.setModel("fsrcnn", UPSCALE_FACTOR)
    return sr

# --- Inference Functions ---
def run_fsrcnn(model, lr_imgs):
    sr_results = []
    for i in range(lr_imgs.size(0)):
        lr_np = (lr_imgs[i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255.0
        lr_np = lr_np.astype(np.uint8)
        lr_np_bgr = cv2.cvtColor(lr_np, cv2.COLOR_RGB2BGR)
        sr_np_bgr = model.upsample(lr_np_bgr)
        sr_np_rgb = cv2.cvtColor(sr_np_bgr, cv2.COLOR_BGR2RGB)
        sr_tensor = torch.from_numpy(sr_np_rgb.astype(np.float32)).permute(2, 0, 1)
        sr_tensor = (sr_tensor / 255.0) * 2.0 - 1.0
        sr_results.append(sr_tensor)
    return torch.stack(sr_results).to(DEVICE)
# --- In test.py, find the run_esrgan function and modify it ---

def run_esrgan(model, hr_imgs):
    """
    256 → 128  ↓,  RealESRGAN ×2,  256 tensor in [-1,1] RGB
    """
    lr_128 = F.interpolate(hr_imgs, scale_factor=0.5, mode="area")  # B×3×128×128
    sr_tensors = []

    for img in lr_128:                                  # loop over batch
        # tensor [-1,1] → uint8 HWC BGR 0-255
        img_bgr = (
            img.add(1).mul(127.5)                       # [-1,1] → [0,255]
            .clamp(0, 255)
            .permute(1, 2, 0)                        # CHW → HWC
            .cpu()
            .numpy()
            .astype("uint8")[..., ::-1]              # RGB → BGR
        )

        sr_bgr, _ = model.enhance(img_bgr, outscale=2)  # uint8 HWC BGR 0-255

        # BGR back to RGB, scale to [-1,1], tensor CHW
        sr_rgb = sr_bgr[..., ::-1].astype("float32") / 255.0
        sr_tensor = (
            torch.from_numpy(sr_rgb)
            .permute(2, 0, 1)                      # HWC → CHW
            .to(hr_imgs.device)
            .mul(2).sub(1)                         # [0,1] → [-1,1]
        )
        sr_tensors.append(sr_tensor)

    return torch.stack(sr_tensors)                      # B×3×256×256
# --- Main Execution ---
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("--- Loading Models ---")
    ckpt_files = sorted(glob.glob(os.path.join(CKPT_DIR, "*.pth")))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found in {CKPT_DIR}. Please run train.py first.")
    latest_ckpt = ckpt_files[-1]
    print(f"Loading SR3 model from: {latest_ckpt}")
    sr3_model = load_sr3_model(latest_ckpt)
    print("Loading FSRCNN...")
    fsrcnn_model = load_fsrcnn_model()
    print("Loading ESRGAN...")
    esrgan_model = load_esrgan_model()
    print("All models loaded.\n")

    print("--- Loading Test Data ---")
    test_dataset = SRFolderDataset(root_dir=TEST_DATA_DIR, hr_size=HR_IMG_SIZE, training=False)
    # ------------------------------------------------------------------
    METRIC_BATCH = 64        # ← crunch the whole test set this many at a time
    GRID_SAMPLES = 5         # ← always show exactly 5 images in the grid
    # ------------------------------------------------------------------

    # --- metric DataLoader (big batch, no shuffling) ------------------
    metric_loader = DataLoader(
        test_dataset,
        batch_size   = METRIC_BATCH,
        shuffle      = False,
        num_workers  = 16,          # or more if you have CPUs to spare
        pin_memory   = True        # speeds up host→GPU copies
    )

    # --- grid DataLoader (just 5 random images) -----------------------
    grid_loader = DataLoader(
        test_dataset,
        batch_size   = GRID_SAMPLES,
        shuffle      = True,       # different set every run
        num_workers  = 0
    )
    try:
        lr_imgs, hr_imgs = next(iter(grid_loader))
    except StopIteration:
        print(f"ERROR: No images found in {TEST_DATA_DIR}. Please run split_lhq.py.")
        return
    lr_imgs, hr_imgs = lr_imgs.to(DEVICE), hr_imgs.to(DEVICE)
    print(f"Loaded a batch of {lr_imgs.shape[0]} test images.\n")

    print("--- Running Inference ---")
    bicubic_sr = F.interpolate(lr_imgs, scale_factor=UPSCALE_FACTOR, mode='bicubic', align_corners=False)
    fsrcnn_sr = run_fsrcnn(fsrcnn_model, lr_imgs)
    esrgan_sr = run_esrgan(esrgan_model, hr_imgs)
    print("Running SR3 sampling loop (this may take a moment)...")
    with torch.no_grad():
        sr3_sr = p_sample_loop(sr3_model, hr_imgs.shape, lr_imgs)
    print("Inference complete.\n")

    print("--- Performance Metrics ---")
    models_and_results = {"Bicubic": bicubic_sr, "FSRCNN": fsrcnn_sr, "ESRGAN": esrgan_sr, "SR3 (Ours)": sr3_sr}
    for name, sr_img in models_and_results.items():
        sr_clamped = torch.clamp(sr_img, -1.0, 1.0)
        hr_clamped = torch.clamp(hr_imgs, -1.0, 1.0)
        avg_psnr = psnr(sr_clamped, hr_clamped).mean().item()
        avg_ssim = ssim_fn(sr_clamped, hr_clamped).mean().item()
        print(f"{name:>12s} | PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")

    print("\n--- Saving Visual Comparison Grid ---")
    lr_upscaled = F.interpolate(lr_imgs, scale_factor=UPSCALE_FACTOR, mode='nearest')
    grid_tensors = []
    for i in range(BATCH_SIZE):
        grid_tensors.extend([lr_upscaled[i:i+1], fsrcnn_sr[i:i+1], esrgan_sr[i:i+1], sr3_sr[i:i+1], hr_imgs[i:i+1]])
    grid = torch.cat(grid_tensors, dim=0)
    grid = grid * 0.5 + 0.5
    grid_path = os.path.join(OUTPUT_DIR, 'grid2.png')
    save_image(grid, grid_path, nrow=5)
    print(f"Saved comparison grid to: {grid_path}")
    print("Grid columns (left to right): LR (Nearest), FSRCNN, ESRGAN, SR3 (Ours), HR (Ground Truth)")
    print("--- Running Inference on full test set ---")

    SR3_SKIP       = 10      # “do SR3 every 10th batch”  (≈ 1/10 images)

    # running sums
    tot_imgs  = 0
    sum_psnr  = {n: 0. for n in ["Bicubic", "FSRCNN", "ESRGAN", "SR3"]}
    sum_ssim  = {n: 0. for n in sum_psnr}
    sr3_imgs  = 0                 # how many images actually saw SR3

    for idx, (lr, hr) in enumerate(metric_loader):
        lr, hr = lr.to(DEVICE), hr.to(DEVICE)

        # --- baselines on *all* batches -------------------------------
        bicubic = F.interpolate(lr, scale_factor=UPSCALE_FACTOR,
                                mode="bicubic", align_corners=False)
        fsrcnn  = run_fsrcnn(fsrcnn_model, lr)
        esrgan  = run_esrgan(esrgan_model, hr)

        sr_dict = {
            "Bicubic": bicubic,
            "FSRCNN" : fsrcnn,
            "ESRGAN" : esrgan
        }

        # --- SR3 only every SR3_SKIP-th batch -------------------------
        if idx % SR3_SKIP == 0:
            with torch.no_grad():
                sr3 = p_sample_loop(sr3_model, hr.shape, lr)
            sr_dict["SR3"] = sr3
            sr3_imgs += hr.size(0)           # add these images to SR3 denom

        # --- accumulate metrics --------------------------------------
        tot_imgs += hr.size(0)

        for name, sr in sr_dict.items():
            ps  = psnr (torch.clamp(sr, -1, 1), torch.clamp(hr, -1, 1)).sum()
            ss  = ssim_fn(torch.clamp(sr, -1, 1), torch.clamp(hr, -1, 1)).sum()
            sum_psnr[name] += ps.item()
            sum_ssim[name] += ss.item()

    # ----------- final report -----------------------------------------
    print(f"\n=== Averages over {tot_imgs} images "
          f"(SR3 on {sr3_imgs}) ===")

    def avg(total, count): return total / count if count else float('nan')

    for name in ["Bicubic", "FSRCNN", "ESRGAN"]:
        print(f"{name:>8s} | PSNR {avg(sum_psnr[name], tot_imgs):6.2f} dB  "
              f"SSIM {avg(sum_ssim[name], tot_imgs):.4f}")

    print(f"{'SR3':>8s} | PSNR {avg(sum_psnr['SR3'], sr3_imgs):6.2f} dB  "
          f"SSIM {avg(sum_ssim['SR3'], sr3_imgs):.4f}  "
          f"(≈{100*sr3_imgs/tot_imgs:.1f}% sample)")

if __name__ == '__main__':
    main()