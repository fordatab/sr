# --- train.py --------------------------------------------------------------
from torch.utils.data import DataLoader
from data import SRFolderDataset
from utils import evaluate, save_ckpt
from sr3 import *
import argparse

if __name__ == "__main__":

    train_set = SRFolderDataset("lhq_split/train", training=True)
    # print("Found", len(train_set), "images")
    val_set   = SRFolderDataset("lhq_split/val", training=False)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True,
                              num_workers=10, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=8, shuffle=False,
                              num_workers=5, pin_memory=True)
    # TODO Set the parameters of the models 
    model = UNetSR3_Refined(
        in_channels_hr=IMG_CHANNELS,
        in_channels_lr=IMG_CHANNELS,
        out_channels=IMG_CHANNELS,
        init_dim=24,
        dim_mults=(1, 2, 3),
        num_resnet_blocks_per_level=1,
        time_emb_dim_input=32,
        time_emb_dim_mlp=128,
        use_attention_at_resolutions=(32,),  # e.g. (16,) or (32, 16) if model is larger
        resnet_groups=8,
        attention_heads=4,
        attention_head_dim=24,
        dropout_rate=0.0,
        hr_size=HR_IMG_SIZE
    ).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(train_loader)*100)

    ema_model = UNetSR3_Refined(
        in_channels_hr=IMG_CHANNELS,
        in_channels_lr=IMG_CHANNELS,
        out_channels=IMG_CHANNELS,
        init_dim=24,
        dim_mults=(1, 2, 3),
        num_resnet_blocks_per_level=1,
        time_emb_dim_input=32,
        time_emb_dim_mlp=128,
        use_attention_at_resolutions=(32,),  # e.g. (16,) or (32, 16) if model is larger
        resnet_groups=8,
        attention_heads=4,
        attention_head_dim=24,
        dropout_rate=0.0,
        hr_size=HR_IMG_SIZE
    ).to(DEVICE)
    ema_model.load_state_dict(model.state_dict())
    ema_decay = 0.9999
    global_step = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="", help="path to .pth to resume from")
    args = parser.parse_args()
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["model"])           # EMA = last model weights
        opt.load_state_dict(ckpt["opt"])
        global_step = ckpt["step"]
        # put scheduler in the right place
        sched.last_epoch = global_step

    print("Resuming at global step", global_step) 

    for epoch in range(100):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for lr_imgs, hr_imgs in pbar:
            lr_imgs, hr_imgs = lr_imgs.to(DEVICE), hr_imgs.to(DEVICE)
            t = torch.randint(0, TIMESTEPS, (lr_imgs.size(0),), device=DEVICE).long()
            loss = p_losses(model, hr_imgs, lr_imgs, t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

            # EMA update
            with torch.no_grad():
                for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                    ema_p.data.mul_(ema_decay).add_(p.data, alpha=1-ema_decay)

            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             lr=sched.get_last_lr()[0])

            # ------ validation & checkpoint ------
            if global_step % 1500 == 0:
                psnr_val, ssim_val = evaluate(ema_model, val_loader, global_step)
                print(f"[step {global_step}]  PSNR: {psnr_val:.2f}  SSIM: {ssim_val:.3f}")
                save_ckpt(ema_model, opt, global_step, f"ckpts/step_{global_step:07d}.pth")
