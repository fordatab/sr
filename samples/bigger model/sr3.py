import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm

# --- Configuration ---
IMG_CHANNELS = 3
LR_IMG_SIZE = 128
HR_IMG_SIZE = 256
UPSCALE_FACTOR = HR_IMG_SIZE // LR_IMG_SIZE
TIMESTEPS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- Helper Modules (SinusoidalPositionEmbeddings, ResidualBlock, AttentionBlock, Downsample, Upsample) ---
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=DEVICE) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None, groups=8, dropout=0.0):
        super().__init__()
        self.time_mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
            if time_emb_dim is not None
            else None
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.act1 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = nn.SiLU()
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb=None):
        h = self.act1(self.norm1(self.conv1(x)))
        if self.time_mlp is not None and t_emb is not None:
            time_cond = self.time_mlp(t_emb)
            h = h + time_cond[:, :, None, None]
        h = self.dropout(h)
        h = self.act2(self.norm2(self.conv2(h)))
        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4, head_dim=None, groups=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else channels // num_heads
        if self.head_dim * num_heads != channels and head_dim is None:
            # This warning can be noisy if channels are prime, etc.
            # print(f"Warning: channels {channels} not perfectly divisible by num_heads {num_heads} for head_dim calculation. Using head_dim={self.head_dim}")
            pass
        self.scale = self.head_dim ** -0.5
        self.norm = nn.GroupNorm(groups, channels)
        self.to_qkv = nn.Conv2d(channels, self.num_heads * self.head_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(self.num_heads * self.head_dim, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        qkv = self.to_qkv(x_norm).chunk(3, dim=1)
        q, k, v = map(
            lambda t: t.reshape(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2), qkv
        )
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.permute(0, 1, 3, 2).reshape(b, self.num_heads * self.head_dim, h, w)
        return self.to_out(out) + x


class Downsample(nn.Module):
    def __init__(self, channels_in, channels_out=None):
        super().__init__()
        channels_out = channels_out if channels_out is not None else channels_in
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels_in, channels_out=None):
        super().__init__()
        channels_out = channels_out if channels_out is not None else channels_in
        self.conv = nn.ConvTranspose2d(channels_in, channels_out, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


# --- Refined U-Net Model ---
class UNetSR3_Refined(nn.Module):
    def __init__(
            self,
            in_channels_hr=IMG_CHANNELS,
            in_channels_lr=IMG_CHANNELS,
            out_channels=IMG_CHANNELS,
            init_dim=64,
            dim_mults=(1, 2, 4, 8),
            num_resnet_blocks_per_level=2,
            time_emb_dim_input=64,
            time_emb_dim_mlp=256,
            use_attention_at_resolutions=(16, 8),
            resnet_groups=8,
            attention_heads=4,
            attention_head_dim=None,
            dropout_rate=0.1,
            hr_size=HR_IMG_SIZE
    ):
        super().__init__()
        self.hr_size = hr_size

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim_input),
            nn.Linear(time_emb_dim_input, time_emb_dim_mlp),
            nn.SiLU(),
            nn.Linear(time_emb_dim_mlp, time_emb_dim_mlp),
        )

        self.init_conv = nn.Conv2d(in_channels_hr + in_channels_lr, init_dim, kernel_size=7, padding=3)

        dims = [init_dim] + [init_dim * m for m in dim_mults]

        self.downs = nn.ModuleList([])
        current_res = self.hr_size
        for i in range(len(dim_mults)):
            dim_in = dims[i]
            dim_out = dims[i + 1]

            level_blocks = nn.ModuleList([])
            for _ in range(num_resnet_blocks_per_level):
                level_blocks.append(ResidualBlock(dim_in, dim_in, time_emb_dim=time_emb_dim_mlp, groups=resnet_groups,
                                                  dropout=dropout_rate))
                if current_res in use_attention_at_resolutions:
                    level_blocks.append(AttentionBlock(dim_in, num_heads=attention_heads, head_dim=attention_head_dim,
                                                       groups=resnet_groups))

            if i != len(dim_mults) - 1:
                level_blocks.append(Downsample(dim_in, dim_out))
                current_res //= 2
            else:  # Bottleneck conv (maintains current_res, dim_out is bottleneck_dim)
                level_blocks.append(nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1))
            self.downs.append(level_blocks)

        mid_dim = dims[-1]
        self.mid_block1 = ResidualBlock(mid_dim, mid_dim, time_emb_dim=time_emb_dim_mlp, groups=resnet_groups,
                                        dropout=dropout_rate)
        # current_res here is the bottleneck resolution
        if current_res in use_attention_at_resolutions:
            self.mid_attn = AttentionBlock(mid_dim, num_heads=attention_heads, head_dim=attention_head_dim,
                                           groups=resnet_groups)
        else:
            self.mid_attn = nn.Identity()
        self.mid_block2 = ResidualBlock(mid_dim, mid_dim, time_emb_dim=time_emb_dim_mlp, groups=resnet_groups,
                                        dropout=dropout_rate)

        self.ups = nn.ModuleList([])
        for i in reversed(range(len(dim_mults))):
            # Determine resolution for attention check: this is the resolution *after* upsampling in the previous level,
            # or bottleneck res if it's the first upsampling level. It should match the skip connection's resolution.
            level_res = self.hr_size // (2 ** i)

            dim_skip_channels = dims[i]
            dim_x_channels_from_lower_level = dims[i + 1]
            dim_out_channels_for_this_level = dims[i]

            level_blocks = nn.ModuleList([])
            # First block after concat: input channels are x_channels + skip_channels
            level_blocks.append(
                ResidualBlock(dim_x_channels_from_lower_level + dim_skip_channels, dim_out_channels_for_this_level,
                              time_emb_dim=time_emb_dim_mlp, groups=resnet_groups, dropout=dropout_rate))
            if level_res in use_attention_at_resolutions:
                level_blocks.append(AttentionBlock(dim_out_channels_for_this_level, num_heads=attention_heads,
                                                   head_dim=attention_head_dim, groups=resnet_groups))

            # Subsequent blocks in the same level operate on dim_out_channels_for_this_level
            for _ in range(num_resnet_blocks_per_level - 1):  # -1 because one block is already added
                level_blocks.append(ResidualBlock(dim_out_channels_for_this_level, dim_out_channels_for_this_level,
                                                  time_emb_dim=time_emb_dim_mlp, groups=resnet_groups,
                                                  dropout=dropout_rate))
                if level_res in use_attention_at_resolutions:  # Attention uses output channels of the ResBlock
                    level_blocks.append(AttentionBlock(dim_out_channels_for_this_level, num_heads=attention_heads,
                                                       head_dim=attention_head_dim, groups=resnet_groups))

            if i != 0:  # If not the last upsampling level (closest to output)
                level_blocks.append(Upsample(dim_out_channels_for_this_level, dim_out_channels_for_this_level))
            else:  # Last upsampling level, followed by a Conv2d instead of Upsample
                level_blocks.append(
                    nn.Conv2d(dim_out_channels_for_this_level, dim_out_channels_for_this_level, kernel_size=3,
                              padding=1))
            self.ups.append(level_blocks)

        # Renamed final layers for clarity and to avoid potential old name conflicts
        self.final_resblock = ResidualBlock(init_dim * 2, init_dim, time_emb_dim=time_emb_dim_mlp,
                                            groups=resnet_groups)  # Takes init_dim from upsample + init_dim from initial_skip
        self.final_projection = nn.Conv2d(init_dim, out_channels, kernel_size=1)

    def forward(self, x_noisy_hr, time, x_lr):
        x_cond = F.interpolate(x_lr, scale_factor=UPSCALE_FACTOR, mode='bicubic', align_corners=False)
        x = torch.cat((x_noisy_hr, x_cond), dim=1)
        t_emb = self.time_mlp(time)

        h = self.init_conv(x)
        initial_skip = h.clone()  # initial_skip has 'init_dim' channels, at HR_SIZE

        # --- Downsampling Path & Skip Collection ---
        skips = []
        for i, level_module_list in enumerate(self.downs):
            for layer_idx, layer in enumerate(level_module_list):
                is_last_op_in_level = (layer_idx == len(level_module_list) - 1)

                if isinstance(layer, ResidualBlock):
                    h = layer(h, t_emb)
                elif isinstance(layer, AttentionBlock):
                    h = layer(h)
                # If it's the Downsample or bottleneck Conv2d layer (last op in this level_module_list)
                elif isinstance(layer, (Downsample, nn.Conv2d)):
                    skips.append(h)  # Store h *before* it's downsampled or convolved for the next level
                    h = layer(h)  # Apply Downsample or Conv2d
                else:  # Should ideally not be reached if structure is as defined
                    h = layer(h)
                    # If the last op was not Downsample/Conv2d (e.g. if a level ends with ResBlock/Attn before bottleneck)
            # This case should not happen with the current __init__ structure where Downsample/Conv2d is always last in level_blocks
            # However, to be robust for the very last down level (bottleneck conv):
            if not isinstance(level_module_list[-1], (
            Downsample)):  # If last element isn't a downsampler, it means h is already at bottleneck input
                # and skip for this level was already added (or this h *is* the skip for bottleneck if bottleneck conv is next)
                # The current logic with skips.append(h) *before* Downsample/Conv2d handles this.
                pass

        # --- Bottleneck ---
        # h is now the input to the bottleneck (output of the last nn.Conv2d in self.downs)
        x_bottleneck = h
        x_bottleneck = self.mid_block1(x_bottleneck, t_emb)
        x_bottleneck = self.mid_attn(x_bottleneck)
        x_bottleneck = self.mid_block2(x_bottleneck, t_emb)

        # --- Upsampling Path ---
        x = x_bottleneck
        for level_module_list in self.ups:
            s = skips.pop()
            x = torch.cat((x, s), dim=1)

            for layer in level_module_list:
                if isinstance(layer, ResidualBlock):
                    x = layer(x, t_emb)
                elif isinstance(layer, AttentionBlock):
                    x = layer(x)
                else:  # Upsample or Conv2d
                    x = layer(x)

        # After all upsampling, x should have 'init_dim' channels and be at HR_SIZE
        x = torch.cat((x, initial_skip), dim=1)  # x becomes init_dim + init_dim channels

        # Use the renamed final layers
        x = self.final_resblock(x, t_emb)  # Output init_dim channels
        x = self.final_projection(x)  # Output out_channels (IMG_CHANNELS)
        return x


# --- Diffusion Utilities (DDPM based) --- (These remain the same)
def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps, device=DEVICE)


betas = linear_beta_schedule(TIMESTEPS)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))


def q_sample(x_start, t, noise=None):
    if noise is None: noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def p_losses(denoise_model, x_start_hr, x_lr, t, noise=None):
    if noise is None: noise = torch.randn_like(x_start_hr)
    x_noisy_hr = q_sample(x_start=x_start_hr, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy_hr, t, x_lr)
    loss = F.l1_loss(noise, predicted_noise)  # SR3 uses L1
    return loss


@torch.no_grad()
def p_sample(model, x_t, t_idx, x_lr):
    betas_t = betas[t_idx]
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t_idx]
    sqrt_recip_alphas_t = torch.sqrt(1.0 / alphas[t_idx])
    t_tensor = torch.full((x_t.shape[0],), t_idx, device=DEVICE, dtype=torch.long)
    predicted_noise = model(x_t, t_tensor, x_lr)
    model_mean = sqrt_recip_alphas_t * (x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
    if t_idx == 0:
        return model_mean
    else:
        posterior_variance_t = posterior_variance[t_idx]
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def p_sample_loop(model, shape_hr, x_lr):
    img = torch.randn(shape_hr, device=DEVICE)
    for i in tqdm(reversed(range(0, TIMESTEPS)), desc='Sampling loop', total=TIMESTEPS):
        img = p_sample(model, img, i, x_lr)
    return img


# --- Main Demo ---
# if __name__ == "__main__":
#     model = UNetSR3_Refined(
#         in_channels_hr=IMG_CHANNELS,
#         in_channels_lr=IMG_CHANNELS,
#         out_channels=IMG_CHANNELS,
#         init_dim=32,
#         dim_mults=(1, 2, 2),
#         num_resnet_blocks_per_level=1,
#         time_emb_dim_input=32,
#         time_emb_dim_mlp=128,
#         use_attention_at_resolutions=(),  # e.g. (16,) or (32, 16) if model is larger
#         resnet_groups=8,
#         attention_heads=2,
#         attention_head_dim=16,
#         dropout_rate=0.0,
#         hr_size=HR_IMG_SIZE
#     ).to(DEVICE)
#
#     print(f"Model parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
#
#     print("\n--- U-Net Direct Call Test ---")
#     model.eval()
#     dummy_noisy_hr = torch.randn(1, IMG_CHANNELS, HR_IMG_SIZE, HR_IMG_SIZE, device=DEVICE)
#     dummy_lr_cond = torch.randn(1, IMG_CHANNELS, LR_IMG_SIZE, LR_IMG_SIZE, device=DEVICE)
#     dummy_time = torch.randint(0, TIMESTEPS, (1,), device=DEVICE).long()
#     try:
#         with torch.no_grad():
#             pred_noise = model(dummy_noisy_hr, dummy_time, dummy_lr_cond)
#         print(f"U-Net output shape: {pred_noise.shape} (expected: {dummy_noisy_hr.shape})")
#         if pred_noise.shape == dummy_noisy_hr.shape:
#             print("U-Net call successful and output shape matches.")
#         else:
#             print(f"U-Net call shape MISMATCH: Expected {dummy_noisy_hr.shape}, Got {pred_noise.shape}")
#     except Exception as e:
#         print(f"Error during U-Net direct call: {e}")
#         import traceback
#
#         traceback.print_exc()
#     print("----------------------------\n")
#
#     print("\n--- Toy Training ---")
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#     epochs = 3
#     batch_size = 2
#     steps_per_epoch = 200
#
#     for epoch in range(epochs):
#         model.train()
#         for step in range(steps_per_epoch):
#             optimizer.zero_grad()
#             dummy_hr_images = torch.randn(batch_size, IMG_CHANNELS, HR_IMG_SIZE, HR_IMG_SIZE, device=DEVICE)
#             dummy_lr_images = F.interpolate(dummy_hr_images, size=LR_IMG_SIZE, mode='area')
#             t = torch.randint(0, TIMESTEPS, (batch_size,), device=DEVICE).long()
#             loss = p_losses(model, dummy_hr_images, dummy_lr_images, t)
#             loss.backward()
#             optimizer.step()
#             if (step + 1) % 10 == 0:
#                 print(f"Epoch {epoch + 1}/{epochs}, Step {step + 1}/{steps_per_epoch}, Loss: {loss.item():.4f}")
#         print(f"Epoch {epoch + 1} finished.")
#
#     print("\n--- Toy Sampling ---")
#     model.eval()
#     test_lr_image = torch.rand(1, IMG_CHANNELS, LR_IMG_SIZE, LR_IMG_SIZE, device=DEVICE) * 2 - 1
#     target_hr_shape = (1, IMG_CHANNELS, HR_IMG_SIZE, HR_IMG_SIZE)
#     with torch.no_grad():
#         sampled_hr_image = p_sample_loop(model, target_hr_shape, test_lr_image)
#     print(f"Sampled HR image shape: {sampled_hr_image.shape}")
#     print("\nDemo finished.")
