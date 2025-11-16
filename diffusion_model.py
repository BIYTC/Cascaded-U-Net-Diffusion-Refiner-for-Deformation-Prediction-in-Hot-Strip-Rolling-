import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_utils import ModelConfig


# Time embedding for diffusion steps
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = (dim + 1) // 2  # Ensure integer division
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1 + 1e-6)  # Avoid division by zero
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        self.register_buffer('emb', emb)
        self.scale = nn.Parameter(torch.ones(1) * 10.0)  # Learnable scaling

    def forward(self, t):
        t = t.float() * self.scale
        emb = t[:, None] * self.emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # Combine sine/cosine embeddings


# Adaptive Group Normalization
class AdaGN(nn.Module):
    def __init__(self, dim, time_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim * 2)
        )
        # Dynamically determine groups for GroupNorm
        if dim >= 8:
            self.num_groups = 8
            while dim % self.num_groups != 0:
                self.num_groups -= 1
        else:
            self.num_groups = 1
        self.norm = nn.GroupNorm(self.num_groups, dim)

    def forward(self, x, t_emb):
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=1)
        x = self.norm(x)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


# Residual Block with attention
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, config):
        super().__init__()
        self.config = config

        # Normalization (AdaGN or standard GroupNorm)
        if config.use_adaGN:
            self.norm1 = AdaGN(in_channels, time_dim)
            self.norm2 = AdaGN(out_channels, time_dim)
        else:
            self.norm1 = nn.GroupNorm(8, in_channels)
            self.norm2 = nn.GroupNorm(8, out_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.dropout = nn.Dropout2d(config.dropout)

        # Channel attention (for larger channels)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 8, 1),
            nn.SiLU(),
            nn.Conv2d(out_channels // 8, out_channels, 1),
            nn.Sigmoid()
        ) if out_channels >= 8 else None

        # Shortcut connection (for channel matching)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        h = self.norm1(x, t_emb) if self.config.use_adaGN else self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h, t_emb) if self.config.use_adaGN else self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        # Apply attention if enabled
        if self.attn is not None:
            attn = self.attn(h)
            h = h * attn

        return self.shortcut(x) + self.dropout(h)


# Downsampling Block
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, config):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualBlock(in_channels if i == 0 else out_channels,
                          out_channels, time_dim, config)
            for i in range(config.num_res_blocks)
        ])
        self.down = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)  # Strided conv for downsampling

        # Self-attention (at specified layers)
        self.attn = nn.MultiheadAttention(out_channels, num_heads=4) if len(self.blocks) in config.attn_layers else None

    def forward(self, x, t):
        for block in self.blocks:
            x = block(x, t)
        # Apply attention if enabled
        if self.attn is not None:
            B, C, H, W = x.shape
            x_attn = x.view(B, C, -1).permute(2, 0, 1)  # (H*W, B, C)
            x_attn = self.attn(x_attn, x_attn, x_attn)[0]
            x = x_attn.permute(1, 2, 0).view(B, C, H, W)  # (B, C, H, W)
        return self.down(x)


# Upsampling Block
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, config):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # Nearest neighbor upsampling
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.blocks = nn.ModuleList([
            ResidualBlock(out_channels * 2 if i == 0 else out_channels,
                          out_channels, time_dim, config)
            for i in range(config.num_res_blocks)
        ])

    def forward(self, x, skip, t):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)  # Concatenate with skip connection
        for block in self.blocks:
            x = block(x, t)
        return x


# Feature upsampling (from input features to spatial map)
class FeatureUpSample(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(21, 2048),  # Assume input features have 21 dimensions
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, 128 * 8 * 8),
            nn.Unflatten(1, (128, 8, 8))  # (B, 128, 8, 8)
        )

        # Decoder blocks for upsampling
        self.decoder_blocks = nn.ModuleList([
            ResidualBlock(128, 256, config.time_dim, config),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 16x16
            ResidualBlock(256, 256, config.time_dim, config),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 32x32
            ResidualBlock(256, 256, config.time_dim, config),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 64x64
            ResidualBlock(256, 128, config.time_dim, config),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 128x128
            ResidualBlock(128, 64, config.time_dim, config),
            nn.Conv2d(64, config.channel_mult[0], 3, padding=1)
        ])

    def forward(self, x, t):
        x = self.fc(x)
        for module in self.decoder_blocks:
            if isinstance(module, ResidualBlock):
                x = module(x, t)
            else:
                x = module(x)
        return x


# Xt upsampling (from (2,21,51) to spatial map)
class XtUpSampler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Encoder for initial processing
        self.encoder_blocks = nn.ModuleList([
            nn.Conv2d(2, 64, 3, padding=1),  # Input: (2,21,51)
            ResidualBlock(64, 128, config.time_dim, config),
            ResidualBlock(128, 256, config.time_dim, config),
        ])

        # Decoder for upsampling to 128x128
        self.decoder_blocks = nn.ModuleList([
            ResidualBlock(256, 256, config.time_dim, config),
            nn.Upsample(scale_factor=(2,1), mode='bilinear', align_corners=True),  # 42x51
            ResidualBlock(256, 256, config.time_dim, config),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 84x102
            ResidualBlock(256, 128, config.time_dim, config),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 168x204
            ResidualBlock(128, 64, config.time_dim, config),
            nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True),  # Force 128x128
            nn.Conv2d(64, config.channel_mult[0], 3, padding=1)
        ])

    def forward(self, x, t_emb):
        # Process encoder
        for module in self.encoder_blocks:
            if isinstance(module, ResidualBlock):
                x = module(x, t_emb)
            else:
                x = module(x)
        # Process decoder
        for module in self.decoder_blocks:
            if isinstance(module, ResidualBlock):
                x = module(x, t_emb)
            else:
                x = module(x)
        return x


# Main Diffusion Model (Denoising Network)
class DXYHead(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.config = config

        # Feature and Xt upsampling paths
        self.feature_upsample = FeatureUpSample(config)
        self.xt_upsample = XtUpSampler(config)

        # Fusion of upsampled features
        self.init_conv = nn.Conv2d(config.channel_mult[0] * 2, config.channel_mult[0], 3, padding=1)

        # Downsampling path with skip connections
        self.down_blocks = nn.ModuleList()
        self.skips = []  # To store skip connections
        channels = [config.channel_mult[i] for i in range(len(config.channel_mult))]
        for i in range(len(channels) - 1):
            self.down_blocks.append(
                DownBlock(channels[i], channels[i + 1], config.time_dim, config)
            )

        # Middle bottleneck
        self.mid_block = nn.Sequential(
            ResidualBlock(channels[-1], channels[-1], config.time_dim, config),
            ResidualBlock(channels[-1], channels[-1], config.time_dim, config)
        )

        # Upsampling path
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(channels) - 1)):
            self.up_blocks.append(
                UpBlock(channels[i + 1], channels[i], config.time_dim, config)
            )

        # Final projection to noise prediction (same shape as input: 2,21,51)
        self.final = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], 2, kernel_size=3, padding=1)  # Output: (2,21,51)
        )

    def forward(self, x, features, t):
        # Time embedding
        t_emb = TimeEmbedding(self.config.time_dim)(t)

        # Upsample features and xt
        feat_map = self.feature_upsample(features, t_emb)
        xt_map = self.xt_upsample(x, t_emb)

        # Fusion
        x = torch.cat([feat_map, xt_map], dim=1)
        x = self.init_conv(x)

        # Downsampling with skip connections
        skips = []
        for block in self.down_blocks:
            skips.append(x)
            x = block(x, t_emb)

        # Middle processing
        x = self.mid_block(x, t_emb)

        # Upsampling with skip connections
        for i, block in enumerate(self.up_blocks):
            x = block(x, skips[-(i + 1)], t_emb)

        # Final prediction
        return self.final(x)


# Wrapper for diffusion model
class DiffusionModel(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.model = DXYHead(config)

    def forward(self, x, features, t):
        return self.model(x, features, t)
