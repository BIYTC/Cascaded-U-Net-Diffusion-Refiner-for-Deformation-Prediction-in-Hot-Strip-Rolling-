import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_utils import ModelConfig


# Enhanced time embedding
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = (dim + 1) // 2  # Fixed dimension calculation
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1 + 1e-6)  # Added: prevent division by zero
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        self.register_buffer('emb', emb)

        # Added: learnable scaling factor
        self.scale = nn.Parameter(torch.ones(1) * 10.0)

    def forward(self, t):
        t = t.float() * self.scale  # Enhanced time embedding expression
        emb = t[:, None] * self.emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


# Adaptive normalization layer
class AdaGN(nn.Module):
    def __init__(self, dim, time_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim * 2)
        )

        # Dynamically calculate number of groups
        if dim >= 8:
            # Find maximum divisible group count (<=8)
            self.num_groups = 8
            while dim % self.num_groups != 0:
                self.num_groups -= 1
        else:
            self.num_groups = 1

        self.norm = nn.GroupNorm(self.num_groups, dim)

    def forward(self, x, t_emb):
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=1)  # Use embedded time parameters
        x = self.norm(x)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


# Enhanced residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, config):
        super().__init__()
        self.config = config

        # Adaptive normalization selection
        if config.use_adaGN:
            self.norm1 = AdaGN(in_channels, time_dim)
            self.norm2 = AdaGN(out_channels, time_dim)
        else:
            self.norm1 = nn.GroupNorm(8, in_channels)
            self.norm2 = nn.GroupNorm(8, out_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_embed = nn.Linear(time_dim, out_channels * 2)

        # Added: channel attention
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 8, 1),
            nn.SiLU(),
            nn.Conv2d(out_channels // 8, out_channels, 1),
            nn.Sigmoid()
        ) if out_channels >= 8 else None

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.dropout = nn.Dropout2d(config.dropout)

        # Added: adaptive shortcut connection
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.norm1(x, t_emb) if self.config.use_adaGN else self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h, t_emb) if self.config.use_adaGN else self.norm2(h)
        h = self.conv2(F.silu(h))

        # Added: channel attention
        if self.attn is not None:
            attn = self.attn(h)
            h = h * attn

        return self.shortcut(x) + self.dropout(h)


# Improved downsampling block
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, config):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualBlock(in_channels if i == 0 else out_channels,
                          out_channels, time_dim, config)
            for i in range(config.num_res_blocks)
        ])
        self.down = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)  # Replaced original downsampling

        # Added: attention layer
        if len(self.blocks) in config.attn_layers:
            self.attn = nn.MultiheadAttention(out_channels, num_heads=4)
        else:
            self.attn = None

    def forward(self, x, t):
        for block in self.blocks:
            x = block(x, t)
        if self.attn is not None:
            B, C, H, W = x.shape
            x_attn = x.view(B, C, -1).permute(2, 0, 1)
            x_attn = self.attn(x_attn, x_attn, x_attn)[0]
            x = x_attn.permute(1, 2, 0).view(B, C, H, W)
        return self.down(x)


# Improved upsampling block
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, config):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # Replaced transposed convolution
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.blocks = nn.ModuleList([
            ResidualBlock(out_channels * 2 if i == 0 else out_channels,
                          out_channels, time_dim, config)
            for i in range(config.num_res_blocks)
        ])

    def forward(self, x, skip, t):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        for block in self.blocks:
            x = block(x, t)
        return x


# Enhanced feature upsampling (key modification)
class FeatureUpSample(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(21, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, 128 * 8 * 8),
            nn.Unflatten(1, (128, 8, 8))
        )

        # Decompose Sequential into manageable modules
        self.decoder_blocks = nn.ModuleList([
            ResidualBlock(128, 256, config.time_dim, config),
            nn.Upsample(scale_factor=2, mode='bilinear'),        # 16*16
            ResidualBlock(256, 256, config.time_dim, config),
            nn.Upsample(scale_factor=2, mode='bilinear'),        # 32*32
            ResidualBlock(256, 256, config.time_dim, config),
            nn.Upsample(scale_factor=2, mode='bilinear'),        # 64*64
            ResidualBlock(256, 128, config.time_dim, config),
            nn.Upsample(scale_factor=2, mode='bilinear'),        # 128*128
            ResidualBlock(128, 64, config.time_dim, config),
            nn.Conv2d(64, config.channel_mult[0], 3, padding=1)
        ])

    def forward(self, x, t):  # Added time parameter
        x = self.fc(x)
        for module in self.decoder_blocks:
            if isinstance(module, ResidualBlock):
                x = module(x, t)
            else:
                x = module(x)
        return x


# Improved Xt upsampling (key modification)
class XtUpSampler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Decompose encoder into separate modules
        self.encoder_blocks = nn.ModuleList([
            nn.Conv2d(1, 64, 3, padding=1),
            ResidualBlock(64, 128, config.time_dim, config),
            ResidualBlock(128, 256, config.time_dim, config),
        ])

        # Decompose decoder into separate modules
        self.decoder_blocks = nn.ModuleList([
            ResidualBlock(256, 256, config.time_dim, config),   # 21*51
            nn.Upsample(scale_factor=(2,1), mode='bilinear'),   # 42*51
            ResidualBlock(256, 256, config.time_dim, config),
            nn.Upsample(scale_factor=2, mode='bilinear'),       # 84*102
            ResidualBlock(256, 128, config.time_dim, config),
            nn.Upsample(scale_factor=2, mode='bilinear'),       # 168*204
            ResidualBlock(128, 64, config.time_dim, config),
            nn.Upsample(size=(128, 128), mode='bilinear'),        # Force size alignment
            nn.Conv2d(64, config.channel_mult[0], 3, padding=1)
        ])

    def forward(self, x, t_emb):  # Added time parameter
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


# Reconstructed DXYHead (key modification)
class DXYHead(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.config = config

        # Feature path
        self.feature_upsample = FeatureUpSample(config)
        self.xt_upsample = XtUpSampler(config)

        # Initial fusion
        self.init_conv = nn.Conv2d(config.channel_mult[0] * 2, config.channel_mult[0], 3, padding=1)

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        channels = [config.channel_mult[i] for i in range(len(config.channel_mult))]
        for i in range(len(channels) - 1):
            self.down_blocks.append(
                DownBlock(channels[i], channels[i + 1], config.time_dim, config)
            )

        # Middle layer
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

        # Final projection
