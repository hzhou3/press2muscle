import torch.nn as nn

class ChannelAttention1D(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)

        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (B, N, T) [PyTorch expects channels-first for 1D conv]
        avg_out = self.mlp(self.avg_pool(x).squeeze(-1))  # (B, N) → (B, N//r) → (B, N)
        max_out = self.mlp(self.max_pool(x).squeeze(-1))  # (B, N)
        channel_weights = self.sigmoid(avg_out + max_out).unsqueeze(-1)  # (B, N, 1)
        return x * channel_weights  # Scale features by channel importance


class MultiFoot_CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2, kernel_size=7, n_embd=512):
        super().__init__()
        self.channel_att = ChannelAttention1D(in_channels, reduction_ratio)
        self.emb = nn.Linear(in_channels, n_embd)

    def forward(self, x):
        # Input shape: (B, T, N) → permute to (B, N, T) for PyTorch 1D conv
        x = x.permute(0, 2, 1)  # (B, N, T)
        x = self.channel_att(x)
        # x = self.spatial_att(x)
        x = x.permute(0, 2, 1)  # Back to (B, T, N)
        return self.emb(x)

class RegionProcessor(nn.Module):
    def __init__(self, region_size, embed_dim):
        super(RegionProcessor, self).__init__()
        self.embed = nn.Linear(region_size, embed_dim)

    def forward(self, x):
        # x: (B, T, region_size)
        return self.embed(x)  # Output: (B, T, embed_dim)
