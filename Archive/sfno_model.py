"""
sfno_model.py — Spherical Fourier Neural Operator for SST forecasting.

A self-contained SFNO implementation using 2D real-FFT spectral
convolutions on a regular lat/lon grid.  All architecture hyper-
parameters are read from ``config.cfg.model_training``.

Architecture overview
─────────────────────
  Input (B, C_in, H, W)
    │
    ├─ Patch / channel lifting  → (B, embed_dim, H, W)
    │
    ├─ N × SpectralBlock
    │     ├─ rFFT2 → learnable complex weights (truncated modes) → irFFT2
    │     ├─ + pointwise MLP skip
    │     ├─ Norm + Activation + Dropout
    │     └─ Residual connection
    │
    ├─ Channel projection → (B, C_out, H, W)
    │
    └─ (optional) global residual skip from input
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


# ── Activation lookup ────────────────────────────────────────────────

ACTIVATIONS = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
}


# ── Helper: build normalisation layer ────────────────────────────────

def _build_norm(norm_type: str, num_features: int):
    if norm_type == "layer_norm":
        # Use GroupNorm(1, C) as a channel-wise LayerNorm for 2-D data
        return nn.GroupNorm(1, num_features)
    if norm_type == "instance_norm":
        return nn.InstanceNorm2d(num_features, affine=True)
    return nn.Identity()


# ── Spectral Convolution Layer ───────────────────────────────────────

class SpectralConv2d(nn.Module):
    """
    Learnable multiplication in the Fourier domain, truncated to
    (num_modes_lat, num_modes_lon) modes.

    Uses ``torch.fft.rfft2`` / ``irfft2`` which operate efficiently
    on real-valued inputs on a regular grid.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 num_modes_lat: int, num_modes_lon: int,
                 bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_modes_lat = num_modes_lat
        self.num_modes_lon = num_modes_lon

        # Two sets of weights for positive and negative latitude
        # frequencies (rfft2 only halves the last dimension).
        scale = 1.0 / (in_channels * out_channels)
        self.weight_upper = nn.Parameter(
            scale * torch.randn(in_channels, out_channels,
                                num_modes_lat, num_modes_lon, dtype=torch.cfloat)
        )
        self.weight_lower = nn.Parameter(
            scale * torch.randn(in_channels, out_channels,
                                num_modes_lat, num_modes_lon, dtype=torch.cfloat)
        )

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.cfloat))

    def _complex_mul(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Einstein summation for batched complex matrix-vector multiply."""
        # x: (B, C_in, M_lat, M_lon)  w: (C_in, C_out, M_lat, M_lon)
        return torch.einsum("bimn,iomn->bomn", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        m_lat = self.num_modes_lat
        m_lon = self.num_modes_lon

        # Forward FFT (real → complex, last dim halved)
        x_ft = torch.fft.rfft2(x, norm="ortho")

        _, _, Hf, Wf = x_ft.shape

        out_ft = torch.zeros(B, self.out_channels, Hf, Wf,
                             device=x.device, dtype=torch.cfloat)

        # Positive latitude frequencies (upper block)
        out_ft[:, :, :m_lat, :m_lon] = self._complex_mul(
            x_ft[:, :, :m_lat, :m_lon], self.weight_upper
        )
        # Negative latitude frequencies (lower block)
        out_ft[:, :, -m_lat:, :m_lon] = self._complex_mul(
            x_ft[:, :, -m_lat:, :m_lon], self.weight_lower
        )

        if self.bias is not None:
            out_ft = out_ft + self.bias[None, :, None, None]

        # Inverse FFT back to spatial domain
        out = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")
        return out


# ── Pointwise MLP (skip path in each block) ─────────────────────────

class PointwiseMLP(nn.Module):
    """1×1 conv → activation → 1×1 conv  (channel mixing)."""

    def __init__(self, channels: int, mlp_ratio: float, act_cls, dropout: float):
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            act_cls(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden, channels, 1),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ── Single SFNO Block ───────────────────────────────────────────────

class SFNOBlock(nn.Module):
    """
    One layer of the Spherical Fourier Neural Operator:
      spectral_conv + pointwise_mlp → norm → activation → residual
    """

    def __init__(self, channels: int, num_modes_lat: int, num_modes_lon: int,
                 mlp_ratio: float, act_cls, norm_type: str, dropout: float,
                 spectral_bias: bool):
        super().__init__()
        self.spectral_conv = SpectralConv2d(
            channels, channels, num_modes_lat, num_modes_lon, bias=spectral_bias
        )
        self.mlp = PointwiseMLP(channels, mlp_ratio, act_cls, dropout)
        self.norm = _build_norm(norm_type, channels)
        self.activation = act_cls()

    def forward(self, x):
        residual = x
        # Parallel spectral + pointwise paths
        x = self.spectral_conv(x) + self.mlp(x)
        x = self.norm(x)
        x = self.activation(x)
        x = x + residual
        return x


# ── Full SFNO Model ─────────────────────────────────────────────────

class SFNO(nn.Module):
    """
    Spherical Fourier Neural Operator for global SST forecasting.

    Takes a stack of ``C_in`` SST fields (+ optional lat/lon grids)
    and predicts ``C_out`` SST fields one time-step ahead.

    All parameters are read from ``cfg.model_training``.
    """

    def __init__(self):
        super().__init__()
        conf = cfg.model_training
        act_cls = ACTIVATIONS[conf.activation]

        c_in = conf.total_input_channels
        c_out = conf.output_channels
        d = conf.embed_dim

        # ── Encoder: lift input channels → embed_dim ─────────────
        self.encoder = nn.Sequential(
            nn.Conv2d(c_in, d, kernel_size=3, padding=1),
            _build_norm(conf.norm_type, d),
            act_cls(),
        )

        # ── Spectral blocks ──────────────────────────────────────
        self.blocks = nn.ModuleList([
            SFNOBlock(
                channels=d,
                num_modes_lat=conf.num_modes_lat,
                num_modes_lon=conf.num_modes_lon,
                mlp_ratio=conf.mlp_ratio,
                act_cls=act_cls,
                norm_type=conf.norm_type,
                dropout=conf.dropout_rate,
                spectral_bias=conf.spectral_bias,
            )
            for _ in range(conf.num_layers)
        ])

        # ── Decoder: project embed_dim → output channels ────────
        self.decoder = nn.Sequential(
            nn.Conv2d(d, d // 2, kernel_size=1),
            act_cls(),
            nn.Conv2d(d // 2, c_out, kernel_size=1),
        )

        # ── Optional global residual (input → output skip) ──────
        self.use_residual = conf.use_residual
        if self.use_residual:
            # Project the most recent input channel to output dims
            self.residual_proj = nn.Conv2d(1, c_out, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """Xavier initialisation for convolutional layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C_in, lat, lon)
            Input tensor — C_in channels of SST history (+ grid features).

        Returns
        -------
        pred : (B, C_out, lat, lon)
            Predicted SST field(s) for the next time step.
        """
        # Most recent SST channel for residual skip
        most_recent = x[:, 0:1, :, :]   # (B, 1, H, W)

        h = self.encoder(x)

        for block in self.blocks:
            h = block(h)

        out = self.decoder(h)

        if self.use_residual:
            out = out + self.residual_proj(most_recent)

        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Grid features helper ────────────────────────────────────────────

def make_grid_features(lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
    """
    Build normalised 2-D lat/lon grids as extra input channels.

    Parameters
    ----------
    lat : 1-D tensor of latitude values
    lon : 1-D tensor of longitude values

    Returns
    -------
    grid : (1, 2, len(lat), len(lon))
    """
    lat_norm = lat / 90.0          # → [-1, 1]
    lon_norm = lon / 180.0 - 1.0   # → [-1, 1]
    lat_2d, lon_2d = torch.meshgrid(lat_norm, lon_norm, indexing="ij")
    grid = torch.stack([lat_2d, lon_2d], dim=0).unsqueeze(0)  # (1, 2, H, W)
    return grid
