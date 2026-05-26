import math
import torch
import torch.nn as nn


class EMGConvBlock(nn.Module):
    """
    Multi-scale temporal convolutional block for EMG preprocessing.

    This block applies three parallel depthwise 1D convolutions with different
    kernel sizes to capture temporal patterns at multiple time scales.
    The filtered signals are averaged, rectified (absolute value),
    and smoothed with a low-pass convolution to compute the EMG envelope.

    Input:
        x : torch.Tensor of shape (B, C, T)
            B - batch size (number of trials)
            C - number of EMG channels
            T - number of time samples

    Output:
        feat : torch.Tensor of shape (B, C, T)
            Smoothed EMG envelope after multi-scale temporal filtering.

    Processing steps:
        1. Depthwise temporal convolutions (per-channel, no cross-channel mixing)
           with kernel sizes k1, k2, k3.
        2. Averaging of the three filtered signals (multi-scale fusion).
        3. Rectification via absolute value.
        4. Low-pass filtering to obtain the amplitude envelope.

    Notes:
        - Convolutions use padding = kernel_size // 2 to preserve time length.
        - groups=C ensures independent filtering of each channel.
        - Designed for EMG feature extraction prior to classification.
    """

    def __init__(
        self,
        n_channels: int,
        k1: int = 21,
        k2: int = 51,
        k3: int = 71,
        env_size: int = 101,
    ):

        super().__init__()

        assert k1 % 2 == 1 and k2 % 2 == 1 and k3 % 2 == 1

        C = n_channels

        self.conv1 = nn.Conv1d(
            C, C, kernel_size=k1, padding=k1 // 2, stride=1, groups=C, bias=False
        )
        self.conv2 = nn.Conv1d(
            C, C, kernel_size=k2, padding=k2 // 2, stride=1, groups=C, bias=False
        )
        self.conv3 = nn.Conv1d(
            C, C, kernel_size=k3, padding=k3 // 2, stride=1, groups=C, bias=False
        )
        self.env_lp = nn.Conv1d(
            C, C, kernel_size=env_size, padding=env_size // 2, groups=C, bias=False
        )

        # опционально: аккуратная инициализация (чтобы не взрывалось)
        for m in (self.conv1, self.conv2, self.conv3, self.env_lp):
            nn.init.normal_(m.weight, mean=0.0, std=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y3 = self.conv3(x)

        y = (y1 + y2 + y3) / 3  # (B, C, T)

        env = torch.abs(y)
        feat = self.env_lp(env)

        return feat  # (B, C, T)


class ChannelCNN(nn.Module):
    """Pointwise mixing (C->D) + depthwise temporal conv (optional downsampling)."""

    def __init__(
        self,
        in_ch: int,
        d_model: int,
        k: int = 7,
        stride: int = 2,
        dropout: float = 0.0,
    ):

        super().__init__()

        assert k % 2 == 1

        self.pw = nn.Conv1d(in_ch, d_model, kernel_size=1, bias=False)  # mix channels
        self.dw = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=k,
            padding=k // 2,  # сохраняем края
            stride=stride,  # stride=2 -> T примерно пополам
            groups=d_model,
            bias=False,
        )
        self.norm = nn.GroupNorm(1, d_model)
        self.act = nn.ELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        z = self.pw(x)  # (B, D, T)
        z = self.dw(z)  # (B, D, T') where T' ~ T/stride
        z = self.drop(self.act(self.norm(z)))
        return z


class EMGAugment(nn.Module):
    """EMG augmentation for x: (B,C,T). Applied only during training."""

    def __init__(
        self,
        # noise relative to signal
        p_noise=0.5,
        noise_rms_frac=0.03,   # noise std = frac * per-sample RMS  (0.03..0.10)

        # gain (log-normal)
        p_gain=0.5,
        gain_log_std=0.08,     # multiplicative factor ~ exp(N(0, log_std)) (0.05..0.20)

        # time shift
        p_shift=0.5,
        max_shift=80,

        # channel dropout
        p_ch_drop=0.2,
        ch_drop_rate=0.125,

        # time mask
        p_tmask=0.2,
        tmask=40,

        # time warp (simple)
        p_twarp=0.0,
        twarp_strength=0.15,   # 0.10..0.25

        # mixup
        p_mixup=0.0,
        mixup_alpha=0.4,
    ):
        super().__init__()

        self.p_noise = p_noise
        self.noise_rms_frac = noise_rms_frac

        self.p_gain = p_gain
        self.gain_log_std = gain_log_std

        self.p_shift = p_shift
        self.max_shift = int(max_shift)

        self.p_ch_drop = p_ch_drop
        self.ch_drop_rate = ch_drop_rate

        self.p_tmask = p_tmask
        self.tmask = int(tmask)

        self.p_twarp = p_twarp
        self.twarp_strength = twarp_strength

        self.p_mixup = p_mixup
        self.mixup_alpha = mixup_alpha

    def _rms(self, x, eps=1e-6):
        # x: (B,C,T) -> per-sample per-channel RMS (B,C,1)
        return torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)

    def _time_warp(self, x):
        # lightweight warp via random speed-up/slow-down around 1.0
        # x: (B,C,T)
        B, C, T = x.shape
        # scale in [1-s, 1+s]
        s = self.twarp_strength
        scale = (1.0 + (2 * torch.rand(B, 1, 1, device=x.device) - 1.0) * s).clamp(0.7, 1.3)

        # create warped grid for interpolation
        t = torch.linspace(0, 1, T, device=x.device).view(1, 1, T)  # (1,1,T)
        t_w = (t / scale).clamp(0, 1)  # (B,1,T) via broadcast
        t_w = t_w.expand(B, 1, T)

        # convert to indices for linear interpolation
        idx = t_w * (T - 1)
        idx0 = torch.floor(idx).long().clamp(0, T - 1)
        idx1 = (idx0 + 1).clamp(0, T - 1)
        w = (idx - idx0.float())

        # gather
        x0 = x.gather(dim=2, index=idx0.expand(B, C, T))
        x1 = x.gather(dim=2, index=idx1.expand(B, C, T))
        return x0 * (1 - w) + x1 * w

    def _mixup(self, x, y):
        # x: (B,C,T), y: (B,)
        if x.size(0) < 2:
            return x, y, None

        lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().to(x.device)
        perm = torch.randperm(x.size(0), device=x.device)
        x2, y2 = x[perm], y[perm]
        x_mix = lam * x + (1 - lam) * x2
        return x_mix, y, (y2, float(lam))

    def forward(self, x, y=None):
        # x: (B,C,T); y optional for mixup
        if not self.training:
            return (x, y, None) if y is not None else x

        B, C, T = x.shape
        out = x

        # noise scaled by RMS
        if self.p_noise and torch.rand(()) < self.p_noise:
            rms = self._rms(out)  # (B,C,1)
            noise = torch.randn_like(out) * (self.noise_rms_frac * rms)
            out = out + noise

        # log-normal gain
        if self.p_gain and torch.rand(()) < self.p_gain:
            g = torch.exp(self.gain_log_std * torch.randn(B, C, 1, device=out.device, dtype=out.dtype))
            out = out * g

        # time shift
        if self.p_shift and self.max_shift > 0 and torch.rand(()) < self.p_shift:
            shift = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item())
            out = torch.roll(out, shifts=shift, dims=-1)

        # channel dropout
        if self.p_ch_drop and torch.rand(()) < self.p_ch_drop:
            drop = torch.rand(B, C, 1, device=out.device) < self.ch_drop_rate
            out = out.masked_fill(drop, 0.0)

        # time mask
        if self.p_tmask and 0 < self.tmask < T and torch.rand(()) < self.p_tmask:
            t0 = int(torch.randint(0, T - self.tmask, (1,)).item())
            out = out.clone()
            out[..., t0 : t0 + self.tmask] = 0.0

        # time warp
        if self.p_twarp and torch.rand(()) < self.p_twarp:
            out = self._time_warp(out)

        # mixup (return extra info so loss can be computed properly)
        mix_info = None
        if y is not None and self.p_mixup and torch.rand(()) < self.p_mixup:
            out, y, mix_info = self._mixup(out, y)

        return (out, y, mix_info) if y is not None else out


class SinusoidalPosEnc(nn.Module):
    """Sinusoidal positional encoding for x: (B, T, D). Adds pe and applies dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):

        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        pe = torch.zeros(max_len, d_model)  # (T, D)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)  # (T, 1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        pe = pe.unsqueeze(0)  # (1, T, D) for broadcasting
        self.register_buffer("pe", pe)
        self.pe: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        pe = self.pe[:, :T, :]
        x = x + pe
        return self.dropout(x)


class CNNTransformerClassifier(nn.Module):
    def __init__(
        self,
        in_ch,
        n_classes,
        d_model=64,
        n_heads=4,
        ff_dim=256,
        num_layers=2,
        dropout=0.2,
        k_cnn=21,
        stride_cnn=8,
        k1=21,
        k2=51,
        k3=71,
        env_size=101,
        variable_length: bool = True,
    ):

        super().__init__()
        _ = variable_length  # kept for backwards-compatible signature
        # We use the same safe augmentation policy everywhere:
        # disable temporal shift/warp to avoid mixing valid and padded regions.
        self.augment = EMGAugment(p_shift=0.0, max_shift=0, p_twarp=0.0)
        self.pre = EMGConvBlock(in_ch, k1=k1, k2=k2, k3=k3, env_size=env_size)
        self.cnn = ChannelCNN(
            in_ch, d_model, k=k_cnn, stride=stride_cnn, dropout=dropout
        )
        self.pos = SinusoidalPosEnc(d_model, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        # Disable nested tensor optimization to avoid PyTorch warning when norm_first=True.
        self.tr = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d_model)
        self.cls = nn.Linear(d_model, n_classes)

    def _downsample_padding_mask(
        self, padding_mask: torch.Tensor | None, out_len: int
    ) -> torch.Tensor | None:
        """
        Convert input-time padding mask to CNN output-time mask using Conv1d shape rule.
        padding_mask: (B, Tin), True for padded positions.
        """
        if padding_mask is None:
            return None

        if padding_mask.ndim != 2:
            raise ValueError(f"padding_mask must be 2D (B,T), got {padding_mask.shape}")

        lengths = (~padding_mask).sum(dim=1).to(dtype=torch.long)  # (B,)

        stride = int(self.cnn.dw.stride[0])
        pad = int(self.cnn.dw.padding[0])
        kernel = int(self.cnn.dw.kernel_size[0])
        dilation = int(self.cnn.dw.dilation[0])

        out_lengths = (
            (lengths + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1
        ).clamp(min=1, max=out_len)

        t = torch.arange(out_len, device=padding_mask.device).unsqueeze(0)  # (1, Tout)
        out_mask = t >= out_lengths.unsqueeze(1)  # True where padded
        return out_mask

    def forward(self, x, y=None, padding_mask: torch.Tensor | None = None):
        if self.training:
            if y is None:
                x = self.augment(x)
                mix_info = None
            else:
                x, y, mix_info = self.augment(x, y)
        else:
            mix_info = None

        x = self.pre(x)
        z = self.cnn(x)
        z = z.transpose(1, 2)
        tr_mask = self._downsample_padding_mask(padding_mask, out_len=z.size(1))
        z = self.pos(z)
        z = self.tr(z, src_key_padding_mask=tr_mask)
        z = self.norm(z)

        if tr_mask is None:
            pooled = z.mean(dim=1)
        else:
            valid = (~tr_mask).unsqueeze(-1).to(dtype=z.dtype)
            pooled = (z * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

        logits = self.cls(pooled)

        return (logits, y, mix_info) if y is not None else logits
