import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None
    
from megatron.core.extensions.transformer_engine import TELinear
from .dragon_config import DragonConfig

def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p) - math.log(1.0 - p)

class DepthwiseShortConv1d(nn.Module):
    def __init__(self, hidden_size: int, *, kernel_size: int, shift_right1: bool = False) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        if self.kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {self.kernel_size}.")
        self.shift_right1 = bool(shift_right1)
        self.weight = nn.Parameter(torch.empty(hidden_size, self.kernel_size))
        bound = 1 / math.sqrt(self.kernel_size) 
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).contiguous()  # (B, C, T)
        # pad_left = self.kernel_size - 1 + (1 if self.shift_right1 else 0)
        if self.shift_right1:
            # x = F.pad(x, (self.kernel_size, 0)).contiguous()
            x = F.pad(x, (1, 0)).contiguous()
            x = causal_conv1d_fn(x, weight=self.weight)[:, :, :-1]
        else:
            x = causal_conv1d_fn(x, weight=self.weight)
        if x.device.type == "mps":
            x = _Transpose12Contiguous.apply(x)
        else:
            x = x.transpose(1, 2).contiguous()  # (B, T, C)
        return x

class InputEmbedShortConvExpander(nn.Module):
    def __init__(self, config: DragonConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.value_channels = 4
        self.kernel_size = 4

        self.conv = nn.Conv1d(
            self.hidden_size,
            self.hidden_size * self.value_channels,
            kernel_size=self.kernel_size,
            padding=0,
            groups=self.hidden_size,
            bias=False,
        )

    def reset_parameters_identity(self) -> None:
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[:, 0, self.kernel_size - 1] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d) -> (B, T, d, d_v)
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (B, T, d), got {tuple(x.shape)}")
        B, T, d = x.shape
        if d != self.hidden_size:
            raise ValueError(f"Expected x feature dim {self.hidden_size}, got {d}.")

        x_t = x.transpose(1, 2).contiguous()  # (B, d, T)
        pad_left = self.kernel_size - 1
        x_t = F.pad(x_t, (pad_left, 0)).contiguous()
        y = self.conv(x_t)  # (B, d*d_v, T)
        y = y.transpose(1, 2).contiguous()  # (B, T, d*d_v)
        return y #.reshape(B, T, d, self.value_channels)

class ResidualShortConvCompressor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.value_channels = 4
        self.residual_size = self.hidden_size * self.value_channels
        self.shortconv = DepthwiseShortConv1d(self.residual_size, kernel_size=4)

        read_init_raw = None
        if read_init_raw is None:
            read_init = 1.0 / float(self.value_channels)
        else:
            read_init = float(read_init_raw)
        self.read = nn.Parameter(torch.full((self.value_channels,), read_init))

    def forward(self, x: torch.Tensor, return_past: bool = False) -> torch.Tensor:
        # x: (B, T, d, d_v)
        #print("ddl x.shape before rearrange: ", x.shape)
        assert return_past is False
        x = rearrange(x, 'l b (m c) -> b l m c', c=self.value_channels)
        #print("ddl x.shape : ", x.shape)
        B, T, d, dv = x.shape
        if d != self.hidden_size:
            raise ValueError(f"Expected residual d={self.hidden_size}, got {d}.")
        if dv != self.value_channels:
            raise ValueError(f"Expected residual d_v={self.value_channels}, got {dv}.")

        x_flat = x.reshape(B, T, self.residual_size)
        x_conv = self.shortconv(x_flat).reshape(B, T, d, dv)
        y = torch.sum(x_conv * self.read, dim=-1).transpose(0, 1).contiguous()
        #print("ddl y.shape : ", y.shape)
        if return_past:
            past = x_flat.transpose(1, 2)[:, :, -(self.shortconv.kernel_size - 1):].contiguous()
            return y, past
        else:
            return y

class DeepDeltaResidualExpanded(nn.Module):
    def __init__(self, config: DragonConfig, input_scalar: float):
        super().__init__()
        self.config = config
        
        hidden_size = config.hidden_size
        self.value_channels = 4
        self.k_eps = 1e-5
        self.v_sigmoid = True
        self.v_sigmoid_scale = 4.
        self.v_constant = False
        self.v_constant_value = 2.0

        self.beta_single_linear = True
        self.beta = TELinear(
            config.hidden_size,
            1,
            config=self.config,
            parallel_mode="duplicated",
            init_method=self.config.init_method,
            bias=True,
            alpha_fwd=input_scalar,
            alpha_bwd=input_scalar,
            skip_bias_add=True,
            skip_weight_param_allocation=False,
            tp_comm_buffer_name="ddl_beta_proj",
        )
        
        # v is a vector in R^{d_v} in the expanded-state regime.
        self.v_proj = TELinear(
            config.hidden_size,
            self.value_channels,
            config=self.config,
            parallel_mode="duplicated",
            init_method=self.config.init_method,
            bias=True,
            alpha_fwd=1.,
            alpha_bwd=1.,
            skip_bias_add=True,
            skip_weight_param_allocation=False,
            tp_comm_buffer_name="ddl_v_proj",
        )

        beta_init = 0.
        beta_init = min(max(beta_init, 0.0), 2.0)
        beta_init_p = beta_init / 2.0
        with torch.no_grad():
            if self.beta_single_linear:
                self.beta.bias.fill_(_logit(beta_init_p))
            else:
                self.beta_out.bias.fill_(_logit(beta_init_p))

    def forward(self, x: torch.Tensor, *, k_in: torch.Tensor, v_in: torch.Tensor, context: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d*d_v), k_in: (B, T, d), v_in: (B, T, d), context: (B, T, d)
        # Keep large tensors in the model dtype; only compute `beta` in fp32 for stability.
        #print("expand ddl x.shape before rearrange: ", x.shape)
        x = rearrange(x, 'l b (m c) -> l b m c', c=self.value_channels)
        #print("expand ddl x.shape after rearrange: ", x.shape)
        k_dim = int(k_in.size(-1))
        eps_rms = (self.k_eps * self.k_eps) / float(k_dim)
        k_rms = F.rms_norm(k_in, [k_dim], eps=eps_rms)
        k_scale = 1.0 / math.sqrt(k_dim)

        # beta(X) in [0, 2]
        beta_logits, beta_bias = self.beta(context)
        beta = 2.0 * torch.sigmoid(beta_logits.float() + beta_bias.float())  # fp32

        if x.ndim != 4:
            raise ValueError(f"Expected x with shape (B, T, d, d_v), got {tuple(x.shape)}")
        if int(x.size(-2)) != k_dim:
            raise ValueError(f"Expected x feature dim {k_dim}, got {int(x.size(-2))}.")
        if int(x.size(-1)) != self.value_channels:
            raise ValueError(f"Expected x value channels {self.value_channels}, got {int(x.size(-1))}.")

        # k^T X, row vector projection (B, T, d_v)
        proj_rms = torch.sum(k_rms.unsqueeze(-1) * x, dim=-2, dtype=torch.bfloat16)  # fp32
        proj = proj_rms * k_scale

        v, v_bias = self.v_proj(v_in)
        v = torch.sigmoid(v + v_bias) * self.v_sigmoid_scale

        # X <- X + beta * k * (v^T - k^T X)
        delta_row = (beta * (v - proj)) * k_scale  # fp32 (B, T, d_v)
        update = k_rms.unsqueeze(-1) * delta_row.to(dtype=x.dtype).unsqueeze(-2)  # (B, T, d, d_v)
        y= x + scalar * update
        #print("ddl y.shape before rearrange: ", y.shape)
        y = rearrange(y, 'b l m c -> l b (m c)').contiguous()
        #print("ddl y.shape after rearrange: ", y.shape)
        return y