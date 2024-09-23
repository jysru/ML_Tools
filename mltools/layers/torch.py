import torch
import torch.nn as nn
from torch.nn import functional as F



class Abs(nn.Module):

    def forward(self, x):
        if x.dtype.is_complex:
            return torch.abs(x)
        return torch.sqrt(torch.square(x[..., 0]) + torch.square(x[..., 1]))
    


class Angle(nn.Module):

    def forward(self, x):
        if x.dtype.is_complex:
            return torch.angle(x)



class _CRLinear(nn.Module):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None):
        super().__init__()
        dtype = torch.zeros((1,), dtype=dtype).real.dtype
        self.layer_re = nn.Linear(in_features, out_features, bias, device, dtype)
        self.layer_im = nn.Linear(in_features, out_features, bias, device, dtype)

    def forward(self, x: torch.Tensor):
        is_complex = x.dtype.is_complex
        if is_complex:
            x = torch.view_as_real(x)
        x_re, x_im = x[..., 0], x[..., 1]
        dot_re = F.linear(x_re, self.layer_re.weight, self.layer_re.bias) - F.linear(x_im, self.layer_im.weight)
        dot_im = F.linear(x_im, self.layer_re.weight, self.layer_im.bias) + F.linear(x_re, self.layer_im.weight)
        dot = torch.concat([dot_re[..., None], dot_im[..., None]], dim=-1)
        if is_complex:
            dot = torch.view_as_complex(dot)
        return dot



class Linear(nn.Module):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 real2_domain=False):
        super().__init__()
        if dtype is None:
            dtype = torch.float
        self.layer = nn.Linear(in_features, out_features, bias, device, dtype)
        if dtype.is_complex and real2_domain:
            self.layer = _CRLinear(in_features, out_features, bias, device, dtype)

    def forward(self, x):
        return self.layer(x)