import numpy as np

import torch
import torch.nn.functional as F

import torchvision as tv

from utils import transforms
from model.esresnet.base import _ESResNet
from model.esresnet.base import Bottleneck

from typing import cast
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional


class LinearFBSP(torch.nn.Module):

    def __init__(self, out_features: int, bias: bool = True, normalized: bool = False):
        super(LinearFBSP, self).__init__()

        self.out_features = out_features
        self.normalized = normalized
        self.eps = 1e-8

        default_dtype = torch.get_default_dtype()

        self.register_parameter('m', torch.nn.Parameter(torch.zeros(self.out_features, dtype=default_dtype)))
        self.register_parameter('fb', torch.nn.Parameter(torch.ones(self.out_features, dtype=default_dtype)))
        self.register_parameter('fc', torch.nn.Parameter(torch.arange(self.out_features, dtype=default_dtype)))
        self.register_parameter(
            'bias',
            torch.nn.Parameter(
                torch.normal(
                    0.0, 0.5, (self.out_features, 2), dtype=default_dtype
                ) if bias else cast(
                    torch.nn.Parameter, None
                )
            )
        )

        self.m.register_hook(lambda grad: grad / (torch.norm(grad, p=float('inf')) + self.eps))
        self.fb.register_hook(lambda grad: grad / (torch.norm(grad, p=float('inf')) + self.eps))
        self.fc.register_hook(lambda grad: grad / (torch.norm(grad, p=float('inf')) + self.eps))

    @staticmethod
    def power(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        magnitudes = (x1[..., 0] ** 2 + x1[..., 1] ** 2) ** 0.5
        phases = x1[..., 1].atan2(x1[..., 0])

        power_real = x2[..., 0]
        power_imag = x2[..., 1]

        mag_out = ((magnitudes ** 2) ** (0.5 * power_real) * torch.exp(-power_imag * phases))

        return mag_out.unsqueeze(-1) * torch.stack((
            (power_real * phases + 0.5 * power_imag * (magnitudes ** 2).log()).cos(),
            (power_real * phases + 0.5 * power_imag * (magnitudes ** 2).log()).sin()
        ), dim=-1)

    @staticmethod
    def sinc(x: torch.Tensor) -> torch.Tensor:
        return torch.where(cast(torch.Tensor, x == 0), torch.ones_like(x), torch.sin(x) / x)

    def _materialize_weights(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        x_is_complex = x.shape[-1] == 2
        in_features = x.shape[-1 - int(x_is_complex)]

        t = np.pi * torch.linspace(-1.0, 1.0, in_features, dtype=x.dtype, device=x.device).reshape(1, -1, 1) + self.eps

        m = self.m.reshape(-1, 1, 1)
        fb = self.fb.reshape(-1, 1, 1)
        fc = self.fc.reshape(-1, 1, 1)

        kernel = torch.cat((torch.cos(fc * t), -torch.sin(fc * t)), dim=-1)  # complex
        scale = fb.sqrt()  # real
        win = self.sinc(fb * t / (m + self.eps))  # real
        win = self.power(
            torch.cat((win, torch.zeros_like(win)), dim=-1),
            torch.cat((m, torch.zeros_like(m)), dim=-1)
        )  # complex

        weights = scale * torch.cat((
            win[..., :1] * kernel[..., :1] - win[..., 1:] * kernel[..., 1:],
            win[..., :1] * kernel[..., 1:] + win[..., 1:] * kernel[..., :1]
        ), dim=-1)

        if self.normalized:
            weights = weights / (in_features ** 0.5)

        return weights, x_is_complex

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights, x_is_complex = self._materialize_weights(x)

        if x_is_complex:
            x = torch.stack((
                F.linear(x[..., 0], weights[..., 0]) - F.linear(x[..., 1], weights[..., 1]),
                F.linear(x[..., 0], weights[..., 1]) + F.linear(x[..., 1], weights[..., 0])
            ), dim=-1)
        else:
            x = torch.stack((
                F.linear(x, weights[..., 0]),
                F.linear(x, weights[..., 1])
            ), dim=-1)

        if (self.bias is not None) and (self.bias.numel() == (self.out_features * 2)):
            x = x + self.bias

        return x, weights

    def extra_repr(self) -> str:
        return 'out_features={}, bias={}, normalized={}'.format(
            self.out_features,
            (self.bias is not None) and (self.bias.numel() == (self.out_features * 2)),
            self.normalized
        )


ttf_weights = dict()


class _ESResNetFBSP(_ESResNet):

    def _inject_members(self):
        self.add_module(
            'fbsp',
            LinearFBSP(
                out_features=int(round(self.n_fft / 2)) + 1 if self.onesided else self.n_fft,
                normalized=self.normalized,
                bias=False
            )
        )

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            frames = transforms.frame_signal(
                signal=x.view(-1, x.shape[-1]),
                frame_length=self.win_length,
                hop_length=self.hop_length,
                window=self.window
            )

            if self.n_fft > self.win_length:
                pad_length = self.n_fft - self.win_length
                pad_left = pad_length // 2
                pad_right = pad_length - pad_left
                frames = F.pad(frames, [pad_left, pad_right])

        spec, ttf_weights_ = self.fbsp(frames)

        spec = spec.transpose(-2, -3)
        ttf_weights[x.device] = ttf_weights_

        return spec

    def loss_ttf(self, device: torch.device) -> torch.Tensor:
        ttf_norm = torch.norm(ttf_weights[device], p=2, dim=[-1, -2])
        loss_ttf_norm = F.mse_loss(
            ttf_norm,
            torch.full_like(ttf_norm, 1.0 if self.normalized else self.n_fft ** 0.5)
        )

        return loss_ttf_norm

    def loss_fn(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss_pred = super(_ESResNetFBSP, self).loss_fn(y_pred, y)
        loss_ttf_norm = self.loss_ttf(y_pred.device)
        loss = loss_pred + loss_ttf_norm

        return loss


class ESResNetFBSP(_ESResNetFBSP):

    loading_func = staticmethod(tv.models.resnet50)

    def __init__(self,
                 n_fft: int = 256,
                 hop_length: Optional[int] = None,
                 win_length: Optional[int] = None,
                 window: Optional[str] = None,
                 normalized: bool = False,
                 onesided: bool = True,
                 spec_height: int = 224,
                 spec_width: int = 224,
                 num_classes: int = 1000,
                 apply_attention: bool = False,
                 pretrained: bool = False,
                 lock_pretrained: Optional[Union[bool, List[str]]] = None):

        super(ESResNetFBSP, self).__init__(
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            apply_attention=apply_attention,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            normalized=normalized,
            onesided=onesided,
            spec_height=spec_height,
            spec_width=spec_width,
            num_classes=num_classes,
            pretrained=pretrained,
            lock_pretrained=lock_pretrained
        )


class ESResNeXtFBSP(_ESResNetFBSP):

    loading_func = staticmethod(tv.models.resnext50_32x4d)

    def __init__(self,
                 n_fft: int = 256,
                 hop_length: Optional[int] = None,
                 win_length: Optional[int] = None,
                 window: Optional[str] = None,
                 normalized: bool = False,
                 onesided: bool = True,
                 spec_height: int = 224,
                 spec_width: int = 224,
                 num_classes: int = 1000,
                 apply_attention: bool = False,
                 pretrained: Union[bool, str] = False,
                 lock_pretrained: Optional[Union[bool, List[str]]] = None):

        super(ESResNeXtFBSP, self).__init__(
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            apply_attention=apply_attention,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            normalized=normalized,
            onesided=onesided,
            spec_height=spec_height,
            spec_width=spec_width,
            num_classes=num_classes,
            pretrained=pretrained,
            lock_pretrained=lock_pretrained,
            groups=32,
            width_per_group=4
        )
