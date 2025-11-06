# File: wavenet.py
# Created Date: Friday July 29th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
WaveNet implementation
https://arxiv.org/abs/1609.03499
"""

import abc as _abc
import random as _random
from contextlib import contextmanager as _contextmanager
from copy import deepcopy as _deepcopy
from pathlib import Path as _Path
from tempfile import TemporaryDirectory as _TemporaryDirectory
from typing import (
    Dict as _Dict,
    Optional as _Optional,
    Sequence as _Sequence,
    Tuple as _Tuple,
)

import numpy as _np
import torch as _torch
import torch.nn as _nn

from nam.models._abc import ImportsWeights as _ImportsWeights
from nam.models._activations import get_activation as _get_activation
from nam.models.base import BaseNet as _BaseNet
from nam.models._names import (
    ACTIVATION_NAME as _ACTIVATION_NAME,
    CONV_NAME as _CONV_NAME,
)


class _Slimmable(_abc.ABC, _nn.Module):
    """
    Mixin for slimmable modules.
    """

    def __init__(self):
        super().__init__()
        # Dicatates at what size the layer is currently operating.
        self._adjust_size: _Optional[int] = None

    @_abc.abstractmethod
    def max_adjust_size(self) -> int:
        """
        The maximum size that this module can be adjusted to.
        """
        pass

    def adjust_to(self, adjust_size: _Optional[int] = None, _recurse: bool = True):
        self._adjust_size = adjust_size
        if _recurse:
            # Since .modules() recurses, don't have the modules found recurse themselves or else we'll visit things twice.
            for module in self.modules():
                if module is self:
                    continue
                if isinstance(module, _Slimmable):
                    module.adjust_to(adjust_size, _recurse=False)

    def adjust_to_random(self):
        a = _random.randint(1, self.max_adjust_size())
        self.adjust_to(a)

    @_contextmanager
    def context_adjust_to_random(self):
        self.adjust_to_random()
        try:
            yield
        finally:
            self.adjust_to(None)


class _Conv1d(_nn.Conv1d, _Slimmable):
    """
    Base class for the various 1D convolution layers used.

    Subclasses differ in how they adjust their size.
    """

    def export_weights(self) -> _torch.Tensor:
        tensors = []
        if self.weight is not None:
            tensors.append(self.weight.data.flatten())
        if self.bias is not None:
            tensors.append(self.bias.data.flatten())
        if len(tensors) == 0:
            return _torch.zeros((0,))
        else:
            return _torch.cat(tensors)

    def forward(self, input: _torch.Tensor) -> _torch.Tensor:
        w, b = (
            (self.weight, self.bias)
            if self._adjust_size is None
            else self._get_adjusted_weight_and_bias()
        )
        return _nn.functional.conv1d(
            input, w, b, self.stride, self.padding, self.dilation, self.groups
        )

    def import_weights(self, weights: _torch.Tensor, i: int) -> int:
        if self.weight is not None:
            n = self.weight.numel()
            self.weight.data = (
                weights[i : i + n].reshape(self.weight.shape).to(self.weight.device)
            )
            i += n
        if self.bias is not None:
            n = self.bias.numel()
            self.bias.data = (
                weights[i : i + n].reshape(self.bias.shape).to(self.bias.device)
            )
            i += n
        return i

    @_abc.abstractmethod
    def _get_adjusted_weight_and_bias(
        self,
    ) -> _Tuple[_torch.Tensor, _Optional[_torch.Tensor]]:
        """
        Get the weight and bias tensors for the current adjust size.
        """
        pass


class _ConvIn(_Conv1d):
    """
    The convolution layer that takes the previous layer's output as input
    """

    def __init__(self, *args, gated: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._gated = gated

    def max_adjust_size(self) -> int:
        return self.in_channels

    def _get_adjusted_weight_and_bias(
        self,
    ) -> _Tuple[_torch.Tensor, _Optional[_torch.Tensor]]:
        # COut, CIn, K
        if self._gated:
            raise NotImplementedError("Gated in-conv not implemented")
        w = self.weight[: self._adjust_size, : self._adjust_size, :]
        b = None if self.bias is None else self.bias[: self._adjust_size]
        return w, b


class _InputMixer(_Conv1d):
    """
    The convolution layer that takes the "skip-in" condition input.
    """

    def __init__(self, *args, gated: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._gated = gated

    def max_adjust_size(self) -> int:
        return self.out_channels if not self._gated else self.out_channels // 2

    def _get_adjusted_weight_and_bias(
        self,
    ) -> _Tuple[_torch.Tensor, _Optional[_torch.Tensor]]:
        if self._gated:
            raise NotImplementedError("Gated input mixer not implemented")
        # COut, CIn, K
        # Input channels don't change
        w = self.weight[: self._adjust_size, :, :]
        b = None if self.bias is None else self.bias[: self._adjust_size]
        return w, b


class _1x1(_Conv1d):
    """
    The 1x1 convolution layer
    """

    def max_adjust_size(self) -> int:
        if self.in_channels != self.out_channels:
            raise NotImplementedError(
                "1x1 conv with different input and output channels not implemented"
            )
        else:
            return self.in_channels

    def _get_adjusted_weight_and_bias(
        self,
    ) -> _Tuple[_torch.Tensor, _Optional[_torch.Tensor]]:
        # COut, CIn, K
        w = self.weight[: self._adjust_size, : self._adjust_size, :]
        b = None if self.bias is None else self.bias[: self._adjust_size]
        return w, b


class _Rechannel(_Conv1d):
    """
    Start of a WaveNet Layer array

    NOTE: Doesn't know how to know the size of the previous layer array. We'll
    figure that out later.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def max_adjust_size(self) -> int:
        return self.out_channels

    def _get_adjusted_weight_and_bias(
        self,
    ) -> _Tuple[_torch.Tensor, _Optional[_torch.Tensor]]:
        # FIXME previous layer array's output channels
        w = self.weight[: self._adjust_size, :, :]
        b = None if self.bias is None else self.bias[: self._adjust_size]
        return w, b


class _HeadRechannel(_Conv1d):
    """
    Exiting a WaveNet layer to the head array
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def max_adjust_size(self) -> int:
        return self.in_channels

    def _get_adjusted_weight_and_bias(
        self,
    ) -> _Tuple[_torch.Tensor, _Optional[_torch.Tensor]]:
        # Out to 1
        if self.out_channels != 1:
            raise NotImplementedError(
                "Head rechannel only implemented for 1 output channel"
            )
        else:
            w = self.weight[:, : self._adjust_size, :]
            b = None if self.bias is None else self.bias
        return w, b


class _Layer(_nn.Module):
    """
    A WaveNet layer
    """

    def __init__(
        self,
        condition_size: int,
        channels: int,
        kernel_size: int,
        dilation: int,
        activation: str,
        gated: bool,
    ):
        super().__init__()
        # Input mixer takes care of the bias
        mid_channels = 2 * channels if gated else channels
        self._conv = _ConvIn(
            channels, mid_channels, kernel_size, dilation=dilation, gated=gated
        )
        # Custom init: favors direct input-output
        # self._conv.weight.data.zero_()
        self._input_mixer = _InputMixer(
            condition_size, mid_channels, 1, bias=False, gated=gated
        )
        self._activation = _get_activation(activation)
        self._activation_name = activation
        self._1x1 = _1x1(channels, channels, 1)
        self._gated = gated

    @property
    def activation_name(self) -> str:
        return self._activation_name

    @property
    def conv(self) -> _Conv1d:
        return self._conv

    @property
    def gated(self) -> bool:
        return self._gated

    @property
    def kernel_size(self) -> int:
        return self._conv.kernel_size[0]

    def export_weights(self) -> _torch.Tensor:
        return _torch.cat(
            [
                self.conv.export_weights(),
                self._input_mixer.export_weights(),
                self._1x1.export_weights(),
            ]
        )

    def forward(
        self, x: _torch.Tensor, h: _Optional[_torch.Tensor], out_length: int
    ) -> _Tuple[_Optional[_torch.Tensor], _torch.Tensor]:
        """
        :param x: (B,C,L1) From last layer
        :param h: (B,DX,L2) Conditioning. If first, ignored.

        :return:
            If not final:
                (B,C,L1-d) to next layer
                (B,C,L1-d) to mixer
            If final, next layer is None
        """
        zconv = self.conv(x)
        z1 = zconv + self._input_mixer(h)[:, :, -zconv.shape[2] :]
        current_channels = z1.shape[1] if not self._gated else z1.shape[1] // 2
        post_activation = (
            self._activation(z1)
            if not self._gated
            else (
                self._activation(z1[:, :current_channels])
                * _torch.sigmoid(z1[:, current_channels:])
            )
        )
        return (
            x[:, :, -post_activation.shape[2] :] + self._1x1(post_activation),
            post_activation[:, :, -out_length:],
        )

    def import_weights(self, weights: _torch.Tensor, i: int) -> int:
        i = self.conv.import_weights(weights, i)
        i = self._input_mixer.import_weights(weights, i)
        return self._1x1.import_weights(weights, i)

    @property
    def _channels(self) -> int:
        return self._1x1.in_channels


class _LayerArray(_nn.Module):
    """
    Takes in the input and condition (and maybe the head input so far); outputs the
    layer output and head input.

    The original WaveNet only uses one of these, but you can stack multiple of this
    module to vary the channels throughout with minimal extra channel-changing conv
    layers.
    """

    def __init__(
        self,
        input_size: int,
        condition_size: int,
        head_size,
        channels: int,
        kernel_size: int,
        dilations: _Sequence[int],
        activation: str = "Tanh",
        gated: bool = True,
        head_bias: bool = True,
    ):
        super().__init__()
        self._rechannel = _Rechannel(input_size, channels, 1, bias=False)
        self._layers = _nn.ModuleList(
            [
                _Layer(
                    condition_size, channels, kernel_size, dilation, activation, gated
                )
                for dilation in dilations
            ]
        )
        # Convert the head input from channels to head_size
        self._head_rechannel = _HeadRechannel(channels, head_size, 1, bias=head_bias)

        self._config = {
            "input_size": input_size,
            "condition_size": condition_size,
            "head_size": head_size,
            "channels": channels,
            "kernel_size": kernel_size,
            "dilations": dilations,
            "activation": activation,
            "gated": gated,
            "head_bias": head_bias,
        }

    @property
    def receptive_field(self) -> int:
        return 1 + (self._kernel_size - 1) * sum(self._dilations)

    def export_config(self):
        return _deepcopy(self._config)

    def export_weights(self) -> _torch.Tensor:
        return _torch.cat(
            [self._rechannel.export_weights()]
            + [layer.export_weights() for layer in self._layers]
            + [self._head_rechannel.export_weights()]
        )

    def import_weights(self, weights: _torch.Tensor, i: int) -> int:
        i = self._rechannel.import_weights(weights, i)
        for layer in self._layers:
            i = layer.import_weights(weights, i)
        return self._head_rechannel.import_weights(weights, i)

    def forward(
        self,
        x: _torch.Tensor,
        c: _torch.Tensor,
        head_input: _Optional[_torch.Tensor] = None,
    ) -> _Tuple[_torch.Tensor, _torch.Tensor]:
        """
        :param x: (B,Dx,L) layer input
        :param c: (B,Dc,L) condition

        :return:
            (B,Dc,L-R+1) head input
            (B,Dc,L-R+1) layer output
        """
        out_length = x.shape[2] - (self.receptive_field - 1)
        x = self._rechannel(x)
        for layer in self._layers:
            x, head_term = layer(x, c, out_length)  # Ensures head_term sample length
            head_input = (
                head_term
                if head_input is None
                else head_input[:, :, -out_length:] + head_term
            )
        return self._head_rechannel(head_input), x

    @property
    def _dilations(self) -> _Sequence[int]:
        return self._config["dilations"]

    @property
    def _kernel_size(self) -> int:
        return self._layers[0].kernel_size


class _Head(_nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        activation: str,
        num_layers: int,
        out_channels: int,
    ):
        super().__init__()

        def block(cx, cy):
            net = _nn.Sequential()
            net.add_module(_ACTIVATION_NAME, _get_activation(activation))
            net.add_module(_CONV_NAME, _Conv1d(cx, cy, 1))
            return net

        assert num_layers > 0

        layers = _nn.Sequential()
        cin = in_channels
        for i in range(num_layers):
            layers.add_module(
                f"layer_{i}",
                block(cin, channels if i != num_layers - 1 else out_channels),
            )
            cin = channels
        self._layers = layers

        self._config = {
            "channels": channels,
            "activation": activation,
            "num_layers": num_layers,
            "out_channels": out_channels,
        }

    def export_config(self):
        return _deepcopy(self._config)

    def export_weights(self) -> _torch.Tensor:
        return _torch.cat([layer[1].export_weights() for layer in self._layers])

    def forward(self, *args, **kwargs):
        return self._layers(*args, **kwargs)

    def import_weights(self, weights: _torch.Tensor, i: int) -> int:
        for layer in self._layers:
            i = layer[1].import_weights(weights, i)
        return i


class _WaveNet(_Slimmable):
    def __init__(
        self,
        layers_configs: _Sequence[_Dict],
        head_config: _Optional[_Dict] = None,
        head_scale: float = 1.0,
    ):
        super().__init__()

        if len(layers_configs) > 1:
            raise NotImplementedError("Multiple layer arrays not implemented")
        self._layer_arrays = _nn.ModuleList(
            [_LayerArray(**lc) for lc in layers_configs]
        )
        self._head = None if head_config is None else _Head(**head_config)
        self._head_scale = head_scale

    @property
    def receptive_field(self) -> int:
        return 1 + sum([(layer.receptive_field - 1) for layer in self._layer_arrays])

    def export_config(self):
        return {
            "layers": [layers.export_config() for layers in self._layer_arrays],
            "head": None if self._head is None else self._head.export_config(),
            "head_scale": self._head_scale,
        }

    def export_weights(self) -> _np.ndarray:
        """
        :return: 1D array
        """
        weights = _torch.cat([layer.export_weights() for layer in self._layer_arrays])
        if self._head is not None:
            weights = _torch.cat([weights, self._head.export_weights()])
        weights = _torch.cat([weights.cpu(), _torch.Tensor([self._head_scale])])
        return weights.detach().cpu().numpy()

    def import_weights(self, weights: _torch.Tensor):
        if self._head is not None:
            raise NotImplementedError("Head importing isn't implemented yet.")
        i = 0
        for layer in self._layer_arrays:
            i = layer.import_weights(weights, i)

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        """
        :param x: (B,Cx,L)
        :return: (B,Cy,L-R)
        """
        y, head_input = x, None
        for layer in self._layer_arrays:
            head_input, y = layer(y, x, head_input=head_input)
        head_input = self._head_scale * head_input
        return head_input if self._head is None else self._head(head_input)

    def max_adjust_size(self) -> int:
        return min(
            [
                m.max_adjust_size()
                for m in self.modules()
                if m is not self and isinstance(m, _Slimmable)
            ]
        )


class SlimmableWaveNet(_BaseNet, _ImportsWeights):
    def __init__(self, *args, sample_rate: _Optional[float] = None, **kwargs):
        super().__init__(sample_rate=sample_rate)
        self._net = _WaveNet(*args, **kwargs)

    @classmethod
    def registry_key(cls) -> str:
        return "SlimmableWaveNet"

    @property
    def pad_start_default(self) -> bool:
        return True

    @property
    def receptive_field(self) -> int:
        return self._net.receptive_field

    def import_weights(self, weights: _Sequence[float]):
        if not isinstance(weights, _torch.Tensor):
            weights = _torch.Tensor(weights)
        self._net.import_weights(weights)

    def _export_config(self):
        return self._net.export_config()

    def _export_weights(self) -> _np.ndarray:
        return self._net.export_weights()

    def _forward(self, x):
        def forward(x):
            if x.ndim == 2:
                x = x[:, None, :]
            y = self._net(x)
            assert y.shape[1] == 1
            return y[:, 0, :]

        if self.training:
            with self._net.context_adjust_to_random():
                return forward(x)
        else:
            return forward(x)
