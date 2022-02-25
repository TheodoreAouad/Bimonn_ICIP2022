from typing import List, Union
import math

import torch
import torch.nn as nn
from torch.nn.init import kaiming_uniform_

from general.utils import max_min_norm

from .unfolder import Unfolder


class UnfoldingLayer(nn.Module):

    def __init__(
        self,
        kernel_size,
        init_P,
        padding: Union[int, str] = 'same',
        padding_value: float = 0,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        if padding == 'same':
            if isinstance(kernel_size, int):
                padding = kernel_size // 2
            else:
                padding = tuple(sum([[k//2, k // 2] for k in kernel_size], start=[]))
        self.padding = tuple([padding for _ in range(self.ndim * 2)]) if not isinstance(padding, tuple) else padding

        self.P = nn.Parameter(torch.FloatTensor([init_P]), requires_grad=True)
        # self.P = None  # TODO: init P
        self.unfolder = Unfolder(kernel_size=kernel_size, padding=padding, padding_value=padding_value)
        self.weight = nn.Parameter(self.init_weights()).float()
        self.pooling = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=kernel_size)
        self.pooling.weight.data = torch.ones_like(self.pooling.weight.data, requires_grad=False)


    def forward(self, x):
        raise NotImplementedError


    def init_weights(self):
        # weights = torch.zeros(self.kernel_size) + init_value
        # weights[self.kernel_size[0]//2, self.kernel_size[1]//2] = 0
        # weights = torch.randn(self.kernel_size)
        # TODO
        return

    def kron_weight(self, size):
        return torch.kron(torch.ones([self.padding[2*k] + size[k] + self.padding[2*k + 1] - self.kernel_size[k] + 1 for k in range(self.ndim)], device=self.weight.device), self.weight)

    @property
    def ndim(self):
        return len(self.kernel_size)

    @property
    def weights(self):
        return self.weight


class LMorph(UnfoldingLayer):
    """ We implement the Lmorph layer described in https://arxiv.org/pdf/2102.10038.pdf
    """
    def __init__(self, kernel_size, init_P=0, *args, **kwargs):
        super().__init__(kernel_size=kernel_size, init_P=init_P, padding_value=1, *args, **kwargs)


    def forward(self, x):
        output = max_min_norm(x) + 1  # rescale between [1, 2]
        output = self.unfolder(output)
        output = output + self.kron_weight(x.shape[-self.ndim:])
        output = self.pooling(output ** (self.P+1)) / self.pooling(output ** self.P)
        return output

    def init_weights(self, std=0.01):
        return (torch.randn(self.kernel_size) * std).abs()


class SMorph(UnfoldingLayer):
    """ We implement the Smorph layer described in https://arxiv.org/pdf/2102.10038.pdf
    """

    def __init__(self, kernel_size, init_P=0, *args, **kwargs):
        super().__init__(kernel_size=kernel_size, init_P=init_P, *args, **kwargs)

    def forward(self, x):
        output = self.unfolder(x)
        output = output + self.kron_weight(x.shape[-self.ndim:])
        exp_term = torch.exp(self.alpha * output)
        output = (self.pooling(output * exp_term) / self.pooling(exp_term))
        return output

    @property
    def alpha(self):
        return self.P

    def init_weights(self, std=0.01):
        return torch.randn(self.kernel_size) * std



class AdaptativeMorphologicalLayer(UnfoldingLayer):
    """ We implement the adaptative morphogolical layer described in https://arxiv.org/pdf/1909.01532.pdf
    """

    def __init__(self, kernel_size, init_P=0, *args, **kwargs):
        super().__init__(kernel_size=kernel_size, init_P=init_P, *args, **kwargs)
        self.b = nn.Parameter(torch.FloatTensor([0]))


    def forward(self, x):
        output = self.unfolder(x)
        output = self.pooling(torch.exp(self.softsign(self.A) * self.kron_weight(x.shape[-self.ndim:]) * output))
        output = self.softsign(self.A) * torch.log(output) + self.b
        return output

    @staticmethod
    def softsign(x):
        return torch.tanh(x)

    @property
    def A(self):
        return self.P

    def init_weights(self):
        weights = nn.Parameter(torch.FloatTensor(size=self.kernel_size))
        kaiming_uniform_(weights, a=math.sqrt(5))
        return weights




class NetMulti(nn.Module):
    layer_dict = {"LMorph": LMorph, "SMorph": SMorph, "AdaptativeMorphologicalLayer": AdaptativeMorphologicalLayer}

    def __init__(self, kernel_size: List, name: str):
        super().__init__()
        self.name = name
        self.layer_init = self.layer_dict[name]

        self.kernel_size = kernel_size
        self.layers = self.init_layers()
        self.rescaler = nn.Conv2d(1, 1, 1)

    def init_layers(self):
        layers = []
        for idx, kernel_size in enumerate(self.kernel_size):
            cur_layer = self.layer_init(kernel_size)
            setattr(self, f"layer{idx+1}", cur_layer)
            layers.append(cur_layer)
        return layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class NetLSMorph(NetMulti):

    def __init__(self, kernel_size: List, name: str):
        super().__init__(kernel_size, name)
        self.rescaler = nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.rescaler(super().forward(x))


class NetLMorph(NetLSMorph):
    def __init__(self, kernel_size: List):
        super().__init__(kernel_size, name="LMorph")


class NetSMorph(NetLSMorph):
    def __init__(self, kernel_size: List):
        super().__init__(kernel_size, name="SMorph")


class NetAdaptativeMorphologicalLayer(NetMulti):

    def __init__(self, kernel_size):
        super().__init__(kernel_size=kernel_size, name="AdaptativeMorphologicalLayer")
