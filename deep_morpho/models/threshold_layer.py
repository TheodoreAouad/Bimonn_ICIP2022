import torch
import torch.nn as nn

from ..threshold_fn import *


class ThresholdLayer(nn.Module):

    def __init__(
        self,
        threshold_fn,
        P_: float = 1,
        n_channels: int = 1,
        axis_channels: int = 1,
        threshold_name: str = '',
        bias: float = 0,
        constant_P: bool = False,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.axis_channels = axis_channels
        self.threshold_name = threshold_name
        self.threshold_fn = threshold_fn
        self.bias = bias

        if isinstance(P_, nn.Parameter):
            self.P_ = P_
        else:
            self.P_ = nn.Parameter(torch.tensor([P_ for _ in range(n_channels)]).float())
        if constant_P:
            self.P_.requires_grad = False

    def forward(self, x):
        # print((x + self.bias).shape)
        # print(self.P_.view(*([len(self.P_)] + [1 for _ in range(x.ndim - 1)])).shape)
        # return self.threshold_fn(
        #     (x + self.bias) * self.P_.view(*([1 for _ in range(self.axis_channels)] + [len(self.P_)] + [1 for _ in range(self.axis_channels, x.ndim - 1)]))
        # )
        return self.apply_threshold(x, self.P_, self.bias)

    def apply_threshold(self, x, P_, bias):
        return self.threshold_fn(
            (x + bias) * P_.view(*([1 for _ in range(self.axis_channels)] + [len(P_)] + [1 for _ in range(self.axis_channels, x.ndim - 1)]))
        )


class SigmoidLayer(ThresholdLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(threshold_fn=sigmoid_threshold, threshold_name='sigmoid', *args, **kwargs)


class ArctanLayer(ThresholdLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(threshold_fn=arctan_threshold, threshold_name='arctan', *args, **kwargs)


class TanhLayer(ThresholdLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(threshold_fn=tanh_threshold, threshold_name='tanh', *args, **kwargs)


class ErfLayer(ThresholdLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(threshold_fn=erf_threshold, threshold_name='erf', *args, **kwargs)


class ClampLayer(ThresholdLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(threshold_fn=lambda x: clamp_threshold(x, 0, 1), threshold_name='clamp', *args, **kwargs)


class IdentityLayer(ThresholdLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(threshold_fn=lambda x: x, threshold_name='identity', *args, **kwargs)

dispatcher = {
    'sigmoid': SigmoidLayer, 'arctan': ArctanLayer, 'tanh': TanhLayer, 'erf': ErfLayer, 'clamp': ClampLayer, 'identity': IdentityLayer
}

