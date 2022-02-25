from typing import List, Tuple, Union, Dict

import torch.nn as nn
import numpy as np

from .bise import BiSE
from .bisel import BiSEL


class BiMoNN(nn.Module):

    def __init__(
        self,
        kernel_size: List[Union[Tuple, int]],
        channels: List[int],
        atomic_element: Union[str, List[str]] = 'bise',
        weight_P: Union[float, List[float]] = 1,
        threshold_mode: Union[Union[str, dict], List[Union[str, dict]]] = "tanh",
        activation_P: Union[float, List[float]] = 10,
        constant_activation_P: Union[bool, List[bool]] = False,
        constant_weight_P: Union[bool, List[bool]] = False,
        init_bias_value: Union[float, List[float]] = -2,
        init_weight_mode: Union[bool, List[bool]] = True,
        alpha_init: Union[float, List[float]] = 0,
        init_value: Union[float, List[float]] = -2,
        share_weights: Union[bool, List[bool]] = True,
        constant_P_lui: Union[bool, List[bool]] = False,
        lui_kwargs: Union[Dict, List[Dict]] = {},
    ):
        super().__init__()
        self.kernel_size = self._init_kernel_size(kernel_size)

        for attr in set(self.all_args):
            setattr(self, attr, self._init_attr(attr, eval(attr)))

        self.layers = []
        self.bises_idx = []
        self.bisels_idx = []
        for idx in range(len(self)):
            layer = self._make_layer(idx)
            self.layers.append(layer)
            setattr(self, f'layer{idx+1}', layer)

    @property
    def bises(self):
        return [self.layers[idx] for idx in self.bises_idx]


    @property
    def bisels(self):
        return [self.layers[idx] for idx in self.bisels_idx]

    def forward(self, x):
        output = self.layers[0](x)
        for layer in self.layers[1:]:
            output = layer(output)
        return output

    def get_bise_selems(self) -> Tuple[Dict[int, np.ndarray], Dict[int, str]]:
        """Go through all BiSE indexes and shows the learned selem and operation. If None are learned, puts value
        None.

        Returns:
            (dict, dict): the dictionary of learned selems and operations with indexes as keys.
        """
        selems = {}
        operations = {}
        v1, v2 = 0, 1
        for idx in self.bises_idx:
            if v1 is not None:
                selems[idx], operations[idx] = self.layers[idx].find_selem_and_operation(v1, v2)
                v1, v2 = self.layers[idx].get_outputs_bounds(v1, v2)
            else:
                selems[idx], operations[idx] = None, None
        return selems, operations

    def __len__(self):
        return len(self.kernel_size)

    @staticmethod
    def _init_kernel_size(kernel_size: List[Union[Tuple, int]]):
        res = []
        for size in kernel_size:
            if isinstance(size, int):
                res.append((size, size))
            else:
                res.append(size)
        return res

    def _init_channels(self, channels: List[int]):
        self.out_channels = channels[1:]
        self.in_channels = channels[:-1]
        self.channels = channels
        return self.channels

    def _init_atomic_element(self, atomic_element: Union[str, List[str]]):
        if isinstance(atomic_element, list):
            return [s.lower() for s in atomic_element]

        return [atomic_element.lower() for _ in range(len(self))]

    def _init_attr(self, attr_name, attr_value):
        if attr_name == "kernel_size":
            # return self._init_kernel_size(attr_value)
            return self.kernel_size

        if attr_name == "atomic_element":
            return self._init_atomic_element(attr_value)

        if attr_name == "channels":
            return self._init_channels(attr_value)

        if isinstance(attr_value, list):
            return attr_value

        return [attr_value for _ in range(len(self))]

    def bises_kwargs_idx(self, idx):
        return dict(
            **{'shared_weights': None, 'shared_weight_P': None},
            **{k: getattr(self, k)[idx] for k in self.bises_args}
        )

    def bisels_kwargs_idx(self, idx):
        return dict(
            **{'shared_weights': None, 'shared_weight_P': None},
            **{k: getattr(self, k)[idx] for k in self.bisels_args}
        )

    @property
    def bises_args(self):
        return [
            'kernel_size', 'weight_P', 'threshold_mode', 'activation_P',
            'init_bias_value', 'init_weight_mode', 'out_channels', "constant_activation_P",
            "constant_weight_P"
        ]

    @property
    def bisels_args(self):
        return self.bises_args + ['in_channels', 'constant_P_lui', "lui_kwargs"]

    def _make_layer(self, idx):
        if self.atomic_element[idx] == 'bise':
            layer = BiSE(**self.bises_kwargs_idx(idx))
            self.bises_idx.append(idx)

        elif self.atomic_element[idx] == 'bisel':
            layer = BiSEL(**self.bisels_kwargs_idx(idx))
            self.bisels_idx.append(idx)


        return layer

    @property
    def all_args(self):
        return [
            "kernel_size", "atomic_element", "weight_P", "threshold_mode", "activation_P", "constant_activation_P",
            "init_bias_value", "init_weight_mode", "alpha_init", "init_value", "share_weights",
            "constant_weight_P", "constant_P_lui", "channels", "lui_kwargs",
        ]
