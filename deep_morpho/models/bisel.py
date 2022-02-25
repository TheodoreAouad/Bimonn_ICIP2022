from typing import Union, Tuple, Dict

import torch
import torch.nn as nn

from .bise import BiSE
from .lui import LUI


class BiSEL(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        threshold_mode: str = 'sigmoid',
        constant_P_lui: bool = False,
        lui_kwargs: Dict = {},
        **bise_kwargs
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.threshold_mode = threshold_mode
        self.constant_P_lui = constant_P_lui
        self.bise_kwargs = bise_kwargs
        self.lui_kwargs = lui_kwargs

        self.bises = self._init_bises()
        self.luis = self._init_luis()

    def _init_bises(self):
        bises = []
        for idx in range(self.in_channels):
            layer = BiSE(
                out_channels=self.out_channels, kernel_size=self.kernel_size,
                threshold_mode=self.threshold_mode, **self.bise_kwargs
            )
            setattr(self, f'bise_{idx}', layer)
            bises.append(layer)
        return bises


    def _init_luis(self):
        luis = []
        for idx in range(self.out_channels):
            layer = LUI(
                chan_inputs=self.in_channels,
                threshold_mode=self.threshold_mode,
                chan_outputs=1,
                constant_P=self.constant_P_lui,
                **self.lui_kwargs
            )
            setattr(self, f'lui_{idx}', layer)
            luis.append(layer)
        return luis


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bise_res2 = torch.cat([
            layer(x[:, chan_input:chan_input+1, ...])[:, None, ...] for chan_input, layer in enumerate(self.bises)
        ], axis=1)  # bise_res shape: (batch_size, in_channels, out_channels, width, length)

        lui_res = torch.cat([
            layer(bise_res2[:, :, chan_output, ...]) for chan_output, layer in enumerate(self.luis)
        ], axis=1)

        return lui_res


    @property
    def weight(self) -> torch.Tensor:
        """ Returns the convolution weights, of shape (out_channels, in_channels, W, L).
        """
        return torch.cat([layer.weight for layer in self.bises], axis=1)

    @property
    def weights(self) -> torch.Tensor:
        return self.weight

    @property
    def activation_P_bise(self) -> torch.Tensor:
        """ Returns the activations P of the bise layers, of shape (out_channels, in_channels).
        """
        return torch.stack([layer.activation_P for layer in self.bises], axis=-1)

    @property
    def weight_P_bise(self) -> torch.Tensor:
        """ Returns the weights P of the bise layers, of shape (out_channels, in_channels).
        """
        return torch.stack([layer.weight_P for layer in self.bises], axis=-1)


    @property
    def activation_P_lui(self) -> torch.Tensor:
        """ Returns the activations P of the lui layer, of shape (out_channels).
        """
        return torch.cat([layer.activation_P for layer in self.luis])


    @property
    def bias_bise(self) -> torch.Tensor:
        """ Returns the bias of the bise layers, of shape (out_channels, in_channels).
        """
        return torch.stack([layer.bias for layer in self.bises], axis=-1)

    @property
    def bias_bises(self) -> torch.Tensor:
        return self.bias_bise

    @property
    def bias_lui(self) -> torch.Tensor:
        """Returns the bias of the lui layer, of shape (out_channels).
        """
        return torch.cat([layer.bias for layer in self.luis])

    @property
    def bias_luis(self) -> torch.Tensor:
        return self.bias_lui

    @property
    def normalized_weight(self) -> torch.Tensor:
        """ Returns the convolution weights, of shape (out_channels, in_channels, W, L).
        """
        return torch.cat([layer._normalized_weight for layer in self.bises], axis=1)

    @property
    def normalized_weights(self) -> torch.Tensor:
        return self.normalized_weight

    @property
    def _normalized_weight(self) -> torch.Tensor:
        return self.normalized_weight

    @property
    def _normalized_weights(self) -> torch.Tensor:
        return self.normalized_weights

    @property
    def coefs(self) -> torch.Tensor:
        """ Returns the coefficients of the linear operation of LUI, of shape (out_channels, 2*in_channels).
        """
        return torch.cat([layer.positive_weight for layer in self.luis], axis=0)
