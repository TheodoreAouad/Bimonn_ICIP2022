import torch.nn as nn
import torch.nn.functional as F
import torch


class Unfolder(nn.Module):

    def __init__(self, kernel_size, padding=0, padding_value=0, device='cpu'):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = (padding, padding, padding, padding) if not isinstance(padding, tuple) else padding
        self.padding_value = padding_value
        self._device_tensor = nn.Parameter(torch.FloatTensor(device=device))

        self._right_operator = {}
        self._left_operator = {}


    @property
    def device(self):
        return self._device_tensor.device

    @property
    def ndim(self):
        return len(self.kernel_size)

    def unfold(self, x):
        x = F.pad(x, self.padding, value=self.padding_value)
        return self.get_left_operator(x.shape[-self.ndim:]) @ x @ self.get_right_operator(x.shape[-self.ndim:])

    @staticmethod
    def create_right_operator(size, k, device):
        right_operator = torch.zeros((size[0], k * (size[1] - k+1)), device=device)
        for i in range(right_operator.shape[0] - k + 1):
            right_operator[i:i+k, k*i:k*(i+1)] = torch.eye(k)
        return right_operator

    def get_right_operator(self, size):
        if size not in self._right_operator.keys():
            self._right_operator[size] = self.create_right_operator(size, self.kernel_size[1], device=self.device)
            setattr(self, '_right_operator_' + self.add_size_string(size), self._right_operator[size])
        return self._right_operator[size]

    def get_left_operator(self, size):
        if size not in self._left_operator.keys():
            self._left_operator[size] = self.create_right_operator(size[::-1], self.kernel_size[0], device=self.device).T
            setattr(self, '_left_operator_' + self.add_size_string(size), self._left_operator[size])
        return self._left_operator[size]

    def __call__(self, x):
        return self.unfold(x)

    @staticmethod
    def add_size_string(size, sep="x"):
        return sep.join([f'{s}' for s in size])
