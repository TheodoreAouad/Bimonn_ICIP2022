import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.eps = 1e-6

    def forward(self, ypred, ytrue):
        ypred = ypred.squeeze()
        ytrue = ytrue.squeeze()

        if ypred.ndim == 2:
            ypred = ypred.unsqueeze(0)
        if ytrue.ndim == 2:
            ytrue = ytrue.unsqueeze(0)

        assert len(ypred.shape) == 3
        assert len(ytrue.shape) == 3

        dice_score = 2 * (ytrue * ypred).sum(1) / ((ytrue ** 2).sum(1) + (ypred ** 2).sum(1) + self.eps)

        return self.reduction_fn(1 - dice_score)

    def reduction_fn(self, x):
        if isinstance(self.reduction, str):
            if self.reduction == "sum":
                return x.sum()
            if self.reduction == "mean":
                return x.mean()
            if self.reduction == "none":
                return x

        return self.reduction(x)
