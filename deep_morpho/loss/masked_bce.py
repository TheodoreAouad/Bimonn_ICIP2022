from torch.nn import BCELoss

from .masked_loss import LossMaskedBorder



class MaskedBCELoss(LossMaskedBorder):

    def __init__(self, border, *args, **kwargs):
        super().__init__(loss=BCELoss(*args, **kwargs), border=border)
