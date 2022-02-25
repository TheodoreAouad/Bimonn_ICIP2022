from torch.nn import MSELoss

from .masked_loss import LossMaskedBorder



class MaskedMSELoss(LossMaskedBorder):

    def __init__(self, border, *args, **kwargs):
        super().__init__(loss=MSELoss(*args, **kwargs), border=border)
