from general.nn.loss import DiceLoss

from .masked_loss import LossMaskedBorder


class MaskedDiceLoss(LossMaskedBorder):

    def __init__(self, border, *args, **kwargs):
        super().__init__(loss=DiceLoss(*args, **kwargs), border=border)
