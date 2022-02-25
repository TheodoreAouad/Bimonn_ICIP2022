from typing import Any

from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytorch_lightning as pl

from general.nn.observables import Observable


class SaveLoss(Observable):

    def __init__(self, freq: int = 100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = freq
        self.idx = 0

    def on_train_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the train batch ends."""
        trainer.logger.experiment.add_scalars(
            "loss/train", {k: v for k, v in outputs.items() if 'loss' in k}, trainer.global_step
        )
