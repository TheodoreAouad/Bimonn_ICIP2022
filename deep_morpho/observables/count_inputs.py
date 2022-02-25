import pathlib
from os.path import join
from typing import Any

from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytorch_lightning as pl

from general.nn.observables import Observable


class CountInputs(Observable):

    def __init__(self):
        super().__init__()
        self.n_inputs = 0

    def on_train_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: 'STEP_OUTPUT',
        batch: 'Any',
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.n_inputs += len(batch[0])
        trainer.logger.experiment.add_scalar("n_inputs", self.n_inputs, trainer.global_step)

    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)

        with open(join(final_dir, str(self.n_inputs)), "w"):
            pass

        return self.n_inputs
