from typing import Any

from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytorch_lightning as pl
import torch.nn as nn

from .observable_layers import ObservableLayersChans


class WeightsHistogramBiSE(ObservableLayersChans):

    def __init__(self, *args, freq: int = 100, **kwargs):
        super().__init__(*args, freq=freq, **kwargs)

    def on_train_batch_end_layers_chans(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        dataloader_idx: int,
        layer: "nn.Module",
        layer_idx: int,
        chan_input: int,
        chan_output: int,
    ):
        # if isinstance(layer, (BiSE, BiSEC, COBiSEC, COBiSE)):


        trainer.logger.experiment.add_histogram(
            f"weights_hist/Normalized/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}",
            layer._normalized_weight[chan_output, chan_input],
            trainer.global_step
        )
        trainer.logger.experiment.add_histogram(
            f"weights_hist/Raw/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}",
            layer.weight[chan_output, chan_input],
            trainer.global_step
        )
