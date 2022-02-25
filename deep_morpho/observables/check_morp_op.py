import pathlib
from os.path import join

from typing import Any

from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytorch_lightning as pl
import torch.nn as nn
import matplotlib.pyplot as plt

from .observable_layers import ObservableLayersChans
from general.nn.observables import Observable

from ..models import BiSE


class ShowSelemAlmostBinary(Observable):

    def __init__(self, freq=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = freq
        self.freq_idx = 0
        self.last_selem_and_op = {}

    def on_train_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        dataloader_idx: int,
    ):
        if self.freq_idx % self.freq == 0:
            selems, operations = pl_module.model.get_bise_selems()
            for layer_idx, layer in enumerate(pl_module.model.layers):
                if not isinstance(layer, BiSE):
                    # fig = self.default_figure("Not BiSE")
                    continue

                elif selems[layer_idx] is None:
                    continue
                    # fig = self.default_figure("No operation found.")

                else:
                    fig = self.selem_fig(selems[layer_idx], operations[layer_idx])

                trainer.logger.experiment.add_figure(f"learned_selem/almost_binary_{layer_idx}", fig, trainer.global_step)
                self.last_selem_and_op[layer_idx] = (selems[layer_idx], operations[layer_idx])
        self.freq_idx += 1

    @staticmethod
    def default_figure(text):
        fig = plt.figure(figsize=(5, 5))
        plt.text(2, 2, text, horizontalalignment="center")
        return fig

    @staticmethod
    def selem_fig(selem, operation):
        fig = plt.figure(figsize=(3, 3))
        plt.imshow(selem, interpolation="nearest")
        plt.title(operation)
        return fig

    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)

        saved = []

        for layer_idx, (selem, operation) in self.last_selem_and_op.items():
            fig = self.selem_fig(selem, operation)
            fig.savefig(join(final_dir, f"layer_{layer_idx}.png"))
            saved.append(fig)

        return saved


class ShowSelemBinary(ObservableLayersChans):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_selem_and_op = {}

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
        selem, operation = layer.bises[chan_input].find_selem_and_operation_chan(chan_output, v1=0, v2=1)
        if selem is None:
            return

        fig = self.selem_fig(selem, operation)
        trainer.logger.experiment.add_figure(f"learned_selem/binary/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}", fig, trainer.global_step)
        self.last_selem_and_op[(layer_idx, chan_input, chan_output)] = (selem, operation)


    @staticmethod
    def selem_fig(selem, operation):
        fig = plt.figure(figsize=(3, 3))
        plt.imshow(selem, interpolation="nearest", vmin=0, vmax=1)
        plt.title(operation)
        return fig


    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)

        saved = []

        for (layer_idx, chan_input, chan_output), (selem, operation) in self.last_selem_and_op.items():
            fig = self.selem_fig(selem, operation)
            fig.savefig(join(final_dir, f"layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}.png"))
            saved.append(fig)

        return saved


class ShowLUISetBinary(ObservableLayersChans):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_set_and_op = {}

    def on_train_batch_end_layers_chan_output(
        self,
        trainer='pl.Trainer',
        pl_module='pl.LightningModule',
        outputs="STEP_OUTPUT",
        batch="Any",
        batch_idx=int,
        dataloader_idx=int,
        layer="nn.Module",
        layer_idx=int,
        chan_output=int,
    ):
        C, operation = layer.luis[chan_output].find_set_and_operation_chan(0, v1=None, v2=None)
        if C is None:
            return

        fig = self.set_fig(C, operation)
        trainer.logger.experiment.add_figure(f"learned_set_lui/binary/layer_{layer_idx}_chout_{chan_output}", fig, trainer.global_step)
        self.last_set_and_op[(layer_idx, chan_output)] = (C, operation)


    @staticmethod
    def set_fig(C, operation):
        fig = plt.figure(figsize=(3, 3))
        plt.imshow(C[:, None].astype(int), interpolation="nearest", vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks(range(len(C)))
        plt.title(operation)
        return fig


    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)

        saved = []

        for (layer_idx, chan_output), (C, operation) in self.last_set_and_op.items():
            fig = self.set_fig(C, operation)
            fig.savefig(join(final_dir, f"layer_{layer_idx}_chout_{chan_output}.png"))
            saved.append(fig)

        return saved
