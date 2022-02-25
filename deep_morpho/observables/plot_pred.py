from typing import Any, Dict
import pathlib
from os.path import join
import random

from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from general.nn.observables import Observable


class PlotPreds(Observable):

    def __init__(self, freq: Dict = {"train": 100, "val": 10}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = freq
        self.idx = {"train": 0, "val": 0}
        self.saved_fig = {"train": None, "val": None}

    def on_validation_batch_end_with_preds(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        preds: Any
    ) -> None:
        if self.idx['val'] % self.freq['val'] == 0:
            idx = random.choice(range(len(batch[0])))
            img, target = batch[0][idx], batch[1][idx]
            pred = preds[idx]
            fig = self.plot_three(*[k.cpu().detach().numpy() for k in [img, pred, target]], title=f'val | epoch {trainer.current_epoch}')
            trainer.logger.experiment.add_figure("preds/val/input_pred_target", fig, self.idx['val'])
            self.saved_fig['val'] = fig

        self.idx['val'] += 1


    def on_train_batch_end_with_preds(
            self,
            trainer: 'pl.Trainer',
            pl_module: 'pl.LightningModule',
            outputs: 'STEP_OUTPUT',
            batch: 'Any',
            batch_idx: int,
            preds: 'Any',
    ) -> None:
        if self.idx['train'] % self.freq["train"] == 0:
            idx = random.choice(range(len(batch[0])))
            img, target = batch[0][idx], batch[1][idx]
            pred = preds[idx]
            fig = self.plot_three(*[k.cpu().detach().numpy() for k in [img, pred, target]], title='train')
            trainer.logger.experiment.add_figure("preds/train/input_pred_target", fig, trainer.global_step)
            self.saved_fig['train'] = fig

        self.idx['train'] += 1

    @staticmethod
    def plot_three(img, pred, target, title=''):
        ncols = max(img.shape[0], pred.shape[0])
        fig, axs = plt.subplots(3, ncols, figsize=(4 * ncols, 4 * 3), squeeze=False)
        fig.suptitle(title)

        for chan in range(img.shape[0]):
            axs[0, chan].imshow(img[chan], cmap='gray', vmin=0, vmax=1)
            axs[0, chan].set_title(f'input_{chan}')

        for chan in range(pred.shape[0]):
            axs[1, chan].imshow(pred[chan], cmap='gray', vmin=0, vmax=1)
            axs[1, chan].set_title(f'pred_{chan} vmin={pred[chan].min():.2} vmax={pred[chan].max():.2}')

        for chan in range(target.shape[0]):
            axs[2, chan].imshow(target[chan], cmap='gray', vmin=0, vmax=1)
            axs[2, chan].set_title(f'target_{chan}')

        return fig

    @staticmethod
    def plot_channels(img, pred, target):
        ncols = max(img.shape[-1], 2)
        fig, axs = plt.subplots(2, ncols, figsize=(ncols*7, 2*7))

        for chan in range(img.shape[-1]):
            axs[0, chan].imshow(img[..., chan], cmap='gray')
            axs[0, chan].set_title(f'input_{chan}')

        axs[1, 0].imshow(pred, cmap='gray', vmin=0, vmax=1)
        axs[1, 0].set_title('pred')

        axs[1, 1].imshow(target, cmap='gray')
        axs[1, 1].set_title('target')

        return fig

    def save(self, save_path: str):
        for state in ['train', 'val']:
            if self.saved_fig[state] is not None:
                final_dir = join(save_path, self.__class__.__name__)
                pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
                self.saved_fig[state].savefig(join(final_dir, f"input_pred_target_{state}.png"))

        return self.saved_fig
