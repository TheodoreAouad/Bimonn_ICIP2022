from os.path import join
import itertools
import pathlib

import matplotlib.pyplot as plt

from deep_morpho.observables import ObservableLayers
from general.utils import max_min_norm, save_json


class PlotWeights(ObservableLayers):

    def __init__(self, *args, freq: int = 100, **kwargs):
        super().__init__(*args, freq=freq, **kwargs)
        self.last_weights = []

    def on_train_batch_end_layers(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        dataloader_idx: int,
        layer: "nn.Module",
        layer_idx: int,
    ):
        # if isinstance(layer, (BiSE, COBiSE, BiSEC, COBiSEC)):
        #     trainer.logger.experiment.add_figure(f"weights/Normalized_{layer_idx}", self.get_figure_normalized_weights(
        #         layer._normalized_weight, layer.bias, layer.activation_P), trainer.global_step)
        # trainer.logger.experiment.add_figure(f"weights/Raw_{layer_idx}", self.get_figure_weights(layer.weight), trainer.global_step)

        weights = layer.weight
        trainer.logger.experiment.add_figure(
            f"weights/layer_{layer_idx}",
            self.get_figure_weights(weights, title=f'param={layer.P.item():.2f}'),
            trainer.global_step
        )

    def on_train_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        for layer_idx, layer in enumerate(pl_module.model.layers):
            self.last_weights.append(layer.weight)

    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
        for layer_idx, weight in enumerate(self.last_weights):
            fig = self.get_figure_weights(weight)
            fig.savefig(join(final_dir, f"layer_{layer_idx}.png"))

        return self.last_weights

    @staticmethod
    def get_figure_weights(weights, title=''):
        weights = weights.cpu().detach()
        weights_normed = max_min_norm(weights)
        figure = plt.figure(figsize=(8, 8))
        plt.title(title)
        plt.imshow(weights_normed, interpolation='nearest', cmap=plt.cm.gray)
        plt.colorbar()
        # plt.clim(0, 1)

        # Use white text if squares are dark; otherwise black.
        threshold = weights_normed.max() / 2.

        for i, j in itertools.product(range(weights.shape[0]), range(weights.shape[1])):
            color = "white" if weights_normed[i, j] < threshold else "black"
            plt.text(j, i, round(weights[i, j].item(), 2), horizontalalignment="center", color=color)

        plt.tight_layout()
        return figure


class PlotParameters(ObservableLayers):

    def __init__(self, *args, freq: int = 1, **kwargs):
        super().__init__(*args, freq=freq, **kwargs)
        self.last_params = {}

    def on_train_batch_end_layers(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        dataloader_idx: int,
        layer: "nn.Module",
        layer_idx: int,
    ):
        trainer.logger.log_metrics({f"params/P/layer_{layer_idx}": layer.P}, trainer.global_step)
        self.last_params[layer_idx] = layer.P.item()

    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
        save_json(self.last_params, join(final_dir, "params.json"))
