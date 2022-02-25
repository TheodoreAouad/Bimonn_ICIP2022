import pathlib
from os.path import join

from general.nn.observables import Observable
from deep_morpho.viz.bimonn_viz import BimonnVizualiser


class PlotBimonn(Observable):

    def __init__(self, freq: int = 300, figsize=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = freq
        self.freq_idx = 0
        self.figsize = figsize

        self.last_figs = {}

    def on_train_batch_end_with_preds(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        preds: "Any",
    ) -> None:

        if self.freq_idx % self.freq == 0:

            vizualiser = BimonnVizualiser(pl_module.model, mode="weight")
            fig = vizualiser.get_fig(figsize=self.figsize)
            trainer.logger.experiment.add_figure("model/weight", fig, trainer.global_step)
            self.last_figs['weights'] = fig

            vizualiser = BimonnVizualiser(pl_module.model, mode="selem")
            fig = vizualiser.get_fig(figsize=self.figsize)
            trainer.logger.experiment.add_figure("model/selem", fig, trainer.global_step)
            self.last_figs['selems'] = fig

        self.freq_idx += 1


    def save(self, save_path: str):
        if len(self.last_figs) == 0:
            return

        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)

        for key, fig in self.last_figs.items():
            fig.savefig(join(save_path, f"learned_{key}.png"))

        return self.last_figs
