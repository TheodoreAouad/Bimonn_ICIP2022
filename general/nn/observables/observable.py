from typing import Any, Union, List, Tuple, Dict
import fnmatch

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT


class Observable(pl.callbacks.Callback):
    """
        Abstract base class for training, validation and testing hooks.
        Same logic as pytorch_lightning callbacks. But as of version 1.3.0 of PL,
        these methods do not take the predictions as arguments, so we have to recompute them.
        To avoid this, we create additional methods to catch them (see class MODULE2 #TODO)

    """

    def __init__(self, hp_metrics_mode: Union[str, List, Tuple] = 'all', ignore_hp_metrics: Tuple = (), *args, **kwargs):
        """

        Args:
            hp_metrics_mode (str, list, tuple): specify the hp_metrics that will be logged into the hparams of tensorboard.
            It can be 'all' in this case all the '_hp_metrics' will be logged, 'none' nothing will be logged or a list
            or a tuple with the specific hp_metrics.
            ignore_hp_metrics: specify the hp_metrics that will be ignored.
        """
        super().__init__(*args, **kwargs)
        self.hp_metrics_mode = hp_metrics_mode
        self.ignore_hp_metrics = ignore_hp_metrics
        self._hp_metrics = {}  # metrics to log into the hparams of tensorboard
        self.logger = None

    @property
    def hp_metrics(self) -> Dict[str, Any]:
        """
        Returns the hp_metrics that will be logged into the hparams of tensorboard.
        """
        if self.hp_metrics_mode == 'all':
            res = self._hp_metrics

        elif self.hp_metrics_mode == 'none':
            res = {}

        elif isinstance(self.hp_metrics_mode, (list, tuple)):
            res = {k: v for k, v in self._hp_metrics.items() if k in self.hp_metrics_mode}

        return {k: v for k, v in res.items() if not self.must_be_ignored(k)}


    def set_hp_metrics(self, hp_metrics: Dict[str, Any]):
        self._hp_metrics = hp_metrics


    def must_be_ignored(self, elt: str) -> bool:
        """
        Returns True if the element must be ignored.
        """
        for pattern in self.ignore_hp_metrics:
            if len(fnmatch.filter([elt], pattern)) != 0:
                return True
        return False


    def on_train_batch_end_with_preds(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
            preds: Any,
    ) -> None:
        pass

    def on_validation_batch_end_with_preds(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
            preds: Any,
    ) -> None:
        pass

    def on_test_batch_end_with_preds(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
            preds: Any,
    ) -> None:
        pass


    def save(self, savepath: str):
        pass
