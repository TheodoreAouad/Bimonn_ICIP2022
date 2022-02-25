from pytorch_lightning import LightningModule
from typing import Any, List, Optional, Callable, Dict
from ..observables.observable import Observable
from pytorch_lightning.utilities.types import EPOCH_OUTPUT


class ObsLightningModule(LightningModule):

    def __init__(self, observables=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observables: Optional[List[Observable]] = observables

    def training_step(self, batch: Any, batch_idx: int):
        outputs, preds = self.obs_training_step(batch, batch_idx)

        for obs in self.observables:
            obs.on_train_batch_end_with_preds(
                self.trainer,
                self.trainer.lightning_module,
                outputs,
                batch,
                batch_idx,
                preds
            )

        return outputs

    def validation_step(self, batch: Any, batch_idx: int):
        outputs, preds = self.obs_validation_step(batch, batch_idx)

        for obs in self.observables:
            obs.on_validation_batch_end_with_preds(
                self.trainer,
                self.trainer.lightning_module,
                outputs,
                batch,
                batch_idx,
                preds
            )
        return outputs

    def test_step(self, batch: Any, batch_idx: int):
        outputs, preds = self.obs_test_step(batch, batch_idx)

        for obs in self.observables:
            obs.on_test_batch_end_with_preds(
                self.trainer,
                self.trainer.lightning_module,
                outputs,
                batch,
                batch_idx,
                preds
            )

        return outputs

    def test_epoch_end(self, outputs: EPOCH_OUTPUT):
        self.obs_test_epoch_end(outputs)
        for obs in self.observables:
            obs.on_test_epoch_end(self.trainer, self.trainer.lightning_module)

    def obs_training_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError

    def obs_validation_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError

    def obs_test_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError

    def obs_test_epoch_end(self, outputs: EPOCH_OUTPUT):
        pass


class NetLightning(ObsLightningModule):
    def __init__(
            self,
            model: "nn.Module",
            learning_rate: float,
            loss: Callable,
            optimizer: Callable,
            optimizer_args: Dict = {},
            observables: Optional[List[Observable]] = [],
    ):

        super().__init__(observables)
        self.model = model
        self.learning_rate = learning_rate
        self.loss = loss
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        # self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.learning_rate, **self.optimizer_args)

    def obs_training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.forward(x)

        loss = self.loss(predictions, y)
        self.log('loss/train_loss', loss)

        return {'loss': loss}, predictions

    def obs_validation_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.forward(x)

        loss = self.loss(predictions, y)
        self.log('loss/val_loss', loss)

        return {'val_loss': loss}, predictions

    def obs_test_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.forward(x)

        loss = self.loss(predictions, y)
        self.log('loss/test_loss', loss)

        return {'test_loss': loss}, predictions
