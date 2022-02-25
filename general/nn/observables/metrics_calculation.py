import pathlib
from os.path import join

import torch
from .observable import Observable
from ...utils import save_json


class CalculateAndLogMetrics(Observable):
    """
    class used to calculate and track metrics in the tensorboard
    """
    def __init__(self, metrics, keep_preds_for_epoch=True):
        self.metrics = metrics
        self.last_value = {k: 0 for k in metrics.keys()}
        self.keep_preds_for_epoch = keep_preds_for_epoch

        if self.keep_preds_for_epoch:
            self.all_preds = {'train': torch.tensor([]), 'val': torch.tensor([]), 'test': torch.tensor([])}
            self.all_targets = {'train': torch.tensor([]), 'val': torch.tensor([]), 'test': torch.tensor([])}
        self.tb_steps = {metric: {"train": 0, "val": 0, "test": 0} for metric in self.metrics.keys()}

        self._hp_metrics = dict(
            **{f"metrics_{batch_or_epoch}/{metric_name}_{state}": -1
               for batch_or_epoch in ['batch', 'epoch']
               for metric_name in self.metrics.keys()
               for state in ['train', 'val']
               }
        )

    def on_train_batch_end_with_preds(self, trainer, pl_module, outputs, batch, batch_idx, preds):
        inputs, targets = batch
        self._calculate_and_log_metrics(trainer, pl_module, targets, preds, state='train')

    def on_validation_batch_end_with_preds(self, trainer, pl_module, outputs, batch, batch_idx, preds):
        inputs, targets = batch
        self._calculate_and_log_metrics(trainer, pl_module, targets, preds, state='val')

    def on_test_batch_end_with_preds(self, trainer, pl_module, outputs, batch, batch_idx, preds):
        inputs, targets = batch
        self._calculate_and_log_metrics(trainer, pl_module, targets, preds, state='test')

    def _calculate_and_log_metrics(self, trainer, pl_module, targets, preds, state='train', batch_or_epoch='batch'):
        for metric_name in self.metrics:
            metric = self.metrics[metric_name](targets, preds)
            self.last_value[metric_name] = metric
            # pl_module.log(f"mean_metrics_{batch_or_epoch}/{metric_name}/{state}", metric)
            if batch_or_epoch == 'batch':
                step = self.tb_steps[metric_name][state]
            else:
                step = trainer.current_epoch

            trainer.logger.experiment.add_scalars(
                f"comparative/metrics_{batch_or_epoch}/{metric_name}", {state: metric}, step
            )

            trainer.logger.log_metrics(
                {f"metrics_{batch_or_epoch}/{metric_name}_{state}": metric}, step
            )

            if batch_or_epoch == 'batch':
                self.tb_steps[metric_name][state] = step + 1

            # f"metrics_multi_label_{batch_or_epoch}/{metric_name}/{state}", {f'label_{name_label}': metric}, step
            # trainer.logger.experiment.add_scalars(metric_name, {f'{metric_name}_{state}': metric})

    def on_train_epoch_end(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', unused: 'Optional' = None
    ):
        if self.keep_preds_for_epoch:
            self._calculate_and_log_metrics(trainer, pl_module, self.all_targets['train'], self.all_preds['train'], state='train', batch_or_epoch='epoch')
            self.all_preds['train'] = torch.tensor([])
            self.all_targets['train'] = torch.tensor([])

    def on_validation_epoch_end(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'
    ):
        if self.keep_preds_for_epoch:
            self._calculate_and_log_metrics(trainer, pl_module, self.all_targets['val'], self.all_preds['val'], state='val', batch_or_epoch='epoch')
            self.all_preds['val'] = torch.tensor([])
            self.all_targets['val'] = torch.tensor([])

    def on_test_epoch_end(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'
    ):
        if self.keep_preds_for_epoch:
            self._calculate_and_log_metrics(trainer, pl_module, self.all_targets['test'], self.all_preds['test'], state='test', batch_or_epoch='epoch')
            self.all_preds['test'] = torch.tensor([])
            self.all_targets['test'] = torch.tensor([])

    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
        save_json({k: str(v) for k, v in self.last_value.items()}, join(final_dir, "metrics.json"))
        return self.last_value
