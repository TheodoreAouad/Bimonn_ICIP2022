from time import time
import os
from os.path import join
import pathlib

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from deep_morpho.datasets.mnist_dataset import MnistMorphoDataset


from deep_morpho.datasets.multi_rect_dataset import InputOutputGeneratorDataset, MultiRectDataset
import deep_morpho.observables as obs
import biblio_comparison.observables as biblio_obs
from general.nn.observables import CalculateAndLogMetrics
from general.utils import format_time, log_console, create_logger, save_yaml
from deep_morpho.metrics import masked_dice
from biblio_comparison.args import all_args
from general.code_saver import CodeSaver
from biblio_comparison.lightning_models import LightningLMorph, LightningSMorph, LightningAdaptativeMorphologicalLayer

model_dict = {
    "adaptative": LightningAdaptativeMorphologicalLayer,
    "smorph": LightningSMorph,
    "lmorph": LightningLMorph,
}


def get_dataloader(args):

    if args['dataset_type'] == 'diskorect':
        trainloader = InputOutputGeneratorDataset.get_loader(
            batch_size=args['batch_size'],
            n_inputs=args['n_inputs'],
            random_gen_fn=args['random_gen_fn'],
            random_gen_args=args['random_gen_args'],
            morp_operation=args['morp_operation'],
            device=device,
            num_workers=args['num_workers']
        )
        valloader = None
        testloader = None


    elif args['dataset_type'] == "mnist":
        prop_train, prop_val, prop_test = args['train_test_split']
        trainloader, valloader, testloader = MnistMorphoDataset.get_train_val_test_loader(
            n_inputs_train=int(prop_train * args['n_inputs']),
            n_inputs_val=int(prop_val * args['n_inputs']),
            n_inputs_test=int(prop_test * args['n_inputs']),
            batch_size=args['batch_size'],
            morp_operation=args['morp_operation'],
            preprocessing=args['preprocessing'],
            shuffle=True,
            num_workers=args['num_workers'],
            **args['mnist_args']
        )

    return trainloader, valloader, testloader


def main(args, logger):

    trainloader, valloader, _ = get_dataloader(args)
    metrics = {'dice': lambda y_true, y_pred: masked_dice(y_true, y_pred, border=(args['kernel_size'] // 2, args['kernel_size'] // 2), threshold=.5).mean()}

    observables_dict = {
        "SaveLoss": obs.SaveLoss(),
        "CalculateAndLogMetric": CalculateAndLogMetrics(
            metrics=metrics,
            keep_preds_for_epoch=False,
        ),
        "PlotPreds": obs.PlotPreds(freq={'train': args['freq_imgs'], 'val': 2}),
        "InputAsPredMetric": obs.InputAsPredMetric(metrics),
        "CountInputs": obs.CountInputs(),
        "PlotParameters": biblio_obs.PlotParameters(freq=1),
        "PlotWeights": biblio_obs.PlotWeights(freq=args['freq_imgs']),
        # "PlotLUIParametersBiSEL": obs.PlotLUIParametersBiSEL(),
        # "WeightsHistogramBiSE": obs.WeightsHistogramBiSE(freq=args['freq_imgs']),
        # "PlotGradientBise": obs.PlotGradientBise(freq=args['freq_imgs']),
        "ConvergenceMetrics": obs.ConvergenceMetrics(metrics),

    }

    observables = list(observables_dict.values())

    model = model_dict[args['model']](
        model_args={
            "kernel_size": [(args['kernel_size'], args['kernel_size']) for _ in range(args['n_atoms'])],
        },
        learning_rate=args['learning_rate'],
        loss=args['loss'],
        optimizer=args['optimizer'],
        observables=observables,
    )

    model.to(device)

    logger.experiment.add_graph(model, torch.ones(1, 1, 50, 50).to(device))

    pathlib.Path(join(logger.log_dir, "target_SE")).mkdir(exist_ok=True, parents=True)
    figs_selems = args['morp_operation'].plot_selem_arrays()
    for (layer_idx, chan_input, chan_output), fig in figs_selems.items():
        fig.savefig(join(logger.log_dir, "target_SE", f"target_SE_l_{layer_idx}_chin_{chan_input}_chout_{chan_output}.png"))
        logger.experiment.add_figure(f"target_SE/target_SE_l_{layer_idx}_chin_{chan_input}_chout_{chan_output}", fig)

    pathlib.Path(join(logger.log_dir, "morp_operations")).mkdir(exist_ok=True, parents=True)
    fig_morp_operation = args['morp_operation'].vizualise().fig
    fig_morp_operation.savefig(join(logger.log_dir, "morp_operations", "morp_operations.png"))
    logger.experiment.add_figure("target_operations/morp_operations", fig_morp_operation)

    trainer = Trainer(
        max_epochs=args['n_epochs'],
        gpus=1 if torch.cuda.is_available() else 0,
        logger=logger,
        # progress_bar_refresh_rate=10,
        callbacks=observables.copy(),
        log_every_n_steps=10,
    )

    trainer.fit(model, trainloader, valloader)

    for observable in observables:
        observable.save(join(trainer.log_dir, 'observables'))



if __name__ == '__main__':
    start_all = time()

    code_saver = CodeSaver(
        src_path=os.getcwd(),
        temporary_path="results",
        ignore_patterns=("*__pycache__*", "*results*", "data", "*.ipynb", '.git', 'ssm'),
    )

    code_saver.save_in_temporary_file()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    bugged = []
    results = []

    for args_idx, args in enumerate(all_args):

        name = join(args["experiment_name"], args['experiment_subname'])

        logger = TensorBoardLogger("results", name=name, default_hp_metric=False)
        code_saver.save_in_final_file(join(logger.log_dir, 'code'))
        save_yaml(args, join(logger.log_dir, 'args.yaml'))

        console_logger = create_logger(
            f'args_{args_idx}', all_logs_path=join(logger.log_dir, 'all_logs.log'), error_path=join(logger.log_dir, 'error_logs.log')
        )

        log_console('Device: {}'.format(device), logger=console_logger)
        log_console('==================', logger=console_logger)
        log_console('==================', logger=console_logger)
        log_console(f'Args number {args_idx + 1} / {len(all_args)}', logger=console_logger)
        log_console('Time since beginning: {} '.format(format_time(time() - start_all)), logger=console_logger)
        log_console(logger.log_dir, logger=console_logger)
        log_console(args['morp_operation'], logger.log_dir, logger=console_logger)
        results.append(main(args, logger))
        # try:
        #     main(args, logger)
        # except Exception:
        #     console_logger.exception(
        #         f'Args nb {args_idx + 1} / {len(all_args)} failed : ')
        #     bugged.append(args_idx+1)

    code_saver.delete_temporary_file()


    log_console(f'{len(bugged)} Args Bugged: ', bugged, logger=console_logger)
    log_console(f'{len(all_args)} args done in {format_time(time() - start_all)} ', logger=console_logger)
