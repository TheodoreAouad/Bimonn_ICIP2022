# READ ME

This is the repository of the paper **Binary Morphological Neural Network**, submitted at [ICIP 2022](https://2022.ieeeicip.org/).

This repository allows to get the results of table 1 and table 2, as well as a some results on the $\mathcal{L} Morph$ and $\mathcal{S} Morph$ models introduced in [Going beyond p-convolutions to learn grayscale morphological operators](https://arxiv.org/abs/2102.10038).

# BiMoNN results

See the files inside `deep_morpho/saved_args`. There are:

- `args.py`: all the arguments to handle. Please check that the `num_workers` key is adapted to your number of CPU cores. Each argument is given in a list. Finally, a cartesian product of all keys will be aggregated, and the experiment will be launched with all these arguments.
- `args_morp_ops_diskorect.py`: handle which structuring element and which operation as target for the diskorect dataset.
- `args_morp_ops_mnist.py`: handle which structuring element and which operation as target for the mnist dataset.

The command to run is:
> python deep_morpho/train_net.py

The results will be saved in a `results/bimonn` folder inside the current working directory.

# Bibliography results

See the files inside `biblio_comparison`. There are:

- `args.py`: all the arguments to handle. Please check that the `num_workers` key is adapted to your number of CPU cores. Each argument is given in a list. Finally, a cartesian product of all keys will be aggregated, and the experiment will be launched with all these arguments.
- `args_morp_ops_diskorect.py`: handle which structuring element and which operation as target for the diskorect dataset.
- `args_morp_ops_mnist.py`: handle which structuring element and which operation as target for the mnist dataset.

The command to run is:
> python biblio_comparison/train_biblio.py

The results will be saved in a `results/biblio` folder inside the current working directory.

# Results

The results are saved inside the `results` folder. There are 3 ways to look at the results.

## Tensorboard

Tensorboard will be written and all metrics, loss and weights are displayed across the iterations. Then, you can just launch the tensorboard to look at the evolution of these values during training.

> tensorboard --logdir results/bimonn/diskorect/erosion/version_0


## Using Observables

Each time a training is done, a set of "observables" are saved. They are used for the tensorboard and to save final results. If the tensorboard folder of the experiment is `{TB_PATH}`, then the observables are inside `{TB_PATH}/observables`. You can take a look at each saved value. The most important ones are:

- the dice value, inside `CalculateAndLogMetrics/metrics.json`
- the weights, inside `PlotWeightsBiSE`
- the convergence values
    - for the structuring element: inside `ConvergenceBinary/convergence_step.json`
    - for the perfect metric (DICE = 1): inside `ConvergenceMetrics /convergence_step.json`

## Using HTML created file

This method only works for the `Bimonn` experiments, not the `biblio`. At the end of a batch of experiments, the results of the observables of each experiment are saved inside an html file and saved inside `exp/{NB_EXP}/bimonn_results.html`. The weights and metrics are supposed to be set, as well as the learned structuring elements for the activated BiSEs.

