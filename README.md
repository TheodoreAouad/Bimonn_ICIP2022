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
