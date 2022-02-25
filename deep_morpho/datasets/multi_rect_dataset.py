import os
from os.path import join
import re

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from deep_morpho.morp_operations import SequentialMorpOperations
from general.utils import load_json, log_console


class InputOutputGeneratorDataset(Dataset):
    def __init__(
            self,
            random_gen_fn,
            random_gen_args,
            morp_operation: SequentialMorpOperations,
            device: str = "cpu",
            len_dataset: int = 1000,
    ):
        self.random_gen_fn = random_gen_fn
        self.random_gen_args = random_gen_args
        self.device = device
        self.len_dataset = len_dataset
        self.morp_fn = morp_operation


    def __getitem__(self, idx):
        input_ = self.random_gen_fn(**self.random_gen_args)
        target = self.morp_fn(input_).float()
        input_ = torch.tensor(input_).float()

        if input_.ndim == 2:
            input_ = input_.unsqueeze(-1)  # Must have at least one channel

        input_ = input_.permute(2, 0, 1)  # From numpy format (W, L, H) to torch format (H, W, L)
        target = target.permute(2, 0, 1)  # From numpy format (W, L, H) to torch format (H, W, L)

        return input_, target

    def __len__(self):
        return self.len_dataset

    @staticmethod
    def get_loader(batch_size, n_inputs, random_gen_fn, random_gen_args, morp_operation, device='cpu', **kwargs):
        return DataLoader(
            InputOutputGeneratorDataset(random_gen_fn, random_gen_args, morp_operation=morp_operation, device=device, len_dataset=n_inputs, ),
            batch_size=batch_size, **kwargs
        )


class MultiRectDataset(Dataset):
    def __init__(
            self,
            inputs_path: str,
            targets_path: str,
            do_load_in_ram: bool = False,
            verbose: bool = True,
            n_inputs: int = None,
            logger=None,
    ):
        self.inputs_path = inputs_path
        self.targets_path = targets_path
        self.do_load_in_ram = do_load_in_ram
        self.verbose = verbose
        self.logger = logger
        self.n_inputs = n_inputs

        self.all_inputs_name = sorted(os.listdir(inputs_path), key=lambda x: int(re.findall(r'\d+', x)[0]))
        if self.n_inputs is not None:
            self.all_inputs_name = self.all_inputs_name[:self.n_inputs]

        if self.do_load_in_ram:
            self.all_inputs = []
            self.all_targets = []

            if verbose:
                log_console('Loading data in RAM...', logger=self.logger)
            for inpt in self.get_verbose_iterator(self.all_inputs_name):
                self.all_inputs.append(np.load(join(inputs_path, inpt)))
                self.all_targets.append(np.load(join(targets_path, inpt)))


    def get_verbose_iterator(self, iterator):
        if self.verbose:
            return tqdm(iterator)
        return iterator

    def __getitem__(self, idx):
        if self.do_load_in_ram:
            input_, target = self.all_inputs[idx], self.all_targets[idx]
        else:
            img_name = self.all_inputs_name[idx]
            input_, target = np.load(join(self.inputs_path, img_name)), np.load(join(self.targets_path, img_name))
        target = torch.tensor(target).float()
        input_ = torch.tensor(input_).unsqueeze(0).float()

        return input_, target

    def __len__(self):
        return len(self.all_inputs_name)

    @staticmethod
    def get_loader(batch_size, dataset_path, do_load_in_ram, morp_operation, logger=None, n_inputs=None, **kwargs):
        inputs_path = join(dataset_path, 'images')
        metadata = load_json(join(dataset_path, 'metadata.json'))
        targets_path = metadata["seqs"][morp_operation.get_saved_key()]['path_target']
        return DataLoader(
            MultiRectDataset(inputs_path, targets_path, do_load_in_ram=do_load_in_ram, verbose=True, logger=logger, n_inputs=n_inputs),
            batch_size=batch_size, **kwargs
        )
