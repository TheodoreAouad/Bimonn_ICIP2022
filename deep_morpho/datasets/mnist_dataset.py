from os.path import join
from typing import Tuple, Any
import cv2

from torchvision.datasets import MNIST
from torch.utils.data.dataloader import DataLoader
import torch

from deep_morpho.morp_operations import ParallelMorpOperations


ROOT_MNIST_DIR = join('/', 'hdd', 'datasets', 'MNIST')


class MnistMorphoDataset(MNIST):

    def __init__(
        self,
        morp_operation: ParallelMorpOperations,
        n_inputs: int = "all",
        threshold: float = .5,
        size=(50, 50),
        first_idx: int = 0,
        preprocessing=None,
        root: str = ROOT_MNIST_DIR,
        train: bool = True,
        invert_input_proba: bool = 0,
        **kwargs,
    ) -> None:
        super().__init__(root, train, *kwargs)
        self.morp_operation = morp_operation
        self.threshold = threshold
        self.preprocessing = preprocessing
        self.size = size
        self.invert_input_proba = invert_input_proba

        if n_inputs != "all":
            self.data = self.data[first_idx:n_inputs+first_idx]


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        input_ = (
            cv2.resize(self.data[index].numpy(), (self.size[1], self.size[0]), interpolation=cv2.INTER_CUBIC)
            >= (self.threshold)
        )[..., None]

        if torch.rand(1) < self.invert_input_proba:
            input_ = 1 - input_

        # input_[..., 0] = set_borders_to(input_[..., 0], np.array(self.morp_operation.max_selem_shape[0]) // 2, value=0)

        target = self.morp_operation(input_).float()
        input_ = torch.tensor(input_).float()

        input_ = input_.permute(2, 0, 1)  # From numpy format (W, L, H) to torch format (H, W, L)
        target = target.permute(2, 0, 1)  # From numpy format (W, L, H) to torch format (H, W, L)

        # target = target != self.data['value_bg'].iloc[idx]

        if self.preprocessing is not None:
            input_ = self.preprocessing(input_)
            target = self.preprocessing(target)

        return input_.float(), target.float()


    @property
    def processed_folder(self) -> str:
        return join(self.root, 'processed')


    @staticmethod
    def get_loader(batch_size, n_inputs, morp_operation, train, first_idx=0, threshold=.5, size=(50, 50), invert_input_proba=0, preprocessing=None, **kwargs):
        if n_inputs == 0:
            return DataLoader([])
        return DataLoader(
            MnistMorphoDataset(
                morp_operation=morp_operation, n_inputs=n_inputs, first_idx=first_idx,
                train=train, threshold=threshold, preprocessing=preprocessing,
                size=size, invert_input_proba=invert_input_proba,
            ), batch_size=batch_size, **kwargs)

    @staticmethod
    def get_train_val_test_loader(n_inputs_train, n_inputs_val, n_inputs_test, *args, **kwargs):
        trainloader = MnistMorphoDataset.get_loader(first_idx=0, n_inputs=n_inputs_train, train=True, *args, **kwargs)
        valloader = MnistMorphoDataset.get_loader(first_idx=0, n_inputs=n_inputs_val, train=False, *args, **kwargs)
        testloader = MnistMorphoDataset.get_loader(first_idx=n_inputs_val, n_inputs=n_inputs_test, train=False, *args, **kwargs)
        return trainloader, valloader, testloader
