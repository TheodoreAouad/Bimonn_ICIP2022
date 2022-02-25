
from torch.utils.data.dataloader import DataLoader

from .compose_dataloader import ComposeDataloaders


def dataloader_resolution(df, dataset, dataset_args={}, shuffle=True, *args, **kwargs):
    """
    returns a data loader with same batch resolution.

    Args:
        df (pd.DataFrame): dataframe of dataset. Must have 'resolution' column.
        dataset (torch.utils.data.Dataset): dataset constructor
        shuffle (bool): shuffle or not dataset.

    Returns:
        ComposedIterators
    """

    data_loader = []
    if len(df) == 0:
        return ComposeDataloaders([DataLoader(data_loader, *args, **kwargs)])

    for resolution in df['resolution'].unique():
        data = dataset(df[df['resolution'] == resolution], **dataset_args)
        data_loader.append(DataLoader(data, *args, **kwargs))

    return ComposeDataloaders(data_loader, shuffle=shuffle)
