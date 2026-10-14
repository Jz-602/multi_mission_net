import numpy as np
import torch


def NumpyToTorch(item):
    if isinstance(item, dict):
        return {key: NumpyToTorch(value) for key, value in item.items()}
    if isinstance(item, np.ndarray):
        return torch.from_numpy(item)
    return item


def StackBatchItems(item_list):
    first_item = item_list[0]
    if isinstance(first_item, dict):
        return {key: StackBatchItems([item[key] for item in item_list]) for key in first_item.keys()}
    if torch.is_tensor(first_item):
        return torch.stack(item_list, dim=0)
    if isinstance(first_item, (int, float, str)):
        return item_list
    return item_list
