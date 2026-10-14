
import os.path as osp
from collections import OrderedDict

import torch


def GetModelModule(model):
    if hasattr(model, "module"):
        return model.module
    return model


def ExtractCheckpointStateDict(checkpoint, filename):
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        raise RuntimeError("No state_dict found in checkpoint file {}".format(filename))

    if len(state_dict) > 0 and list(state_dict.keys())[0].startswith("module."):
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    return state_dict


def TryExpandInputChannelWeight(source_weight, target_weight):
    if source_weight.ndim < 2 or source_weight.ndim != target_weight.ndim:
        return None

    diff_dim_list = [
        dim_idx
        for dim_idx, (source_dim, target_dim) in enumerate(zip(source_weight.shape, target_weight.shape))
        if source_dim != target_dim
    ]
    if len(diff_dim_list) != 1:
        return None

    channel_dim = diff_dim_list[0]
    if source_weight.shape[channel_dim] != 3 or target_weight.shape[channel_dim] != 4:
        return None

    expanded_weight = source_weight.new_zeros(target_weight.shape)
    copy_slice = [slice(None)] * source_weight.ndim
    copy_slice[channel_dim] = slice(0, 3)
    expanded_weight[tuple(copy_slice)] = source_weight
    return expanded_weight


def BuildCompatibleStateDict(source_state_dict, target_state_dict):
    matched_state_dict = {}
    expanded_input_keys = []
    skipped_unexpected_keys = []
    skipped_shape_keys = []

    for key, value in source_state_dict.items():
        if key not in target_state_dict:
            skipped_unexpected_keys.append(key)
            continue

        target_value = target_state_dict[key]
        if target_value.shape == value.shape:
            matched_state_dict[key] = value
            continue

        if key.startswith("conv_input"):
            expanded_value = TryExpandInputChannelWeight(value, target_value)
            if expanded_value is not None:
                matched_state_dict[key] = expanded_value
                expanded_input_keys.append(key)
                continue

        skipped_shape_keys.append((key, tuple(value.shape), tuple(target_value.shape)))

    return matched_state_dict, expanded_input_keys, skipped_unexpected_keys, skipped_shape_keys


def LoadCompatibleModelWeights(model, filename, map_location=None):
    if not osp.isfile(filename):
        raise IOError("{} is not a checkpoint file".format(filename))

    try:
        checkpoint = torch.load(filename, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(filename, map_location=map_location)

    source_state_dict = ExtractCheckpointStateDict(checkpoint, filename)
    model_module = GetModelModule(model)
    target_state_dict = model_module.state_dict()
    matched_state_dict, expanded_input_keys, skipped_unexpected_keys, skipped_shape_keys = BuildCompatibleStateDict(
        source_state_dict,
        target_state_dict,
    )

    if len(matched_state_dict) == 0:
        raise RuntimeError("No matching parameters were loaded from checkpoint {}".format(filename))

    target_state_dict.update(matched_state_dict)
    model_module.load_state_dict(target_state_dict, strict=True)

    load_info = {
        "matched": len(matched_state_dict),
        "target_uninitialized": len(target_state_dict) - len(matched_state_dict),
        "unexpected": len(skipped_unexpected_keys),
        "shape_mismatch": len(skipped_shape_keys),
        "expanded_input_keys": expanded_input_keys,
        "skipped_shape_keys": skipped_shape_keys,
    }
    return checkpoint, load_info


def load_checkpoint(model, filename, map_location=None, strict=False):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Either a filepath or URL or modelzoo://xxxxxxx.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    # load checkpoint from modelzoo or file or url
    if not osp.isfile(filename):
        raise IOError("{} is not a checkpoint file".format(filename))
    
    try:
        checkpoint = torch.load(filename, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(filename, map_location=map_location)
    state_dict = ExtractCheckpointStateDict(checkpoint, filename)
    # load state_dict
    GetModelModule(model).load_state_dict(state_dict, strict=strict)
    return checkpoint


def weights_to_cpu(state_dict):
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.

    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu


def save_checkpoint(model, filename, optimizer=None, scheduler=None, meta=None):
    """Save checkpoint to file.

    The checkpoint will have 3 fields: ``meta``, ``state_dict`` and
    ``optimizer``. By default ``meta`` will contain version and time info.

    Args:
        model (Module): Module whose params are to be saved.
        filename (str): Checkpoint filename.
        optimizer (:obj:`Optimizer`, optional): Optimizer to be saved.
        meta (dict, optional): Metadata to be saved in checkpoint.
    """
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError("meta must be a dict or None, but got {}".format(type(meta)))

    if hasattr(model, "module"):
        model = model.module

    checkpoint = {"meta": meta, "state_dict": weights_to_cpu(model.state_dict())}
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()

    torch.save(checkpoint, filename)
