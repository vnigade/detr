import torch.backends.cudnn as cudnn
import torch
import numpy as np
import random
from collections import OrderedDict


def set_deterministic_behaviour(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.enabled = False
    cudnn.deterministic = True
    cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def print_summary(model, data_shape=(3, 800, 1000)):
    # summary(model, data_shape) # @TODO: This does not work because of the output shape in the detection output.

    # Just print paramters with their name
    print("Printing trainable parameters")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)


def load_checkpoint(path: str):
    """
    @TODO: This function is at many places in the main files. Make it clean.
    """
    if path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(path, map_location='cpu')
    return checkpoint


def load_ofa_state(model_state_dict, ofa_checkpoint_state, key_split_idx=2):
    """
    @TODO:
    Merging state dicts with different checkpoints is also in other main files, so it's also a duplicate. WTH, Make it clean.
    """
    _BACKBONE_LAYER_PREFIX = "backbone"

    new_state_dict = OrderedDict()
    for model_key, model_value in model_state_dict.items():
        # Keep other keys intact
        if not model_key.startswith(_BACKBONE_LAYER_PREFIX):
            new_state_dict[model_key] = model_value
            continue

        common_key = model_key.split(".", key_split_idx)[key_split_idx]
        # Change only the backbone
        for ckpt_key, ckpt_value in ofa_checkpoint_state.items():
            if common_key in ckpt_key and model_key not in new_state_dict:
                new_state_dict[model_key] = ckpt_value

        assert model_key in new_state_dict, f"Could not load model key {model_key}"
        assert new_state_dict[model_key].shape == model_value.shape, f"Shape does not match {model_key}"
    return new_state_dict


def load_partial_state(model_state_dict, chkpt_path: str, layer_prefix: str, key_split_idx=2):
    checkpoint = load_checkpoint(chkpt_path)
    checkpoint = checkpoint["model"]

    new_state_dict = OrderedDict()
    for model_key, model_value in model_state_dict.items():
        if not model_key.startswith(layer_prefix):
            continue
        common_key = model_key.split(".", key_split_idx)[key_split_idx]

        for ckpt_key, ckpt_value in checkpoint.items():
            if common_key in ckpt_key and model_key not in new_state_dict:
                # @TODO: There could be conflict between multiple common keys.
                # We use order to resolve the conflict.
                new_state_dict[model_key] = ckpt_value
        assert model_key in new_state_dict, f"Could not load model key {model_key}"
        print(f"shape of parameters {model_key}: {new_state_dict[model_key].shape}, {model_value.shape}")
        assert new_state_dict[model_key].shape == model_value.shape, f"Shape does not match {model_key}"

    return new_state_dict
