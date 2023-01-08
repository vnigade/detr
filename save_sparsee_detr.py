# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from collections import OrderedDict
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate
from models.sparsee_detr import _NAME_FMT_EXIT
import util.misc as utils
from models import build_sparsee_model


def print_summary(model, data_shape=(3, 800, 1000)):
    # Just print paramters with their name
    print("Printing trainable parameters")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)


def load_checkpoint(path: str):
    if path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(path, map_location='cpu')
    return checkpoint


def load_partial_state(model_state_dict, chkpt_path: str, layer_prefix: str):
    checkpoint = load_checkpoint(chkpt_path)
    checkpoint = checkpoint["model"]

    new_state_dict = OrderedDict()
    for model_key, model_value in model_state_dict.items():
        if not model_key.startswith(layer_prefix):
            continue
        common_key = model_key.split(".", 2)[2]

        for ckpt_key, ckpt_value in checkpoint.items():
            if common_key in ckpt_key and model_key not in new_state_dict:
                # @TODO: There could be conflict between multiple common keys.
                # We use order to resolve the conflict.
                new_state_dict[model_key] = ckpt_value
        assert model_key in new_state_dict, f"Could not load model key {model_key}"
        print(f"shape of parameters {model_key}: {new_state_dict[model_key].shape}, {model_value.shape}")
        assert new_state_dict[model_key].shape == model_value.shape, f"Shape does not match {model_key}"

    return new_state_dict


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int)
    # Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # Transformer
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Exits
    parser.add_argument('--num_exits', default=2, type=int,
                        help="Number of exits in the model")

    # Segmentation: Needed in build criterion function.
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=2, type=int)

    # Checkpoints
    parser.add_argument('--backbone_ckpt', type=str,
                        help="Checkpoint path used for the backbone stage")
    parser.add_argument('--exits_ckpt', action='append',
                        help="The list of checkpoint paths used for the exit branches")
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion_dict, postprocessors_dict = build_sparsee_model(args)
    model.to(device)
    print_summary(model)

    model_state_dict = {}
    # First load the backbone stages
    state_dict = load_partial_state(model.state_dict(), args.backbone_ckpt, layer_prefix="stages")
    model_state_dict = {**model_state_dict, **state_dict}

    # Now load all the exits
    for exit_idx in range(args.num_exits):
        state_dict = load_partial_state(model.state_dict(),
                                        args.exits_ckpt[exit_idx],
                                        layer_prefix=f"exits.{_NAME_FMT_EXIT.format(exit_idx)}")
        model_state_dict = {**model_state_dict, **state_dict}

    # Try to load model parameters from the model state dictionary
    model.load_state_dict(model_state_dict, strict=True)

    output_dir = Path(args.output_dir)

    # Evaluate on coco dataset
    dataset_val = build_dataset(image_set='val', args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    base_ds = get_coco_api_from_dataset(dataset_val)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    test_stats, coco_evaluator = evaluate(model, criterion_dict[1], postprocessors_dict[1],
                                          data_loader_val, base_ds, device, args.output_dir)

    utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

    checkpoint_path = output_dir / 'checkpoint.pth'
    utils.save_on_master({
        'model': model.state_dict(),
        'optimizer': None,
        'lr_scheduler': None,
        'epoch': None,
        'args': args,
    }, checkpoint_path)

    print(f"Loaded and saved SparsEE-DETR at {checkpoint_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SparsEE_DETR load and save model from checkpoints', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
