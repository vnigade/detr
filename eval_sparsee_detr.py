# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import random
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import build_dataset, get_coco_api_from_dataset
from datasets.coco_eval import CocoEvaluator
from engine import evaluate
from models.detr import PostProcess
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


def get_args_parser():
    parser = argparse.ArgumentParser('SparsEE DETR', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int)

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
    parser.add_argument('--checkpoint', type=str,
                        help="Checkpoint path to a trained SparsEE DETR")

    return parser


def evaluate(args, model, data_loader, base_ds, device):
    model.eval()
    output_dir = Path(args.output_dir)

    exit_stats = np.zeros(args.num_exits)
    exec_time = []
    iou_types = tuple(['bbox'])
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    postprocessors = {'bbox': PostProcess()}

    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        start_time = time.time()
        exit_idx, outputs = model(samples)
        torch.cuda.synchronize()
        end_time = time.time()
        exec_time.append((end_time - start_time) * 1e3)

        exit_stats[exit_idx] += 1  # Keeping batch size 1

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    stats = {}
    stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()

    for exit_idx in range(args.num_exits):
        print(f"Fraction of samples exited at {exit_idx}: {exit_stats[exit_idx] / exit_stats.sum()}")
    print(f"Execution time: {np.array(exec_time).mean()}")
    utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")


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

    # Build model
    model, _, _ = build_sparsee_model(args)
    model.to(device)
    print_summary(model)

    # Load a checkpoint
    checkpoint = load_checkpoint(args.checkpoint)
    model.load_state_dict(checkpoint["model"], strict=True)

    # Create COCO dataloader
    dataset_val = build_dataset(image_set='val', args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    base_ds = get_coco_api_from_dataset(dataset_val)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # Evaluate a model
    evaluate(args, model, data_loader_val, base_ds, device)
    print("Evaluated SparsEE DETR")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SparsEE_DETR load and save model from checkpoints', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
