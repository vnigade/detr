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


def compare_exit_outputs(args, model, data_loader, criterion_dict, base_ds, device):
    """ Function to compare outputs of an exit branch. 
    This compares the loss value defined in the DETR paper.
    """
    _DIFFICULTY_THRESHOLDS = {0: 1.0, 1: 2.0}

    def compute_loss(output, targets, exit_idx):
        with torch.no_grad():
            loss_dict = criterion_dict[exit_idx](output, targets)
            weight_dict = criterion_dict[exit_idx].weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        return losses

    def pred_to_target(output, orig_targets):
        """ Function to convert prediction from an exit branch to a format similar to Target, 
        as expected by loss functions in DETR
        """
        _NO_OBJECT_LABEL = 91
        targets = []
        for orig_target in orig_targets:  # for every batch target
            target = {}
            labels = torch.argmax(output["pred_logits"], axis=-1).squeeze(0)
            # remove no_object labels (value 91 for coco dataset) and boxes
            valid_indices = (labels != _NO_OBJECT_LABEL).nonzero(as_tuple=True)[0]
            target["labels"] = labels[valid_indices]
            target["boxes"] = torch.index_select(output["pred_boxes"], 1, valid_indices).squeeze(0)
            target["image_id"] = orig_target["image_id"]
            target["orig_size"] = orig_target["orig_size"]
            target["size"] = orig_target["size"]
            targets.append(target)
            min, max = orig_targets[0]["labels"].min(), orig_targets[0]["labels"].max()
            print(f"min, max from orig targets {min}, {max}")
        return targets

    output_dir = Path(args.output_dir)
    model.eval()
    exit_stats = np.zeros(args.num_exits)
    iou_types = tuple(['bbox'])
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    postprocessors = {'bbox': PostProcess()}

    difficulty_dataset_f = open(output_dir / "exit0_labels.csv", "w")

    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        exit_idx, out_exit_0 = model(samples, exit_choice=0)
        assert exit_idx == 0

        exit_idx, out_exit_1 = model(samples, exit_choice=1)
        assert exit_idx == 1

        # Now try to match outputs of every exit with the main exit
        # @TODO: Can we use the same loss function?
        losses_0 = compute_loss(out_exit_0, targets, exit_idx=0)
        losses_1 = compute_loss(out_exit_1, targets, exit_idx=1)

        # This block compares the output an exit branch with the main branch.
        # target_exit_1 = pred_to_target(out_exit_1, targets)
        # losses_soft_0 = compute_loss(out_exit_0, target_exit_1, exit_idx=0)
        # losses_soft_1 = compute_loss(out_exit_1, target_exit_1, exit_idx=1)

        print(f"Exit losses: exit_0 = {losses_0}, exit_1 = {losses_1}")

        if losses_0 <= _DIFFICULTY_THRESHOLDS[0]:
            outputs = out_exit_0
            exit_stats[0] += 1
            difficulty_dataset_f.write(f"{targets[0]['image_id'].cpu().numpy()[0]} 1\n")
        else:
            outputs = out_exit_1
            exit_stats[1] += 1
            difficulty_dataset_f.write(f"{targets[0]['image_id'].cpu().numpy()[0]} 0\n")

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
    utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

    difficulty_dataset_f.close()


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
    model, criterion_dict, _ = build_sparsee_model(args)
    model.to(device)
    print_summary(model)

    # Load a checkpoint
    checkpoint = load_checkpoint(args.checkpoint)
    model.load_state_dict(checkpoint["model"], strict=True)

    # Create COCO dataloader
    dataset_val = build_dataset(image_set='train_exit_condition', args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    base_ds = get_coco_api_from_dataset(dataset_val)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # Evaluate a model
    # evaluate(args, model, data_loader_val, base_ds, device)
    # print("Evaluated SparsEE DETR")

    # Compare exit outputs using loss criterion
    compare_exit_outputs(args, model, data_loader_val, criterion_dict, base_ds, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate SparsEE_DETR model', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
