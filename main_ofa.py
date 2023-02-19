# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import copy
import datetime
import json
import random
import time
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.backends.cudnn as cudnn

import datasets
from models.ofa_backbone import SUPPORTED_INPUT_SIZES, build_ofa_backbone, get_static_ofa, set_active_backbone
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model


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


def load_ofa_state(model_state_dict, ofa_checkpoint_state):
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

        common_key = model_key.split(".", 2)[2]
        # Change only the backbone
        for ckpt_key, ckpt_value in ofa_checkpoint_state.items():
            if common_key in ckpt_key and model_key not in new_state_dict:
                new_state_dict[model_key] = ckpt_value

        assert model_key in new_state_dict, f"Could not load model key {model_key}"
        assert new_state_dict[model_key].shape == model_value.shape, f"Shape does not match {model_key}"
    return new_state_dict


def build_ofa_detr(args, ofa_type, input_size):
    device = torch.device(args.device)

    # 1. Build static DETR for the input size
    ofa_detr_model, criterion, postprocessors = build_model(args)
    ofa_detr_model.to(device)

    # 2. First load the whole DETR pretrained which would load input_proj, transformer blocks and other modules instead of OFA backbone
    detr_checkpoint_state = load_checkpoint(args.detr_checkpoint)
    ofa_detr_model.load_state_dict(detr_checkpoint_state["model"], strict=False)

    # 3. Now get the pretrained checkpoints for the OFA component
    ofa_checkpoint_state = torch.load(args.ofa_checkpoint, map_location='cpu')["state_dict"]
    ofa_dynamic_model = build_ofa_backbone(ofa_type="dynamic")
    ofa_dynamic_model.load_state_dict(ofa_checkpoint_state, strict=True)
    if ofa_type == "static":
        # Get the static ofa from dynamic ofa with preserved weights.
        # Thus, it's state dictionary now contains pretrained weights.
        ofa_static_model = get_static_ofa(ofa_model=ofa_dynamic_model, input_size=input_size)
        ofa_checkpoint_state = ofa_static_model.state_dict()
    elif ofa_type == "dynamic":
        ofa_checkpoint_state = ofa_dynamic_model.state_dict()

    # 4. Load the checkpoints for the OFA component from the ofa checkpoint
    new_model_state = load_ofa_state(ofa_detr_model.state_dict(), ofa_checkpoint_state)

    # 5. Finally, load the whole DETR state with strict flag on
    ofa_detr_model.load_state_dict(new_model_state, strict=True)

    return ofa_detr_model


def merge_ofa_detr(args):
    output_dir = Path(args.output_dir)

    # First save dynamic ofa-detr model
    ofa_detr_model = build_ofa_detr(args, ofa_type="dynamic", input_size=None)
    checkpoint_path = output_dir / 'checkpoint_dynamic.pth'
    utils.save_on_master({'model': ofa_detr_model.state_dict()}, checkpoint_path)

    # Now build static ofa-detr model
    for input_size in SUPPORTED_INPUT_SIZES:
        _args = copy.deepcopy(args)
        _args.image_size = input_size
        _args.ofa_type = "static"

        ofa_detr_model = build_ofa_detr(_args, ofa_type="static", input_size=input_size)
        checkpoint_path = output_dir / f'checkpoint_static_{input_size}.pth'
        utils.save_on_master({'model': ofa_detr_model.state_dict()}, checkpoint_path)

    return


def eval_timings(args, detr_model: torch.nn.Module):
    output_dir = Path(args.output_dir)
    NUM_ITERS = 100
    max_batch_size = args.batch_size

    exec_times = np.zeros(shape=(max_batch_size, NUM_ITERS), dtype=np.float32)
    backbone_times = np.zeros((max_batch_size, NUM_ITERS), dtype=np.float32)
    transformer_times = np.zeros((max_batch_size, NUM_ITERS), dtype=np.float32)
    detection_times = np.zeros((max_batch_size, NUM_ITERS), dtype=np.float32)

    detr_model.eval()
    set_active_backbone(detr_model.backbone.feature_extractor, input_size=args.image_size)

    for batch_size in range(1, max_batch_size + 1):
        # Warmup
        for _ in range(10):
            data = torch.rand(batch_size, 3, args.image_size, args.image_size)
            data = data.cuda()
            with torch.no_grad():
                output, (backbone_time, transformer_time, detection_time) = detr_model(data,
                                                                                       stats=True)

        for iter in range(NUM_ITERS):
            data = torch.rand(batch_size, 3, args.image_size, args.image_size)
            data = data.cuda()
            torch.cuda.synchronize()
            t1 = time.time()
            with torch.no_grad():
                output, (backbone_time, transformer_time, detection_time) = detr_model(data,
                                                                                       stats=True)
            torch.cuda.synchronize()
            t2 = time.time()

            elapsed_time = (t2 - t1) * 1e3
            exec_times[batch_size - 1][iter] = elapsed_time
            backbone_times[batch_size - 1][iter] = backbone_time
            transformer_times[batch_size - 1][iter] = transformer_time
            detection_times[batch_size - 1][iter] = detection_time

    with open(output_dir / f"eval_timings_{args.image_size}.txt", "w") as out_file:
        out_file.write(f"BatchSize\tElapsedTiming(ms)\tBackboneTime\tTransformerTime\tDetectionTime\n")
        for batch_size in range(1, max_batch_size + 1):
            elapsed_time = '%.3f' % round(exec_times[batch_size - 1].mean(), 3)
            backbone_time = '%.3f' % round(backbone_times[batch_size - 1].mean(), 3)
            transformer_time = '%.3f' % round(transformer_times[batch_size - 1].mean(), 3)
            detection_time = '%.3f' % round(detection_times[batch_size - 1].mean(), 3)

            out_file.write(f"{batch_size}\t{elapsed_time}\t{backbone_time}"
                           f"\t{transformer_time}\t{detection_time}\n")

    return


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector with OFA backbone', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=0, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='ofa_resnet', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
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

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--image_size', type=int)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval_accuracy', action='store_true')
    parser.add_argument('--eval_timings', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # OFA-DETR parameters
    parser.add_argument('--ofa_type', help="Type of OFA mode", choices=["static", "dynamic"], default="dynamic")
    parser.add_argument('--merge_ofa_detr', help="merge pretrained checkpoints from OFA and DETR", action='store_true')
    parser.add_argument('--ofa_checkpoint', default='', help='path to the pretrained OFA checkpoint')
    parser.add_argument('--detr_checkpoint', default='', help='path to the pretrained DETR checkpoint')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    seed = args.seed + utils.get_rank()
    set_deterministic_behaviour(seed)

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    if args.merge_ofa_detr:
        return merge_ofa_detr(args)

    device = torch.device(args.device)
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    # print_summary(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'], strict=False)

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        print(f"Resuming from checkpoint {args.resume}")
        if not args.eval_accuracy and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval_timings:
        return eval_timings(args, model)

    dataset_train = build_dataset(image_set='train_ofa_detr', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.eval_accuracy:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('OFA-DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
