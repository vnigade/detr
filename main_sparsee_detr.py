import argparse
from pathlib import Path

import torch
from models import build_sparsee_model
from models.fixee_detr import _NAME_FMT_EXIT
from models.ofa_backbone import SUPPORTED_INPUT_SIZES

import util.misc as utils
from utils import load_checkpoint, load_ofa_state, load_partial_state, set_deterministic_behaviour


def get_args_parser():
    parser = argparse.ArgumentParser('SparsEE-DETR with OFA backbone', add_help=False)
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
    parser.add_argument('--num_workers', default=2, type=int)

    # OFA-DETR parameters
    parser.add_argument('--merge_sparsee_detr',
                        help="merge pretrained checkpoints from OFA and DETR", action='store_true')
    parser.add_argument('--ofa_checkpoint', default='', help='path to the pretrained OFA checkpoint')
    parser.add_argument('--exits_checkpoint', action='append',
                        help="The list of checkpoint paths used for the exit branches")

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def merge_sparsee_detr(agrs):
    output_dir = Path(args.output_dir)

    # 1. First build SPARSE + EE + DETR model
    sparsee_detr_model, criterion, postprocessors = build_sparsee_model(args)

    # 2. Load the checkpoints for the OFA backbone component from the ofa checkpoint
    model_state_dict = {}
    ofa_checkpoint_state = load_checkpoint(args.ofa_checkpoint)["state_dict"]
    new_model_state = load_ofa_state(sparsee_detr_model.state_dict(), ofa_checkpoint_state, key_split_idx=1)
    model_state_dict = {**model_state_dict, **new_model_state}

    # 3. Now load exit checkpoints
    for exit_idx in range(args.num_exits):
        state_dict = load_partial_state(sparsee_detr_model.state_dict(),
                                        args.exits_checkpoint[exit_idx],
                                        layer_prefix=f"exits.{_NAME_FMT_EXIT.format(exit_idx)}")
        model_state_dict = {**model_state_dict, **state_dict}

    # 4. Now load the checkpoint into the SparsEE-DETR model with strict flag on
    sparsee_detr_model.load_state_dict(model_state_dict, strict=True)

    checkpoint_path = output_dir / 'checkpoint_dynamic.pth'
    utils.save_on_master({
        'model': sparsee_detr_model.state_dict(),
        'args': args
    }, checkpoint_path)

    print(f"Merged and saved SparsEE-DETR at {checkpoint_path}")


def eval_timings(args, NUM_ITERS=100):
    sparsee_detr_model, criterion, postprocessors = build_sparsee_model(args)

    checkpoint = load_checkpoint(args.resume)
    sparsee_detr_model.load_state_dict(checkpoint["model"], strict=True)

    model_number = 0
    sparsee_detr_model.set_subnet_model(model_number=model_number)

    input_size = SUPPORTED_INPUT_SIZES[model_number]
    for batch_size in range(4, 4 + 1):
        for iter in range(NUM_ITERS):
            data = torch.rand(batch_size, 3, input_size, input_size)
            # data = data.cuda(gpu_number)

            sparsee_detr_model.forward(samples=data)


def main(args):
    seed = args.seed + utils.get_rank()
    set_deterministic_behaviour(seed)

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.merge_sparsee_detr:
        return merge_sparsee_detr(args)

    eval_timings(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SparsEE-DETR main script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
