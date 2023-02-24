
import argparse
import csv
from pathlib import Path
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from datasets import build_dataset, get_coco_api_from_dataset
from datasets.coco_eval import CocoEvaluator
from models import build_sparsee_model
from models.detr import PostProcess
from models.sparsee_detr import ExitCondition, SparsEE_DETR
from util.misc import NestedTensor
import util.misc as utils
import time

_SparsEE_DETR_BATCHSIZE = 1

# @TODO: There are so much of dupliacte codes in the main execution scripts.
# Move common functions to a separate helper or util files.


def load_checkpoint(path: str):
    if path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(path, map_location='cpu')
    return checkpoint


def get_stage_output(sparsee_detr: SparsEE_DETR, tensor_list: NestedTensor, stage_idx):
    x = tensor_list
    for idx in range(stage_idx + 1):
        out = sparsee_detr._forward_stage(x, idx)
        x = out

    return x.tensors


def read_exit_cond_labels(output_dir, exit_idx):
    def read_csv(csv_file):
        labels = {}
        with open(csv_file) as f:
            csv_reader = csv.reader(f, delimiter=' ')
            for row in csv_reader:
                labels[int(row[0])] = int(row[1])  # key: ImgId, value: Binary exit decision
        return labels

    train_labels = read_csv(output_dir + f"/exit{exit_idx}_labels_train.csv")
    val_labels = read_csv(output_dir + f"/exit{exit_idx}_labels_val.csv")
    return train_labels, val_labels


def get_args_parser():
    parser = argparse.ArgumentParser('Train exit condition', add_help=False)
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
    parser.add_argument('--image_size', type=int)
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
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    # Training parameters
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--print_freq', default=50, type=int)
    parser.add_argument('--saving_freq', default=1, type=int)

    return parser


# def get_dataset_samples(sparsee_detr_model, samples, targets, labels):
#     features = []
#     exit_targets = []
#     with torch.no_grad():
#         for i in range(0, args.batch_size, _SparsEE_DETR_BATCHSIZE):
#             input = NestedTensor(samples.tensors[i:i + _SparsEE_DETR_BATCHSIZE],
#                                  samples.mask[i:i + _SparsEE_DETR_BATCHSIZE])
#             stage_out = get_stage_output(sparsee_detr_model, input, stage_idx=0)
#             features.append(stage_out.clone().detach().cpu())

#             # Get exit condition labels for the image ids.
#             for target in targets[i:i + _SparsEE_DETR_BATCHSIZE]:
#                 image_id = target["image_id"].numpy()[0]
#                 assert image_id in labels
#                 exit_targets.append(labels[image_id])
#     features = torch.cat(features, dim=0)
#     exit_targets = torch.tensor(exit_targets, dtype=float).unsqueeze(dim=-1)
#     return features, exit_targets

def get_dataset_samples(sparsee_detr_model, samples, targets, labels):
    exit_targets = []
    with torch.no_grad():
        input = samples
        stage_out = get_stage_output(sparsee_detr_model, input, stage_idx=0)
        features = stage_out.clone().detach().cpu()

        # Get exit condition labels for the image ids.
        for target in targets:
            image_id = target["image_id"].numpy()[0]
            assert image_id in labels
            exit_targets.append(labels[image_id])

    exit_targets = torch.tensor(exit_targets, dtype=float).unsqueeze(dim=-1)
    return features, exit_targets


def validate(exit_cond_model, sparsee_detr_model, data_loader, labels, base_ds, device):
    _EXIT_THRESHOLD = 0.75
    TP, FP, FN, TN = 0, 0, 0, 0
    iou_types = tuple(['bbox'])
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    postprocessors = {'bbox': PostProcess().to(device)}
    exit_cond_model.eval()
    sparsee_detr_model.eval()
    with torch.no_grad():
        for samples, targets in data_loader:
            samples = samples.to(device)
            features, exit_targets = get_dataset_samples(sparsee_detr_model, samples, targets, labels)

            features = features.to(device)
            exit_targets = exit_targets.to(device)

            # Now pass it through exit conditiom model
            exit_outputs = exit_cond_model(features)
            exit_outputs = torch.sigmoid(exit_outputs)

            for exit_output, exit_target, target in zip(exit_outputs, exit_targets, targets):
                # print(f"ImageID: {target['image_id']}, ExitCondition: {exit_output}, ExitTarget: {exit_target}")
                exit_now: bool = exit_output >= _EXIT_THRESHOLD
                if exit_now == 1 and exit_target == 1:
                    TP += 1
                elif exit_now == 1 and exit_target == 0:
                    FP += 1
                elif exit_now == 0 and exit_target == 1:
                    FN += 1
                elif exit_now == 0 and exit_target == 0:
                    TN += 1
                else:
                    raise NotImplementedError("Decision is not binary integers")

            # Measure mAP
            exit_idx, out_exit_0 = sparsee_detr_model(samples, exit_choice=0)
            exit_idx, out_exit_1 = sparsee_detr_model(samples, exit_choice=1)

            outputs = {"pred_logits": [], "pred_boxes": []}
            for i, exit_output in enumerate(exit_outputs):
                output = out_exit_0 if exit_output >= _EXIT_THRESHOLD else out_exit_1
                outputs["pred_logits"].append(output["pred_logits"][i].unsqueeze(dim=0))
                outputs["pred_boxes"].append(output["pred_boxes"][i].unsqueeze(dim=0))
            outputs["pred_logits"] = torch.cat(outputs["pred_logits"], dim=0)
            outputs["pred_boxes"] = torch.cat(outputs["pred_boxes"], dim=0)
            # outputs = out_exit_0 if exit_targets[0] == 1 else out_exit_1

            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
            results = postprocessors['bbox'](outputs, orig_target_sizes)

            res = {target['image_id'].item(): output for target, output in zip(targets, results)}
            coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    stats = {}
    stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
    total = TP + FP + FN + TN
    print(f"Validation: TP = {TP/total}, FP = {FP/total}, FN = {FN/total}, TN = {TN/total}")

    return


def train_one_epoch(args, epoch, exit_cond_model, sparsee_detr_model, data_loader, labels, device):
    sparsee_detr_model.eval()
    exit_cond_model.train()

    class_weights = torch.tensor(68764 / 49523)  # pos_samples / neg_samples
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights).to(device)
    optimizer = torch.optim.Adam(exit_cond_model.parameters(), lr=0.0001)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    t1 = time.time()
    for samples, targets in metric_logger.log_every(data_loader, args.print_freq, header):
        samples = samples.to(device)
        t2 = time.time()
        data_loading_time = (t2 - t1) * 1e3
        # First get the features from the backbone stage
        t1 = time.time()
        features, exit_targets = get_dataset_samples(sparsee_detr_model, samples, targets, labels)
        features = features.to(device)
        exit_targets = exit_targets.to(device)
        t2 = time.time()
        backbone_time = (t2 - t1) * 1e3

        # Now pass it through exit conditiom model
        t1 = time.time()
        outputs = exit_cond_model(features)
        t2 = time.time()
        exit_time = (t2 - t1) * 1e3

        # Compute loss and backprop step
        t1 = time.time()
        loss = criterion(outputs, exit_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t2 = time.time()
        backward_time = (t2 - t1) * 1e3
        # print(f"Time: {data_loading_time}, {backward_time}, {exit_time}, {backward_time}")

        metric_logger.update(loss=loss)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        t1 = time.time()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train(args, exit_cond_model, sparsee_detr_model, data_loader_train, data_loader_val, labels_train, labels_val, base_ds, device):
    output_dir = Path(args.output_dir)
    for epoch in range(args.start_epoch, args.epochs):
        validate(exit_cond_model, sparsee_detr_model, data_loader_val, labels_val, base_ds, device)

        train_stats = train_one_epoch(args, epoch, exit_cond_model, sparsee_detr_model,
                                      data_loader_train, labels_train, device)

        print(train_stats)
        # val(exit_cond_model, sparsee_detr_model, data_loader_val, labels_val, device)

        if epoch % args.saving_freq == 0:
            output_path = output_dir / "latest.pth"
            states = {
                "epochs": epoch,
                "model_state_dict": exit_cond_model.state_dict()
            }
            torch.save(states, output_path)


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

    # Build sparsee_detr model
    sparsee_detr_model, _, _ = build_sparsee_model(args)

    # Load a checkpoint
    checkpoint = load_checkpoint(args.checkpoint)
    sparsee_detr_model.load_state_dict(checkpoint["model"], strict=True)
    sparsee_detr_model.to(device)
    sparsee_detr_model.eval()

    # Build exit condition model
    exit_cond_model = ExitCondition(in_channels=512)

    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        exit_cond_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        args.start_epoch = checkpoint["epochs"]
    else:
        args.start_epoch = 0
    exit_cond_model.to(device)

    # Create COCO dataloader
    # dataset_train = build_dataset(image_set='train_exit_condition', args=args)
    dataset_train = build_dataset(image_set='train_exit_condition', args=args)
    sampler_train = torch.utils.data.SequentialSampler(dataset_train)
    data_loader_train = DataLoader(dataset_train, args.batch_size, sampler=sampler_train,
                                   drop_last=True, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    dataset_val = build_dataset(image_set='val', args=args)
    base_ds = get_coco_api_from_dataset(dataset_val)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=True, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # Load binary targets for this particular exit branch
    train_labels, val_labels = read_exit_cond_labels(args.output_dir, exit_idx=0)

    train(args,
          exit_cond_model,
          sparsee_detr_model,
          data_loader_train,
          data_loader_val,
          train_labels,
          val_labels,
          base_ds,
          device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train an exit condition model', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
