
import argparse
import csv
from pathlib import Path
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from datasets import build_dataset
from models import build_sparsee_model
from models.sparsee_detr import ExitCondition, SparsEE_DETR
from util.misc import NestedTensor
import util.misc as utils

_SparsEE_DETR_BATCHSIZE = 4

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
    parser.add_argument('--batch_size', default=4, type=int)

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
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    # Training parameters
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--print_freq', default=5, type=int)
    parser.add_argument('--saving_freq', default=5, type=int)

    return parser


def get_dataset_samples(sparsee_detr_model, samples, targets, labels):
    features = []
    exit_targets = []
    with torch.no_grad():
        for i in range(0, args.batch_size, _SparsEE_DETR_BATCHSIZE):
            input = NestedTensor(samples.tensors[i:i + _SparsEE_DETR_BATCHSIZE],
                                 samples.mask[i:i + _SparsEE_DETR_BATCHSIZE])
            stage_out = get_stage_output(sparsee_detr_model, input, stage_idx=0)
            features.append(stage_out.detach().cpu())

            # Get exit condition labels for the image ids.
            for target in targets[i:i + _SparsEE_DETR_BATCHSIZE]:
                image_id = target["image_id"].numpy()[0]
                assert image_id in labels
                exit_targets.append(labels[image_id])
    features = torch.cat(features, dim=0)
    exit_targets = torch.tensor(exit_targets, dtype=float).unsqueeze(dim=-1)
    return features, exit_targets


def val(exit_cond_model, sparsee_detr_model, data_loader, labels, device):
    _EXIT_THRESHOLD = 0.75
    total_pred, correct_pred = 0, 0
    with torch.no_grad():
        for samples, targets in data_loader:
            samples = samples.to(device)
            features, exit_targets = get_dataset_samples(sparsee_detr_model, samples, targets, labels)

            features = features.to(device)
            exit_targets = exit_targets.to(device)

            # Now pass it through exit conditiom model
            outputs = exit_cond_model(features)
            outputs = torch.sigmoid(outputs)

            for output, exit_target in zip(outputs, exit_targets):
                exit_now: bool = output >= _EXIT_THRESHOLD
                if exit_now == exit_target:
                    # print("Correct prediction")
                    correct_pred += 1
                # else:
                    # print("Wrong prediction")
                total_pred += 1

    print(f"Validation: accuracy {(correct_pred/total_pred)*100}")


def train_one_epoch(args, epoch, exit_cond_model, sparsee_detr_model, data_loader, labels, device):
    sparsee_detr_model.eval()
    exit_cond_model.train()

    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(exit_cond_model.parameters())

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for samples, targets in metric_logger.log_every(data_loader, args.print_freq, header):
        samples = samples.to(device)
        # First get the features from the backbone stage
        features, exit_targets = get_dataset_samples(sparsee_detr_model, samples, targets, labels)
        features = features.to(device)
        exit_targets = exit_targets.to(device)

        # Now pass it through exit conditiom model
        outputs = exit_cond_model(features)

        # Compute loss and backprop step
        loss = criterion(outputs, exit_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train(args, exit_cond_model, sparsee_detr_model, data_loader_train, data_loader_val, labels_train, labels_val, device):
    output_dir = Path(args.output_dir)
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(args, epoch, exit_cond_model, sparsee_detr_model,
                                      data_loader_train, labels_train, device)

        print(train_stats)
        val(exit_cond_model, sparsee_detr_model, data_loader_val, labels_val, device)

        if epoch % args.saving_freq == 0:
            output_path = output_dir / "latest.pth"
            states = {
                "epochs": args.epochs,
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
    dataset_train = build_dataset(image_set='train', args=args)
    sampler_train = torch.utils.data.SequentialSampler(dataset_train)
    data_loader_train = DataLoader(dataset_train, args.batch_size, sampler=sampler_train,
                                   drop_last=True, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    dataset_val = build_dataset(image_set='val', args=args)
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
          val_labels, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train an exit condition model', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
