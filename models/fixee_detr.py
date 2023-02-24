from collections import OrderedDict
import copy
from typing import List
from torch import nn
import torch
import torchvision
import torch.nn.functional as F

from models.backbone import FrozenBatchNorm2d
from models.detr import DETR, build_criterion, build_postprocessor
from models.position_encoding import build_position_encoding
from models.transformer import build_transformer
from util.misc import NestedTensor, is_main_process, nested_tensor_from_tensor_list

_STAGE_LAYERS = {
    0: dict(start_layer="conv1", end_layer="layer2"),
    1: dict(start_layer="layer3", end_layer="layer4")
}

_COUPLING_MODULES = {
    0: [nn.AvgPool2d(kernel_size=(3, 3), stride=2),
        nn.MaxPool2d(kernel_size=(3, 3), stride=2)],
    1: [nn.Identity()]
}

_NUM_CHANNELS = {
    "resnet50": dict(layer2=512, layer4=2048),
    "resnet101": dict(layer2=512, layer4=2048)
}

_TRANSFORMER_CFG = {
    0: dict(enc_layers=3, dec_layers=3),
    1: dict(enc_layers=6, dec_layers=6)
}

_NAME_FMT_STAGE = "stage{0:02}"
_NAME_FMT_EXIT = "exit{0:02}"


class FixEE_DETR(nn.Module):
    """ The sparse version of the DETR model together with early exit branches"""

    def __init__(self, stages: OrderedDict, exits: OrderedDict):
        super(FixEE_DETR, self).__init__()

        self.stages = nn.ModuleDict(stages)
        self.exits = nn.ModuleDict(exits)

    def exit_now(self, outputs, exit_idx):
        _FILTER_CONF = {0: 0.25, 1: 0.25}
        _EXIT_THRESHOLD = {0: 0.70, 1: 0.75}

        out_logits = outputs['pred_logits']
        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # First filter outputs with lower confidence scores
        filtered_scores = scores[scores >= _FILTER_CONF[exit_idx]]

        # Now aggregate filtered scores and check the threshold value
        scores_mean = filtered_scores.mean()

        if scores_mean >= _EXIT_THRESHOLD[exit_idx]:
            return True

        return False

    def _forward_stage(self, samples, stage_idx):
        stage_module = self.stages[_NAME_FMT_STAGE.format(stage_idx)]
        return stage_module(samples)

    def _forward_exit_branch(self, samples, exit_idx):
        stage_out = self._forward_stage(samples, exit_idx)

        exit_module = self.exits[_NAME_FMT_EXIT.format(exit_idx)]
        exit_out = exit_module(stage_out)

        return stage_out, exit_out

    def forward(self, samples: NestedTensor, exit_choice=1):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        stage_out0, exit_out0 = self._forward_exit_branch(samples, exit_idx=0)
        if exit_choice == 0:  # or self.exit_now(exit_out0, exit_idx=0):
            return (0, exit_out0)

        stage_out1, exit_out1 = self._forward_exit_branch(stage_out0, exit_idx=1)

        return (1, exit_out1)


class BackboneStage(nn.ModuleDict):
    """ A class to represent a stage (a subset of contigous layers) in the backbone network.
        @TODO: May be change the inheritance of this class to nn.Sequential
    """

    def __init__(self, backbone, start_layer: str, end_layer: str) -> None:
        named_children = [name for name, _ in backbone.named_children()]
        start_idx, end_idx = named_children.index(start_layer), named_children.index(end_layer)
        assert start_idx <= end_idx, "start and end layers of the backbone stage are not in the order"
        stage_layers = named_children[start_idx:(end_idx + 1)]

        subset_layers = OrderedDict()
        for name, module in backbone.named_children():
            if name in stage_layers:
                subset_layers[name] = module

        super(BackboneStage, self).__init__(subset_layers)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        for name, module in self.items():
            x = module(x)

        m = tensor_list.mask
        assert m is not None
        mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        out = NestedTensor(x, mask)
        return out


class StageJoiner(nn.Module):
    """ A class to bind Stage, Coupling and Position Embedding together """

    def __init__(self, coupling: nn.Sequential, position_embeddings: nn.Module):
        super(StageJoiner, self).__init__()
        self.coupling = coupling
        self.pos_embeddings = position_embeddings

    def forward(self, tensor_list: NestedTensor):
        xc = self.coupling(tensor_list.tensors)

        m = tensor_list.mask
        assert m is not None
        mask = F.interpolate(m[None].float(), size=xc.shape[-2:]).to(torch.bool)[0]

        x = NestedTensor(xc, mask)
        out: List[NestedTensor] = [x]
        pos = [self.pos_embeddings(x).to(x.tensors.dtype)]

        return out, pos


class ExitBranch(DETR):
    """ This class is similar to DETR model. It represents on exit branch in the SparsEE-DETR"""

    def __init__(self, stage_joiner: nn.Module, transformer: nn.Module, **kwargs) -> None:
        super(ExitBranch, self).__init__(backbone=stage_joiner,
                                         transformer=transformer,
                                         **kwargs)


class ExitCondition(nn.Sequential):
    """ Exit condition model to learn exit conditions for exit branches. 

        Currently, it takes features from a backbone stage. @TODO: We should also accept features
        from transformer block (exit branch), without this the exit condition model acts as a gated model.
    """

    def __init__(self, in_channels) -> None:
        layers = [nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3),
                  nn.ReLU(),
                  # nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                  nn.AdaptiveAvgPool2d((5, 5)),  # Adaptive average pooling for constant feature size
                  nn.Flatten(),
                  nn.Linear(in_features=1600, out_features=1024),
                  nn.ReLU(),
                  nn.Linear(in_features=1024, out_features=1)]

        # layers = [nn.Conv2d(in_channels=in_channels, out_channels=1024, kernel_size=3),
        #           nn.ReLU(),
        #           nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        #           nn.AdaptiveAvgPool2d((3, 3)),  # Adaptive average pooling for constant feature size.
        #           nn.Flatten(),
        #           nn.Linear(in_features=9216, out_features=1024),
        #           nn.ReLU(),
        #           nn.Linear(in_features=1024, out_features=512),
        #           nn.ReLU(),
        #           nn.Linear(in_features=512, out_features=1)]

        super(ExitCondition, self).__init__(*layers)

    def forward(self, input):
        return super().forward(input)


def build_backbone_stage(args, backbone, stage_idx):
    return BackboneStage(backbone=backbone,
                         start_layer=_STAGE_LAYERS[stage_idx]["start_layer"],
                         end_layer=_STAGE_LAYERS[stage_idx]["end_layer"])


def build_exit_branch(args, exit_idx, num_classes):
    _args = copy.deepcopy(args)
    _args.enc_layers = _TRANSFORMER_CFG[exit_idx]["enc_layers"]
    _args.dec_layers = _TRANSFORMER_CFG[exit_idx]["dec_layers"]
    stage_idx = exit_idx

    # build stage joiner
    coupling = nn.Sequential(*_COUPLING_MODULES[stage_idx])
    pos_embeddings = build_position_encoding(_args)
    stage_joiner = StageJoiner(coupling=coupling,
                               position_embeddings=pos_embeddings)
    stage_joiner.num_channels = _NUM_CHANNELS[_args.backbone][_STAGE_LAYERS[stage_idx]["end_layer"]]

    # build transformer
    transformer = build_transformer(_args)

    exit_branch = ExitBranch(
        stage_joiner=stage_joiner,
        transformer=transformer,
        num_classes=num_classes,
        num_queries=_args.num_queries,
        aux_loss=_args.aux_loss,
    )

    device = torch.device(_args.device)
    # @TODO: This has to be declared for every exit branch???
    criterion = build_criterion(_args, num_classes)
    criterion.to(device)
    postprocessors = build_postprocessor(_args)

    return exit_branch, criterion, postprocessors


def build(args):
    num_exits = args.num_exits
    num_stages = num_exits
    num_classes = 20 if args.dataset_file != 'coco' else 91  # @TODO: Pass this as argument

    # Get the full pretrained backbone model. @TODO: For sparsity, the backbone network has to be sparse.
    backbone = getattr(torchvision.models, args.backbone)(
        replace_stride_with_dilation=[False, False, args.dilation],
        pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)

    # First build backbone stages.
    backbone_stages = OrderedDict()
    for stage_idx in range(num_stages):
        stage_name = _NAME_FMT_STAGE.format(stage_idx)
        backbone_stages[stage_name] = build_backbone_stage(args, backbone, stage_idx=stage_idx)

    # Now build exit branches
    exit_branches = OrderedDict()
    criterion_dict = OrderedDict()
    postprocessors_dict = OrderedDict()
    for exit_idx in range(num_exits):
        exit_name = _NAME_FMT_EXIT.format(exit_idx)
        exit_branches[exit_name], criterion_dict[exit_idx], postprocessors_dict[exit_idx] = build_exit_branch(
            args, exit_idx, num_classes)

    sparsee_detr = FixEE_DETR(backbone_stages, exit_branches)

    return sparsee_detr, criterion_dict, postprocessors_dict
