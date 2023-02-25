
from torch import nn
import torch
from collections import OrderedDict
from models.fixee_detr import _NAME_FMT_EXIT, build_exit_branches
from models.ofa_backbone import SUPPORTED_INPUT_SIZES, build_ofa_backbone, set_active_backbone
from util.misc import NestedTensor, nested_tensor_from_tensor_list


class SparsEE_DETR(nn.Module):
    EXIT_PROBABILITY = 0.4
    MAX_BATCH_SIZE = 8

    def __init__(self, backbone: nn.Module, exits: OrderedDict) -> None:
        super(SparsEE_DETR, self).__init__()
        self.backbone = backbone
        self.exits = nn.ModuleDict(exits)
        self.max_batch_size = torch.tensor(SparsEE_DETR.MAX_BATCH_SIZE)

        # Output vectors
        self.output_pred_logits = torch.zeros(size=(8, 100, 92), dtype=torch.float)
        self.output_pred_boxes = torch.zeros(size=(8, 100, 4), dtype=torch.float)
        self.output_done_flag = torch.zeros(8, dtype=torch.bool)

    def set_subnet_model(self, model_number):
        set_active_backbone(ofa_model=self.backbone, input_size=SUPPORTED_INPUT_SIZES[model_number])

    def exit_list(self, stage_features, exit_output, batch_size):
        def random_exit_list():
            prob_tensor = torch.full(size=(self.max_batch_size,), fill_value=SparsEE_DETR.EXIT_PROBABILITY)
            prob_tensor[batch_size:] = 0.0
            rand_list = torch.bernoulli(prob_tensor) > 0.0
            return rand_list
        return random_exit_list()

    def _forward_exit(self, input, exit_idx):
        exit_module = self.exits[_NAME_FMT_EXIT.format(exit_idx)]
        return exit_module(input)

    def forward(self, samples: NestedTensor, exit_choice=None):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        input = samples.tensors
        batch_size = input.shape[0]

        # Pass through first branch
        stage0_out = self.backbone._forward_part(input, part_idx=0)
        exit0_out = self._forward_exit(input=stage0_out, exit_idx=0)

        # Check which samples should exit at exit-branch 0
        exit_list = self.exit_list(stage0_out, exit0_out, batch_size=batch_size)

        # Copy the samples exited at exit-branch 0 in the output buffer of fixed size
        self.output_pred_logits[exit_list] = exit0_out["pred_logits"][exit_list[:batch_size]]
        self.output_pred_boxes[exit_list] = exit0_out["pred_boxes"][exit_list[:batch_size]]
        self.output_done_flag[exit_list] = True

        # Pass remaining samples to the second branch
        remaining_list = ~exit_list
        remaining_list[batch_size:] = False
        remaining_features = stage0_out[remaining_list[:batch_size]]
        stage1_out = self.backbone._forward_part(remaining_features, part_idx=1)
        exit1_out = self._forward_exit(input=stage1_out, exit_idx=1)

        # Copy the samples exited at exit-branch 0 in the output buffer
        self.output_pred_logits[remaining_list] = exit1_out["pred_logits"]
        self.output_pred_boxes[remaining_list] = exit1_out["pred_boxes"]
        self.output_done_flag[remaining_list] = True

        return

    def forward_async(self, samples: NestedTensor, output: torch.Tensor):
        pass


def build(args):
    num_exits = args.num_exits
    num_classes = 91  # Currently, we do it only for coco dataset

    backbone = build_ofa_backbone(ofa_type="dynamic", input_size=None)

    # Now build exit branches
    exit_branches, criterion_dict, postprocessors_dict = build_exit_branches(args, num_classes)

    sparsee_detr = SparsEE_DETR(backbone, exit_branches)

    return sparsee_detr, criterion_dict, postprocessors_dict
