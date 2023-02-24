# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build
from .fixee_detr import build as build_sparsee_detr


def build_model(args):
    return build(args)


def build_sparsee_model(args):
    return build_sparsee_detr(args)
