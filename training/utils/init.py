import torch


def init_folded_normal_(tensor: torch.Tensor, mean=0.0, std=1.0, fold=0.0):
    tensor.normal_(mean, std).sub_(fold).abs_().add_(fold)
