# -*- coding: utf-8 -*-
# @Time    : 2020
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang
import torch
import torch.nn.functional as F


def rescale_2x(x: torch.Tensor, scale_factor=2):
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)


def resize_to(x: torch.Tensor, tgt_hw: tuple):
    return F.interpolate(x, size=tgt_hw, mode="bilinear", align_corners=False)


def clip_grad(params, mode, clip_cfg: dict):
    if mode == "norm":
        if "max_norm" not in clip_cfg:
            raise ValueError("`clip_cfg` must contain `max_norm`.")
        torch.nn.utils.clip_grad_norm_(
            params,
            max_norm=clip_cfg.get("max_norm"),
            norm_type=clip_cfg.get("norm_type", 2.0),
        )
    elif mode == "value":
        if "clip_value" not in clip_cfg:
            raise ValueError("`clip_cfg` must contain `clip_value`.")
        torch.nn.utils.clip_grad_value_(params, clip_value=clip_cfg.get("clip_value"))
    else:
        raise NotImplementedError
