# -*- coding: utf-8 -*-
# @Time    : 2020/12/19
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang
import types

import torchvision.models
from torch import nn
from torch.optim import SGD, Adam, AdamW


def get_optimizer(mode, params, initial_lr, optim_cfg):
    if mode == "sgd":
        optimizer = SGD(
            params=params,
            lr=initial_lr,
            momentum=optim_cfg["momentum"],
            weight_decay=optim_cfg["weight_decay"],
            nesterov=optim_cfg.get("nesterov", False),
        )
    elif mode == "adamw":
        optimizer = AdamW(
            params=params,
            lr=initial_lr,
            betas=optim_cfg.get("betas", (0.9, 0.999)),
            weight_decay=optim_cfg.get("weight_decay", 0),
            amsgrad=optim_cfg.get("amsgrad", False),
        )
    elif mode == "adam":
        optimizer = Adam(
            params=params,
            lr=initial_lr,
            betas=optim_cfg.get("betas", (0.9, 0.999)),
            weight_decay=optim_cfg.get("weight_decay", 0),
            amsgrad=optim_cfg.get("amsgrad", False),
        )
    else:
        raise NotImplementedError(mode)
    return optimizer


def group_params(model: nn.Module, group_mode: str, initial_lr: float, optim_cfg: dict):
    if group_mode == "yolov5":
        """
        norm, weight, bias = [], [], []  # optimizer parameter groups
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                bias.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                norm.append(v.weight)  # no decay
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                weight.append(v.weight)  # apply decay

        if opt.adam:
            optimizer = optim.Adam(norm, lr=hyp["lr0"], betas=(hyp["momentum"], 0.999))  # adjust beta1 to momentum
        else:
            optimizer = optim.SGD(norm, lr=hyp["lr0"], momentum=hyp["momentum"], nesterov=True)

        optimizer.add_param_group({"params": weight, "weight_decay": hyp["weight_decay"]})  # add weight with weight_decay
        optimizer.add_param_group({"params": bias})  # add bias (biases)
        """
        norm, weight, bias = [], [], []  # optimizer parameter groups
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                bias.append(v.bias)  # conv bias and bn bias
            if isinstance(v, nn.BatchNorm2d):
                norm.append(v.weight)  # bn weight
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                weight.append(v.weight)  # conv weight
        params = [
            {"params": filter(lambda p: p.requires_grad, bias), "weight_decay": 0.0},
            {"params": filter(lambda p: p.requires_grad, norm), "weight_decay": 0.0},
            {"params": filter(lambda p: p.requires_grad, weight)},
        ]
    elif group_mode == "r3":
        params = [
            # 不对bias参数执行weight decay操作，weight decay主要的作用就是通过对网络
            # 层的参数（包括weight和bias）做约束（L2正则化会使得网络层的参数更加平滑）达
            # 到减少模型过拟合的效果。
            {
                "params": [
                    param for name, param in model.named_parameters() if name[-4:] == "bias" and param.requires_grad
                ],
                "lr": 2 * initial_lr,
                "weight_decay": 0,
            },
            {
                "params": [
                    param for name, param in model.named_parameters() if name[-4:] != "bias" and param.requires_grad
                ],
                "lr": initial_lr,
                "weight_decay": optim_cfg["weight_decay"],
            },
        ]
    elif group_mode == "all":
        params = model.parameters()
    elif group_mode == "finetune":
        if hasattr(model, "module"):
            model = model.module
        assert hasattr(model, "get_grouped_params"), "Cannot get the method get_grouped_params of the model."
        params_groups = model.get_grouped_params()
        params = [
            {
                "params": filter(lambda p: p.requires_grad, params_groups["pretrained"]),
                "lr": optim_cfg.get("diff_factor", 0.1) * initial_lr,
            },
            {
                "params": filter(lambda p: p.requires_grad, params_groups["retrained"]),
                "lr": initial_lr,
            },
        ]
    elif group_mode == "finetune2":
        if hasattr(model, "module"):
            model = model.module
        assert hasattr(model, "get_grouped_params"), "Cannot get the method get_grouped_params of the model."
        params_groups = model.get_grouped_params()
        params = [
            {
                "params": filter(lambda p: p.requires_grad, params_groups["pretrained_backbone"]),
                "lr": 0.1 * initial_lr,
            },
            {
                "params": filter(lambda p: p.requires_grad, params_groups["pretrained_siamese"]),
                "lr": 0.5 * initial_lr,
            },
            {
                "params": filter(lambda p: p.requires_grad, params_groups["retrained"]),
                "lr": initial_lr,
            },
        ]
    else:
        raise NotImplementedError
    return params


def construct_optimizer(model, initial_lr, mode, group_mode, cfg):
    params = group_params(model, group_mode=group_mode, initial_lr=initial_lr, optim_cfg=cfg)
    optimizer = get_optimizer(mode=mode, params=params, initial_lr=initial_lr, optim_cfg=cfg)
    optimizer.lr_groups = types.MethodType(get_lr_groups, optimizer)
    optimizer.lr_string = types.MethodType(get_lr_strings, optimizer)
    return optimizer


def get_lr_groups(self):
    return [group["lr"] for group in self.param_groups]


def get_lr_strings(self):
    return ",".join([f"{group['lr']:.3e}" for group in self.param_groups])


if __name__ == "__main__":
    model = torchvision.models.vgg11_bn()
    norm, weight, bias = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            bias.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            norm.append(v.weight)  # no decay
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            weight.append(v.weight)  # apply decay

    optimizer = Adam(norm, lr=0.001, betas=(0.98, 0.999))  # adjust beta1 to momentum
    # optimizer = optim.SGD(norm, lr=hyp["lr0"], momentum=hyp["momentum"], nesterov=True)

    optimizer.add_param_group({"params": weight, "weight_decay": 1e-4})  # add weight with weight_decay
    optimizer.add_param_group({"params": bias})  # add bias (biases)

    print(optimizer)
