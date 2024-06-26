# -*- coding: utf-8 -*-
# @Time    : 2020/12/19
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import copy
import math
import os.path
import warnings
from bisect import bisect_right

import matplotlib
import numpy as np
import torch.optim
from adjustText import adjust_text

matplotlib.use("Agg")
from matplotlib import pyplot as plt

# helper function ----------------------------------------------------------------------


def linear_increase(low_bound, up_bound, percentage):
    """low_bound + [0, 1] * (up_bound - low_bound)"""
    assert 0 <= percentage <= 1, f"percentage({percentage}) must be in [0, 1]"
    return low_bound + (up_bound - low_bound) * percentage


def cos_anneal(low_bound, up_bound, percentage):
    assert 0 <= percentage <= 1, f"percentage({percentage}) must be in [0, 1]"
    cos_percentage = (1 + math.cos(math.pi * percentage)) / 2.0
    return linear_increase(low_bound, up_bound, percentage=cos_percentage)


def poly_anneal(low_bound, up_bound, percentage, lr_decay):
    assert 0 <= percentage <= 1, f"percentage({percentage}) must be in [0, 1]"
    poly_percentage = pow((1 - percentage), lr_decay)
    return linear_increase(low_bound, up_bound, percentage=poly_percentage)


def linear_anneal(low_bound, up_bound, percentage):
    assert 0 <= percentage <= 1, f"percentage({percentage}) must be in [0, 1]"
    return linear_increase(low_bound, up_bound, percentage=1 - percentage)


# coefficient function ----------------------------------------------------------------------


def get_f3_coef_func(num_iters):
    """
    F3Net

    :param num_iters: The number of iterations for the total process.
    :return:
    """

    def get_f3_coef(curr_idx):
        assert 0 <= curr_idx <= num_iters
        return 1 - abs((curr_idx + 1) / (num_iters + 1) * 2 - 1)

    return get_f3_coef


def get_step_coef_func(gamma, milestones):
    """
    lr = baselr * gamma ** 0    if curr_idx < milestones[0]
    lr = baselr * gamma ** 1   if milestones[0] <= epoch < milestones[1]
    ...

    :param gamma:
    :param milestones:
    :return: The function for generating the coefficient.
    """
    if isinstance(milestones, (tuple, list)):
        milestones = list(sorted(milestones))
        return lambda curr_idx: gamma ** bisect_right(milestones, curr_idx)
    elif isinstance(milestones, int):
        return lambda curr_idx: gamma ** ((curr_idx + 1) // milestones)
    else:
        raise ValueError(f"milestones only can be list/tuple/int, but now it is {type(milestones)}")


def get_cos_coef_func(half_cycle, min_coef, max_coef=1):
    """
    :param half_cycle: The number of iterations in a half cycle.
    :param min_coef: The minimum coefficient of the learning rate.
    :param max_coef: The maximum coefficient of the learning rate.
    :return: The function for generating the coefficient.
    """

    def get_cos_coef(curr_idx):
        recomputed_idx = curr_idx % (half_cycle + 1)
        # recomputed \in [0, half_cycle]
        return cos_anneal(low_bound=min_coef, up_bound=max_coef, percentage=recomputed_idx / half_cycle)

    return get_cos_coef


def get_fatcos_coef_func(start_iter, half_cycle, min_coef, max_coef=1):
    """
    :param half_cycle: The number of iterations in a half cycle.
    :param min_coef: The minimum coefficient of the learning rate.
    :param max_coef: The maximum coefficient of the learning rate.
    :return: The function for generating the coefficient.
    """

    def get_cos_coef(curr_idx):
        curr_idx = max(0, curr_idx - start_iter)
        recomputed_idx = curr_idx % (half_cycle + 1)
        # recomputed \in [0, half_cycle]
        return cos_anneal(low_bound=min_coef, up_bound=max_coef, percentage=recomputed_idx / half_cycle)

    return get_cos_coef


def get_poly_coef_func(num_iters, lr_decay, min_coef, max_coef=1):
    """
    :param num_iters: The number of iterations for the polynomial descent process.
    :param lr_decay: The decay item of the polynomial descent process.
    :param min_coef: The minimum coefficient of the learning rate.
    :param max_coef: The maximum coefficient of the learning rate.
    :return: The function for generating the coefficient.
    """

    def get_poly_coef(curr_idx):
        assert 0 <= curr_idx <= num_iters, (curr_idx, num_iters)
        return poly_anneal(low_bound=min_coef, up_bound=max_coef, percentage=curr_idx / num_iters, lr_decay=lr_decay)

    return get_poly_coef


# coefficient entry function ----------------------------------------------------------------------


def get_scheduler_coef_func(mode, num_iters, cfg):
    """
    the region is a closed interval: [0, num_iters]
    """
    assert num_iters > 0
    min_coef = cfg.get("min_coef", 1e-6)
    if min_coef is None or min_coef == 0:
        warnings.warn(f"The min_coef ({min_coef}) of the scheduler will be replaced with 1e-6")
        min_coef = 1e-6

    if mode == "step":
        coef_func = get_step_coef_func(gamma=cfg["gamma"], milestones=cfg["milestones"])
    elif mode == "cos":
        if half_cycle := cfg.get("half_cycle"):
            half_cycle -= 1
        else:
            half_cycle = num_iters
        if (num_iters - half_cycle) % (half_cycle + 1) != 0:
            # idx starts from 0
            percentage = ((num_iters - half_cycle) % (half_cycle + 1)) / (half_cycle + 1) * 100
            warnings.warn(
                f"The final annealing process ({percentage:.3f}%) is not complete. "
                f"Please pay attention to the generated 'lr_coef_curve.png'."
            )
        coef_func = get_cos_coef_func(half_cycle=half_cycle, min_coef=min_coef)
    elif mode == "fatcos":
        assert 0 <= cfg.start_percent < 1, cfg.start_percent
        start_iter = int(cfg.start_percent * num_iters)

        num_iters -= start_iter
        if half_cycle := cfg.get("half_cycle"):
            half_cycle -= 1
        else:
            half_cycle = num_iters
        if (num_iters - half_cycle) % (half_cycle + 1) != 0:
            # idx starts from 0
            percentage = ((num_iters - half_cycle) % (half_cycle + 1)) / (half_cycle + 1) * 100
            warnings.warn(
                f"The final annealing process ({percentage:.3f}%) is not complete. "
                f"Please pay attention to the generated 'lr_coef_curve.png'."
            )
        coef_func = get_fatcos_coef_func(start_iter=start_iter, half_cycle=half_cycle, min_coef=min_coef)
    elif mode == "poly":
        coef_func = get_poly_coef_func(num_iters=num_iters, lr_decay=cfg["lr_decay"], min_coef=min_coef)
    elif mode == "constant":
        coef_func = lambda x: cfg.get("coef", 1)
    elif mode == "f3":
        coef_func = get_f3_coef_func(num_iters=num_iters)
    else:
        raise NotImplementedError(f"{mode} must be in {Scheduler.supported_scheduler_modes}")
    return coef_func


def get_warmup_coef_func(num_iters, min_coef, max_coef=1, mode="linear"):
    """
    the region is a closed interval: [0, num_iters]
    """
    assert num_iters > 0
    if mode == "cos":
        anneal_func = cos_anneal
    elif mode == "linear":
        anneal_func = linear_anneal
    else:
        raise NotImplementedError(f"{mode} must be in {Scheduler.supported_warmup_modes}")

    def get_warmup_coef(curr_idx):
        return anneal_func(low_bound=min_coef, up_bound=max_coef, percentage=1 - curr_idx / num_iters)

    return get_warmup_coef


# main class ----------------------------------------------------------------------


class Scheduler:
    supported_scheduler_modes = ("step", "cos", "fatcos", "poly", "constant", "f3")
    supported_warmup_modes = ("cos", "linear")

    def __init__(self, optimizer, num_iters, epoch_length, scheduler_cfg, step_by_batch=True):
        """A customized wrapper of the scheduler.

        Args:
            optimizer (): Optimizer.
            num_iters (int): The total number of the iterations.
            epoch_length (int): The number of the iterations of one epoch.
            scheduler_cfg (dict): The config of the scheduler.
            step_by_batch (bool, optional): The mode of updating the scheduler. Defaults to True.

        Raises:
            NotImplementedError:
        """
        self.optimizer = optimizer
        self.num_iters = num_iters
        self.epoch_length = epoch_length
        self.step_by_batch = step_by_batch

        self.scheduler_cfg = copy.deepcopy(scheduler_cfg)
        self.mode = scheduler_cfg["mode"]
        if self.mode not in self.supported_scheduler_modes:
            raise NotImplementedError(
                f"{self.mode} is not implemented. Has been supported: {self.supported_scheduler_modes}"
            )
        warmup_cfg = scheduler_cfg.get("warmup", None)

        num_warmup_iters = 0
        if warmup_cfg is not None and isinstance(warmup_cfg, dict):
            num_warmup_iters = warmup_cfg["num_iters"]
            if num_warmup_iters > 0:
                print("Will using warmup")
                self.warmup_coef_func = get_warmup_coef_func(
                    num_warmup_iters,
                    min_coef=warmup_cfg.get("initial_coef", 0.01),
                    mode=warmup_cfg.get("mode", "linear"),
                )
        self.num_warmup_iters = num_warmup_iters

        if step_by_batch:
            num_scheduler_iters = num_iters - num_warmup_iters
        else:
            num_scheduler_iters = (num_iters - num_warmup_iters) // epoch_length
        # the region is a closed interval
        self.lr_coef_func = get_scheduler_coef_func(
            mode=self.mode, num_iters=num_scheduler_iters - 1, cfg=scheduler_cfg["cfg"]
        )
        self.num_scheduler_iters = num_scheduler_iters

        self.last_lr_coef = 0
        self.initial_lrs = None

    def __repr__(self):
        formatted_string = [
            f"{self.__class__.__name__}: (\n",
            f"num_iters: {self.num_iters}\n",
            f"epoch_length: {self.epoch_length}\n",
            f"warmup_iter: [0, {self.num_warmup_iters})\n",
            f"scheduler_iter: [{self.num_warmup_iters}, {self.num_iters - 1}]\n",
            f"mode: {self.mode}\n",
            f"scheduler_cfg: {self.scheduler_cfg}\n",
            f"initial_lrs: {self.initial_lrs}\n",
            f"step_by_batch: {self.step_by_batch}\n)",
        ]
        return "    ".join(formatted_string)

    def record_lrs(self, param_groups):
        self.initial_lrs = [g["lr"] for g in param_groups]

    def update(self, coef: float):
        assert self.initial_lrs is not None, "Please run .record_lrs(optimizer) first."
        for curr_group, initial_lr in zip(self.optimizer.param_groups, self.initial_lrs):
            curr_group["lr"] = coef * initial_lr

    def step(self, curr_idx):
        if curr_idx < self.num_warmup_iters:
            # get maximum value (1.0) when curr_idx == self.num_warmup_iters
            self.update(coef=self.get_lr_coef(curr_idx))
        else:
            # Start from a value lower than 1 (curr_idx == self.num_warmup_iters)
            if self.step_by_batch:
                self.update(coef=self.get_lr_coef(curr_idx))
            else:
                if curr_idx % self.epoch_length == 0:
                    self.update(coef=self.get_lr_coef(curr_idx))

    def get_lr_coef(self, curr_idx):
        coef = None
        if curr_idx < self.num_warmup_iters:
            coef = self.warmup_coef_func(curr_idx)
        else:
            # when curr_idx == self.num_warmup_iters, coef == 1.0
            # down from the largest coef (1.0)
            if self.step_by_batch:
                coef = self.lr_coef_func(curr_idx - self.num_warmup_iters)
            else:
                if curr_idx % self.epoch_length == 0 or curr_idx == self.num_warmup_iters:
                    # warmup结束后尚未开始按照epoch进行调整的学习率调整，此时需要将系数调整为最大。
                    coef = self.lr_coef_func((curr_idx - self.num_warmup_iters) // self.epoch_length)
        if coef is not None:
            self.last_lr_coef = coef
        return self.last_lr_coef

    def plot_lr_coef_curve(self, save_path=""):
        plt.rc("xtick", labelsize="small")
        plt.rc("ytick", labelsize="small")

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 4), dpi=600)
        # give plot a title
        ax.set_title("Learning Rate Coefficient Curve")
        # make axis labels
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Coefficient")

        x_data = np.arange(self.num_iters)
        y_data = np.array([self.get_lr_coef(x) for x in x_data])

        # set lim
        x_min, x_max = 0, self.num_iters - 1
        dx = self.num_iters * 0.1
        ax.set_xlim(x_min - dx, x_max + 2 * dx)

        y_min, y_max = y_data.min(), y_data.max()
        dy = (y_data.max() - y_data.min()) * 0.1
        ax.set_ylim((y_min - dy, y_max + dy))

        if self.step_by_batch:
            marker_on = [0, -1]
            key_point_xs = [0, self.num_iters - 1]
            for idx in range(1, len(y_data) - 1):
                prev_y = y_data[idx - 1]
                curr_y = y_data[idx]
                next_y = y_data[idx + 1]
                if (
                    (curr_y > prev_y and curr_y >= next_y)
                    or (curr_y >= prev_y and curr_y > next_y)
                    or (curr_y <= prev_y and curr_y < next_y)
                    or (curr_y < prev_y and curr_y <= next_y)
                ):
                    marker_on.append(idx)
                    key_point_xs.append(idx)

            marker_on = sorted(set(marker_on))
            key_point_xs = sorted(set(key_point_xs))
            key_point_ys = []

            texts = []
            for x in key_point_xs:
                y = y_data[x]
                key_point_ys.append(y)

                texts.append(ax.text(x=x, y=y, s=f"({x:d},{y:.3e})"))
            adjust_text(texts, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.3"))

            # set ticks
            ax.set_xticks(key_point_xs)
            # ax.set_yticks(key_point_ys)

            ax.plot(x_data, y_data, marker="o", markevery=marker_on)
        else:
            ax.plot(x_data, y_data)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)

        plt.tight_layout()
        if save_path:
            fig.savefig(os.path.join(save_path, "lr_coef.png"))
        plt.close()


if __name__ == "__main__":
    model = torch.nn.Conv2d(10, 10, 3, 1, 1)
    sche = Scheduler(
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        num_iters=30300,
        epoch_length=505,
        scheduler_cfg=dict(
            warmup=dict(
                num_iters=6060,
                initial_coef=0.01,
                mode="cos",
            ),
            mode="cos",
            cfg=dict(
                half_cycle=6060,
                lr_decay=0.9,
                min_coef=0.001,
            ),
        ),
        step_by_batch=True,
    )
    print(sche)
    sche.plot_lr_coef_curve(
        # save_path="/home/lart/Coding/SOD.torch",
        show=True,
    )
