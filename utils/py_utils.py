# -*- coding: utf-8 -*-
import copy
import logging
import os
import shutil
from collections import OrderedDict, abc
from datetime import datetime

LOGGER = logging.getLogger("main")


def construct_path(output_dir: str, exp_name: str) -> dict:
    proj_root = os.path.join(output_dir, exp_name)
    exp_idx = 0
    exp_output_dir = os.path.join(proj_root, f"exp_{exp_idx}")
    while os.path.exists(exp_output_dir):
        exp_idx += 1
        exp_output_dir = os.path.join(proj_root, f"exp_{exp_idx}")

    tb_path = os.path.join(exp_output_dir, "tb")
    save_path = os.path.join(exp_output_dir, "pre")
    pth_path = os.path.join(exp_output_dir, "pth")

    final_full_model_path = os.path.join(pth_path, "checkpoint_final.pth")
    final_state_path = os.path.join(pth_path, "state_final.pth")

    log_path = os.path.join(exp_output_dir, f"log_{str(datetime.now())[:10]}.txt")
    cfg_copy_path = os.path.join(exp_output_dir, f"config.py")
    trainer_copy_path = os.path.join(exp_output_dir, f"trainer.txt")
    excel_path = os.path.join(exp_output_dir, f"results.xlsx")

    path_config = {
        "output_dir": output_dir,
        "pth_log": exp_output_dir,
        "tb": tb_path,
        "save": save_path,
        "pth": pth_path,
        "final_full_net": final_full_model_path,
        "final_state_net": final_state_path,
        "log": log_path,
        "cfg_copy": cfg_copy_path,
        "excel": excel_path,
        "trainer_copy": trainer_copy_path,
    }

    return path_config


def construct_exp_name(model_name: str, cfg: dict):
    # bs_16_lr_0.05_e30_noamp_2gpu_noms_352
    focus_item = OrderedDict(
        {
            "train/batch_size": "bs",
            "train/lr": "lr",
            "train/num_epochs": "e",
            "train/num_iters": "i",
            "train/data/shape/h": "h",
            "train/data/shape/w": "w",
            "train/optimizer/mode": "opm",
            "train/optimizer/group_mode": "opgm",
            "train/scheduler/mode": "sc",
            "train/scheduler/warmup/num_iters": "wu",
            "train/use_amp": "amp",
        }
    )
    config = copy.deepcopy(cfg)

    def _format_item(_i):
        if isinstance(_i, bool):
            _i = "" if _i else "false"
        elif isinstance(_i, (int, float)):
            if _i == 0:
                _i = "false"
        elif isinstance(_i, (list, tuple)):
            _i = "" if _i else "false"  # 只是判断是否非空
        elif isinstance(_i, str):
            if "_" in _i:
                _i = _i.replace("_", "").lower()
        elif _i is None:
            _i = "none"
        # else: other types and values will be returned directly
        return _i

    if (epoch_based := config.train.get("epoch_based", None)) is not None and (not epoch_based):
        focus_item.pop("train/num_epochs")
    else:
        # 默认基于epoch
        focus_item.pop("train/num_iters")

    exp_names = [model_name]
    for key, alias in focus_item.items():
        item = get_value_recurse(keys=key.split("/"), info=config)
        formatted_item = _format_item(item)
        if formatted_item == "false":
            continue
        exp_names.append(f"{alias.upper()}{formatted_item}")

    info = config.get("info", None)
    if info:
        exp_names.append(f"INFO{info.lower()}")

    return "_".join(exp_names)


def pre_mkdir(path_config):
    # 提前创建好记录文件，避免自动创建的时候触发文件创建事件
    check_mkdir(path_config["pth_log"])
    make_log(path_config["log"], f"=== log {datetime.now()} ===")

    # 提前创建好存储预测结果和存放模型的文件夹
    check_mkdir(path_config["save"])
    check_mkdir(path_config["pth"])


def check_mkdir(dir_name, delete_if_exists=False):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    else:
        if delete_if_exists:
            print(f"{dir_name} will be re-created!!!")
            shutil.rmtree(dir_name)
            os.makedirs(dir_name)


def make_log(path, context):
    with open(path, "a") as log:
        log.write(f"{context}\n")


def iterate_nested_sequence(nested_sequence):
    """
    当前支持list/tuple/int/float/range()的多层嵌套，注意不要嵌套的太深，小心超出python默认的最大递归深度

    例子
    ::

        for x in iterate_nested_sequence([[1, (2, 3)], range(3, 10), 0]):
            print(x)

        1
        2
        3
        3
        4
        5
        6
        7
        8
        9
        0

    :param nested_sequence: 多层嵌套的序列
    :return: generator
    """
    for item in nested_sequence:
        if isinstance(item, (int, float)):
            yield item
        elif isinstance(item, (list, tuple, range)):
            yield from iterate_nested_sequence(item)
        else:
            raise NotImplementedError


def get_value_recurse(keys: list, info: dict):
    curr_key, sub_keys = keys[0], keys[1:]

    if (sub_info := info.get(curr_key, "NoKey")) == "NoKey":
        raise KeyError(f"{curr_key} must be contained in {info}")

    if sub_keys:
        return get_value_recurse(keys=sub_keys, info=sub_info)
    else:
        return sub_info


def mapping_to_str(mapping: abc.Mapping, *, prefix: str = "    ", lvl: int = 0, max_lvl: int = 1) -> str:
    """
    Print the structural information of the dict.
    """
    sub_lvl = lvl + 1
    cur_prefix = prefix * lvl
    sub_prefix = prefix * sub_lvl

    if lvl == max_lvl:
        sub_items = str(mapping)
    else:
        sub_items = ["{"]
        for k, v in mapping.items():
            sub_item = sub_prefix + k + ": "
            if isinstance(v, abc.Mapping):
                sub_item += mapping_to_str(v, prefix=prefix, lvl=sub_lvl, max_lvl=max_lvl)
            else:
                sub_item += str(v)
            sub_items.append(sub_item)
        sub_items.append(cur_prefix + "}")
        sub_items = "\n".join(sub_items)
    return sub_items
