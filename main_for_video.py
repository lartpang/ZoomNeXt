# -*- coding: utf-8 -*-
import argparse
import datetime
import glob
import inspect
import logging
import os
import re
import shutil
import time

import albumentations as A
import colorlog
import cv2
import numpy as np
import torch
import yaml
from mmengine import Config
from torch.utils import data
from tqdm import tqdm

import methods as model_zoo
from utils import io, ops, pipeline, pt_utils, py_utils, recorder

LOGGER = logging.getLogger("main")
LOGGER.propagate = False
LOGGER.setLevel(level=logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(colorlog.ColoredFormatter("%(log_color)s[%(filename)s] %(reset)s%(message)s"))
LOGGER.addHandler(stream_handler)


def construct_frame_transform():
    return A.Compose(
        [
            A.Rotate(limit=90, p=0.5, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT101),
            A.RandomBrightnessContrast(brightness_limit=0.02, contrast_limit=0.02, p=0.5),
        ]
    )


def construct_video_transform():
    return A.ReplayCompose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.5),
        ]
    )


def get_number_from_tail(string):
    return int(re.findall(pattern="\d+$", string=string)[0])


class VideoTestDataset(data.Dataset):
    def __init__(self, dataset_info: dict, shape: dict, num_frames: int = 1, overlap: int = 0):
        super().__init__()
        self.shape = shape
        self.num_frames = num_frames
        assert 0 <= overlap < num_frames

        image_path = os.path.join(dataset_info["root"], dataset_info["image"]["path"])
        image_suffix = dataset_info["image"]["suffix"]
        mask_path = os.path.join(dataset_info["root"], dataset_info["mask"]["path"])
        mask_suffix = dataset_info["mask"]["suffix"]

        image_group_paths = sorted(glob.glob(image_path))
        mask_group_paths = sorted(glob.glob(mask_path))
        group_name_place = image_path.find("*")

        self.total_data_paths = []
        for image_group_path, mask_group_path in zip(image_group_paths, mask_group_paths):
            group_name = image_group_path[group_name_place:].split("/")[0]
            image_names = [
                p[: -len(image_suffix)] for p in sorted(os.listdir(image_group_path)) if p.endswith(image_suffix)
            ]
            mask_names = [
                p[: -len(mask_suffix)] for p in sorted(os.listdir(mask_group_path)) if p.endswith(mask_suffix)
            ]
            valid_names = sorted(
                set(image_names).intersection(mask_names), key=lambda item: get_number_from_tail(item)
            )
            assert len(valid_names) >= num_frames, image_group_path

            i = 0
            while i < len(valid_names):
                if i + num_frames > len(valid_names):
                    i = len(valid_names) - num_frames
                clip_info = [
                    (
                        os.path.join(image_group_path, n) + image_suffix,
                        os.path.join(mask_group_path, n) + mask_suffix,
                        group_name,
                    )
                    for n in valid_names[i : i + num_frames]
                ]
                if len(clip_info) < num_frames:
                    times, last = divmod(num_frames, len(clip_info))
                    clip_info.extend(clip_info * (times - 1))
                    clip_info.extend(clip_info[:last])
                if self.total_data_paths and clip_info == self.total_data_paths[-1]:
                    break
                self.total_data_paths.append(clip_info)

                i += num_frames - overlap

    def __getitem__(self, index):
        clip_info = self.total_data_paths[index]

        base_h = self.shape["h"]
        base_w = self.shape["w"]
        image_ss = []
        image_ms = []
        image_ls = []
        mask_paths = []
        for image_path, mask_path, group_name in clip_info:
            image = io.read_color_array(image_path)
            mask_paths.append(mask_path)
            images = ops.ms_resize(image, scales=(0.5, 1.0, 1.5), base_h=base_h, base_w=base_w)
            image_ss.append(torch.from_numpy(images[0]).div(255).permute(2, 0, 1))
            image_ms.append(torch.from_numpy(images[1]).div(255).permute(2, 0, 1))
            image_ls.append(torch.from_numpy(images[2]).div(255).permute(2, 0, 1))

        return dict(
            data={
                "image_s": torch.stack(image_ss, dim=0),
                "image_m": torch.stack(image_ms, dim=0),
                "image_l": torch.stack(image_ls, dim=0),
            },
            info=dict(mask_path={f"frame_{i}": p for i, p in enumerate(mask_paths)}, group_name=group_name),
        )

    def __len__(self):
        return len(self.total_data_paths)


class VideoTrainDataset(data.Dataset):
    def __init__(self, dataset_infos: dict, shape: dict, num_frames: int = 1):
        super().__init__()
        self.shape = shape
        self.num_frames = num_frames
        self.stride = num_frames - 1 if num_frames > 1 else 1

        self.total_data_paths = []
        for dataset_name, dataset_info in dataset_infos.items():
            image_path = os.path.join(dataset_info["root"], dataset_info["image"]["path"])
            image_suffix = dataset_info["image"]["suffix"]
            mask_path = os.path.join(dataset_info["root"], dataset_info["mask"]["path"])
            mask_suffix = dataset_info["mask"]["suffix"]

            image_group_paths = sorted(glob.glob(image_path))
            mask_group_paths = sorted(glob.glob(mask_path))
            group_name_place = image_path.find("*")

            data_paths = []
            for image_group_path, mask_group_path in zip(image_group_paths, mask_group_paths):
                group_name = image_group_path[group_name_place:].split("/")[0]
                image_names = [
                    p[: -len(image_suffix)] for p in sorted(os.listdir(image_group_path)) if p.endswith(image_suffix)
                ]
                mask_names = [
                    p[: -len(mask_suffix)] for p in sorted(os.listdir(mask_group_path)) if p.endswith(mask_suffix)
                ]
                valid_names = sorted(
                    set(image_names).intersection(mask_names), key=lambda item: get_number_from_tail(item)
                )

                length_of_sequence = len(valid_names)
                for clip_idx in range(0, length_of_sequence, self.stride):
                    start_idx = clip_idx * self.stride
                    if start_idx + num_frames > length_of_sequence:
                        start_idx = length_of_sequence - num_frames

                    clip_info = []
                    for i, n in enumerate(valid_names[start_idx : start_idx + num_frames]):
                        clip_info.append(
                            (
                                os.path.join(image_group_path, n) + image_suffix,
                                os.path.join(mask_group_path, n) + mask_suffix,
                                group_name,
                                i,
                                clip_idx,
                            )
                        )
                    if len(clip_info) < num_frames:
                        times, last = divmod(num_frames, len(clip_info))
                        clip_info.extend(clip_info * (times - 1))
                        clip_info.extend(clip_info[:last])
                    data_paths.append(clip_info)

            LOGGER.info(f"Length of {dataset_name}: {len(data_paths)}")
            self.total_data_paths.extend(data_paths)

        self.frame_specific_transformation = construct_frame_transform()
        self.frame_share_transformation = construct_video_transform()

    def __getitem__(self, index):
        base_h = self.shape["h"]
        base_w = self.shape["w"]
        image_ss = []
        image_ms = []
        image_ls = []
        masks = []
        for image_path, mask_path, _, idx_in_group, _ in self.total_data_paths[index]:
            image = io.read_color_array(image_path)
            mask = io.read_gray_array(mask_path, thr=0)
            if image.shape[:2] != mask.shape:
                h, w = mask.shape
                image = ops.resize(image, height=h, width=w)

            if idx_in_group == 0:
                shared_transformed = self.frame_share_transformation(image=image, mask=mask)
            else:
                shared_transformed = A.ReplayCompose.replay(
                    saved_augmentations=shared_transformed["replay"], image=image, mask=mask
                )
            specific_transformed = self.frame_specific_transformation(
                image=shared_transformed["image"], mask=shared_transformed["mask"]
            )
            image = specific_transformed["image"]
            mask = specific_transformed["mask"]

            images = ops.ms_resize(image, scales=(0.5, 1.0, 1.5), base_h=base_h, base_w=base_w)
            mask = ops.resize(mask, height=base_h, width=base_w)

            image_ss.append(torch.from_numpy(images[0]).div(255).permute(2, 0, 1))
            image_ms.append(torch.from_numpy(images[1]).div(255).permute(2, 0, 1))
            image_ls.append(torch.from_numpy(images[2]).div(255).permute(2, 0, 1))
            masks.append(torch.from_numpy(mask).unsqueeze(0))

        return dict(
            data={
                "image_s": torch.stack(image_ss, dim=0),
                "image_m": torch.stack(image_ms, dim=0),
                "image_l": torch.stack(image_ls, dim=0),
                "mask": torch.stack(masks, dim=0),
            }
        )

    def __len__(self):
        return len(self.total_data_paths)


class Evaluator:
    def __init__(self, device, metric_names, clip_range=None):
        self.device = device
        self.clip_range = clip_range
        self.metric_names = metric_names

    @torch.no_grad()
    def eval(self, model, data_loader, save_path=""):
        model.eval()
        all_metrics = recorder.GroupedMetricRecorder(metric_names=self.metric_names)

        num_frames = data_loader.dataset.num_frames
        for batch in tqdm(data_loader, total=len(data_loader), ncols=79, desc="[EVAL]"):
            batch_images = pt_utils.to_device(batch["data"], device=self.device)
            batch_images = {k: v.flatten(0, 1) for k, v in batch_images.items()}  # B_T,C,H,W
            logits = model(data=batch_images)  # BT,1,H,W
            probs = logits.sigmoid().squeeze(1).cpu().detach()
            probs = probs - probs.min()
            probs = probs / (probs.max() + 1e-8)
            probs = torch.unflatten(probs, dim=0, sizes=(-1, num_frames))
            probs = probs.numpy()

            mask_paths = batch["info"]["mask_path"]
            group_names = batch["info"]["group_name"]
            for clip_idx, clip_pred in enumerate(probs):
                for frame_idx, pred in enumerate(clip_pred):
                    mask_path = mask_paths[f"frame_{frame_idx}"][clip_idx]
                    mask_array = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask_array[mask_array > 0] = 255
                    mask_h, mask_w = mask_array.shape
                    pred = ops.resize(pred, height=mask_h, width=mask_w)

                    if self.clip_range is not None:
                        pred = ops.clip_to_normalize(pred, clip_range=self.clip_range)

                    group_name = group_names[clip_idx]
                    if save_path:  # 这里的save_path包含了数据集名字
                        ops.save_array_as_image(
                            data_array=pred,
                            save_name=os.path.basename(mask_path),
                            save_dir=os.path.join(save_path, group_name),
                        )

                    pred = (pred * 255).astype(np.uint8)
                    all_metrics.step(group_name=group_name, pre=pred, gt=mask_array, gt_path=mask_path)
        seg_results, group_seg_results = all_metrics.show(return_group=True)
        return seg_results, group_seg_results


def test(model, cfg):
    test_wrapper = Evaluator(device=cfg.device, metric_names=cfg.metric_names, clip_range=cfg.test.clip_range)

    for te_name in cfg.test.data.names:
        te_info = cfg.dataset_infos[te_name]
        te_dataset = VideoTestDataset(dataset_info=te_info, shape=cfg.test.data.shape, num_frames=cfg.num_frames)
        te_loader = data.DataLoader(
            dataset=te_dataset, batch_size=cfg.test.batch_size, num_workers=cfg.test.num_workers, pin_memory=True
        )
        LOGGER.info(f"Testing with testset: {te_name}: {len(te_dataset)}")

        if cfg.save_results:
            save_path = os.path.join(cfg.path.save, te_name)
            LOGGER.info(f"Results will be saved into {save_path}")
        else:
            save_path = ""

        seg_results, group_seg_results = test_wrapper.eval(model=model, data_loader=te_loader, save_path=save_path)
        seg_results_str = ", ".join([f"{k}: {v:.03f}" for k, v in seg_results.items()])
        LOGGER.info(f"({te_name}): {py_utils.mapping_to_str(te_info)}\n{seg_results_str}")

        max_group_name_length = max([len(n) for n in group_seg_results.keys()])
        for group_name, group_results in group_seg_results.items():
            metric_str = ""
            for metric_name, metric_value in group_results.items():
                metric_str += "|" + metric_name + " " + str(metric_value).ljust(6)
            LOGGER.info(f"{group_name.rjust(max_group_name_length)}: {metric_str}")


def train(model, cfg):
    tr_dataset = VideoTrainDataset(
        dataset_infos={data_name: cfg.dataset_infos[data_name] for data_name in cfg.train.data.names},
        shape=cfg.train.data.shape,
        num_frames=cfg.num_frames,
    )
    LOGGER.info(f"Total Length of Video Trainset: {len(tr_dataset)}")

    tr_loader = data.DataLoader(
        dataset=tr_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=pt_utils.customized_worker_init_fn if cfg.use_custom_worker_init else None,
    )

    counter = recorder.TrainingCounter(
        epoch_length=len(tr_loader),
        epoch_based=cfg.train.epoch_based,
        num_epochs=cfg.train.num_epochs,
        num_total_iters=cfg.train.num_iters,
    )
    optimizer = pipeline.construct_optimizer(
        model=model,
        initial_lr=cfg.train.lr,
        mode=cfg.train.optimizer.mode,
        group_mode=cfg.train.optimizer.group_mode,
        cfg=cfg.train.optimizer.cfg,
    )
    scheduler = pipeline.Scheduler(
        optimizer=optimizer,
        num_iters=counter.num_total_iters,
        epoch_length=counter.num_inner_iters,
        scheduler_cfg=cfg.train.scheduler,
        step_by_batch=cfg.train.sche_usebatch,
    )
    scheduler.record_lrs(param_groups=optimizer.param_groups)
    scheduler.plot_lr_coef_curve(save_path=cfg.path.pth_log)
    scaler = pipeline.Scaler(optimizer, cfg.train.use_amp, set_to_none=cfg.train.optimizer.set_to_none)

    LOGGER.info(f"Scheduler:\n{scheduler}\nOptimizer:\n{optimizer}")

    loss_recorder = recorder.HistoryBuffer()
    iter_time_recorder = recorder.HistoryBuffer()

    LOGGER.info(f"Image Mean: {model.normalizer.mean.flatten()}, Image Std: {model.normalizer.std.flatten()}")
    if cfg.train.bn.freeze_encoder:
        LOGGER.info(" >>> Freeze Backbone !!! <<< ")
        model.encoder.requires_grad_(False)

    train_start_time = time.perf_counter()
    for _ in range(counter.num_epochs):
        LOGGER.info(f"Exp_Name: {cfg.exp_name}")

        model.train()
        if cfg.train.bn.freeze_status:
            pt_utils.frozen_bn_stats(model.encoder, freeze_affine=cfg.train.bn.freeze_affine)

        # an epoch starts
        for batch_idx, batch in enumerate(tr_loader):
            iter_start_time = time.perf_counter()
            scheduler.step(curr_idx=counter.curr_iter)  # update learning rate

            data_batch = pt_utils.to_device(data=batch["data"], device=cfg.device)
            data_batch = {k: v.flatten(0, 1) for k, v in data_batch.items()}
            with torch.cuda.amp.autocast(enabled=cfg.train.use_amp):
                outputs = model(data=data_batch, iter_percentage=counter.curr_percent)

            loss = outputs["loss"]
            loss_str = outputs["loss_str"]
            loss = loss / cfg.train.grad_acc_step
            scaler.calculate_grad(loss=loss)
            if counter.every_n_iters(cfg.train.grad_acc_step):  # Accumulates scaled gradients.
                scaler.update_grad()

            item_loss = loss.item()
            data_shape = tuple(data_batch["mask"].shape)
            loss_recorder.update(value=item_loss, num=data_shape[0])

            if cfg.log_interval > 0 and (
                counter.every_n_iters(cfg.log_interval)
                or counter.is_first_inner_iter()
                or counter.is_last_inner_iter()
                or counter.is_last_total_iter()
            ):
                gpu_mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                eta_seconds = iter_time_recorder.avg * (counter.num_total_iters - counter.curr_iter - 1)
                eta_string = f"ETA: {datetime.timedelta(seconds=int(eta_seconds))}"
                progress = (
                    f"{counter.curr_iter}:{counter.num_total_iters} "
                    f"{batch_idx}/{counter.num_inner_iters} "
                    f"{counter.curr_epoch}/{counter.num_epochs}"
                )
                loss_info = f"{loss_str} (M:{loss_recorder.global_avg:.5f}/C:{item_loss:.5f})"
                lr_info = f"LR: {optimizer.lr_string()}"
                LOGGER.info(f"{eta_string}({gpu_mem}) | {progress} | {lr_info} | {loss_info} | {data_shape}")
                cfg.tb_logger.write_to_tb("lr", optimizer.lr_groups(), counter.curr_iter)
                cfg.tb_logger.write_to_tb("iter_loss", item_loss, counter.curr_iter)
                cfg.tb_logger.write_to_tb("avg_loss", loss_recorder.global_avg, counter.curr_iter)

            if counter.curr_iter < 3:  # plot some batches of the training phase
                recorder.plot_results(
                    dict(img=data_batch["image_m"], msk=data_batch["mask"], **outputs["vis"]),
                    save_path=os.path.join(cfg.path.pth_log, "img", f"iter_{counter.curr_iter}.png"),
                )

            iter_time_recorder.update(value=time.perf_counter() - iter_start_time)
            if counter.is_last_total_iter():
                break
            counter.update_iter_counter()

        # an epoch ends
        recorder.plot_results(
            dict(img=data_batch["image_m"], msk=data_batch["mask"], **outputs["vis"]),
            save_path=os.path.join(cfg.path.pth_log, "img", f"epoch_{counter.curr_epoch}.png"),
        )
        io.save_weight(model=model, save_path=cfg.path.final_state_net)
        counter.update_epoch_counter()

    cfg.tb_logger.close_tb()
    io.save_weight(model=model, save_path=cfg.path.final_state_net)

    total_train_time = time.perf_counter() - train_start_time
    total_other_time = datetime.timedelta(seconds=int(total_train_time - iter_time_recorder.global_sum))
    LOGGER.info(
        f"Total Training Time: {datetime.timedelta(seconds=int(total_train_time))} ({total_other_time} on others)"
    )


def parse_cfg():
    parser = argparse.ArgumentParser("Training and evaluation script")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--data-cfg", type=str, default="./dataset.yaml")
    parser.add_argument("--model-name", type=str, choices=model_zoo.__dict__.keys())
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--load-from", type=str)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument(
        "--metric-names",
        nargs="+",
        type=str,
        default=["sm", "wfm", "mae", "em", "fmeasure", "iou", "dice"],
        choices=recorder.GroupedMetricRecorder.supported_metrics,
    )
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--save-results", action="store_true")
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--info", type=str)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(vars(args))

    with open(cfg.data_cfg, mode="r") as f:
        cfg.dataset_infos = yaml.safe_load(f)

    cfg.proj_root = os.path.dirname(os.path.abspath(__file__))
    cfg.exp_name = py_utils.construct_exp_name(model_name=cfg.model_name, cfg=cfg)
    cfg.output_dir = os.path.join(cfg.proj_root, cfg.output_dir)
    cfg.path = py_utils.construct_path(output_dir=cfg.output_dir, exp_name=cfg.exp_name)
    cfg.device = "cuda:0"

    py_utils.pre_mkdir(cfg.path)
    with open(cfg.path.cfg_copy, encoding="utf-8", mode="w") as f:
        f.write(cfg.pretty_text)
    shutil.copy(__file__, cfg.path.trainer_copy)

    file_handler = logging.FileHandler(cfg.path.log)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("[%(filename)s] %(message)s"))
    LOGGER.addHandler(file_handler)
    LOGGER.info(cfg.pretty_text)

    cfg.tb_logger = recorder.TBLogger(tb_root=cfg.path.tb)
    return cfg


def main():
    cfg = parse_cfg()
    pt_utils.initialize_seed_cudnn(seed=cfg.base_seed, deterministic=cfg.deterministic)

    model_class = model_zoo.__dict__.get(cfg.model_name)
    assert model_class is not None, "Please check your --model-name"
    model_code = inspect.getsource(model_class)
    model = model_class(num_frames=cfg.num_frames, pretrained=cfg.pretrained, use_checkpoint=cfg.use_checkpoint)
    LOGGER.info(model_code)
    model.to(cfg.device)

    if cfg.load_from:
        io.load_weight(model=model, load_path=cfg.load_from, strict=True)

    LOGGER.info(f"Number of Parameters: {sum((v.numel() for v in model.parameters(recurse=True)))}")
    if not cfg.evaluate:
        train(model=model, cfg=cfg)

    if cfg.evaluate or cfg.has_test:
        io.save_weight(model=model, save_path=cfg.path.final_state_net)
        test(model=model, cfg=cfg)

    LOGGER.info("End training...")


if __name__ == "__main__":
    main()
