# -*- coding: utf-8 -*-
import argparse
import datetime
import inspect
import logging
import os
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


class ImageTestDataset(data.Dataset):
    def __init__(self, dataset_info: dict, shape: dict):
        super().__init__()
        self.shape = shape

        image_path = os.path.join(dataset_info["root"], dataset_info["image"]["path"])
        image_suffix = dataset_info["image"]["suffix"]
        mask_path = os.path.join(dataset_info["root"], dataset_info["mask"]["path"])
        mask_suffix = dataset_info["mask"]["suffix"]

        image_names = [p[: -len(image_suffix)] for p in sorted(os.listdir(image_path)) if p.endswith(image_suffix)]
        mask_names = [p[: -len(mask_suffix)] for p in sorted(os.listdir(mask_path)) if p.endswith(mask_suffix)]
        valid_names = sorted(set(image_names).intersection(mask_names))
        self.total_data_paths = [
            (os.path.join(image_path, n) + image_suffix, os.path.join(mask_path, n) + mask_suffix) for n in valid_names
        ]

    def __getitem__(self, index):
        image_path, mask_path = self.total_data_paths[index]
        image = io.read_color_array(image_path)

        base_h = self.shape["h"]
        base_w = self.shape["w"]

        images = ops.ms_resize(image, scales=(0.5, 1.0, 1.5), base_h=base_h, base_w=base_w)
        image_s = torch.from_numpy(images[0]).div(255).permute(2, 0, 1)
        image_m = torch.from_numpy(images[1]).div(255).permute(2, 0, 1)
        image_l = torch.from_numpy(images[2]).div(255).permute(2, 0, 1)

        return dict(
            data={"image_s": image_s, "image_m": image_m, "image_l": image_l},
            info=dict(mask_path=mask_path, group_name="image"),
        )

    def __len__(self):
        return len(self.total_data_paths)


class ImageTrainDataset(data.Dataset):
    def __init__(self, dataset_infos: dict, shape: dict):
        super().__init__()
        self.shape = shape

        self.total_data_paths = []
        for dataset_name, dataset_info in dataset_infos.items():
            image_path = os.path.join(dataset_info["root"], dataset_info["image"]["path"])
            image_suffix = dataset_info["image"]["suffix"]
            mask_path = os.path.join(dataset_info["root"], dataset_info["mask"]["path"])
            mask_suffix = dataset_info["mask"]["suffix"]

            image_names = [p[: -len(image_suffix)] for p in sorted(os.listdir(image_path)) if p.endswith(image_suffix)]
            mask_names = [p[: -len(mask_suffix)] for p in sorted(os.listdir(mask_path)) if p.endswith(mask_suffix)]
            valid_names = sorted(set(image_names).intersection(mask_names))
            data_paths = [
                (os.path.join(image_path, n) + image_suffix, os.path.join(mask_path, n) + mask_suffix)
                for n in valid_names
            ]
            LOGGER.info(f"Length of {dataset_name}: {len(data_paths)}")
            self.total_data_paths.extend(data_paths)

        self.trains = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=90, p=0.5, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REPLICATE),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.5),
            ]
        )

    def __getitem__(self, index):
        image_path, mask_path = self.total_data_paths[index]
        image = io.read_color_array(image_path)
        mask = io.read_gray_array(mask_path, thr=0)
        if image.shape[:2] != mask.shape:
            h, w = mask.shape
            image = ops.resize(image, height=h, width=w)

        transformed = self.trains(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]

        base_h = self.shape["h"]
        base_w = self.shape["w"]

        images = ops.ms_resize(image, scales=(0.5, 1.0, 1.5), base_h=base_h, base_w=base_w)
        image_s = torch.from_numpy(images[0]).div(255).permute(2, 0, 1)
        image_m = torch.from_numpy(images[1]).div(255).permute(2, 0, 1)
        image_l = torch.from_numpy(images[2]).div(255).permute(2, 0, 1)

        mask = ops.resize(mask, height=base_h, width=base_w)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return dict(
            data={
                "image_s": image_s,
                "image_m": image_m,
                "image_l": image_l,
                "mask": mask,
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

        for batch in tqdm(data_loader, total=len(data_loader), ncols=79, desc="[EVAL]"):
            batch_images = pt_utils.to_device(batch["data"], device=self.device)
            logits = model(data=batch_images)  # B,1,H,W
            probs = logits.sigmoid().squeeze(1).cpu().detach().numpy()
            probs = probs - probs.min()
            probs = probs / (probs.max() + 1e-8)

            mask_paths = batch["info"]["mask_path"]
            group_names = batch["info"]["group_name"]
            for pred_idx, pred in enumerate(probs):
                mask_path = mask_paths[pred_idx]
                mask_array = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask_array[mask_array > 0] = 255
                mask_h, mask_w = mask_array.shape
                pred = ops.resize(pred, height=mask_h, width=mask_w)

                if self.clip_range is not None:
                    pred = ops.clip_to_normalize(pred, clip_range=self.clip_range)

                group_name = group_names[pred_idx]
                if save_path:  # 这里的save_path包含了数据集名字
                    ops.save_array_as_image(
                        data_array=pred,
                        save_name=os.path.basename(mask_path),
                        save_dir=os.path.join(save_path, group_name),
                    )

                pred = (pred * 255).astype(np.uint8)
                all_metrics.step(group_name=group_name, pre=pred, gt=mask_array, gt_path=mask_path)
        return all_metrics.show()


def test(model, cfg):
    test_wrapper = Evaluator(device=cfg.device, metric_names=cfg.metric_names, clip_range=cfg.test.clip_range)

    for te_name in cfg.test.data.names:
        te_info = cfg.dataset_infos[te_name]
        te_dataset = ImageTestDataset(dataset_info=te_info, shape=cfg.test.data.shape)
        te_loader = data.DataLoader(
            dataset=te_dataset, batch_size=cfg.test.batch_size, num_workers=cfg.test.num_workers, pin_memory=True
        )
        LOGGER.info(f"Testing with testset: {te_name}: {len(te_dataset)}")

        if cfg.save_results:
            save_path = os.path.join(cfg.path.save, te_name)
            LOGGER.info(f"Results will be saved into {save_path}")
        else:
            save_path = ""

        seg_results = test_wrapper.eval(model=model, data_loader=te_loader, save_path=save_path)
        seg_results_str = ", ".join([f"{k}: {v:.03f}" for k, v in seg_results.items()])
        LOGGER.info(f"({te_name}): {py_utils.mapping_to_str(te_info)}\n{seg_results_str}")


def train(model, cfg):
    tr_dataset = ImageTrainDataset(
        dataset_infos={data_name: cfg.dataset_infos[data_name] for data_name in cfg.train.data.names},
        shape=cfg.train.data.shape,
    )
    LOGGER.info(f"Total Length of Image Trainset: {len(tr_dataset)}")

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
        default=["sm", "wfm", "mae", "em", "fmeasure"],
        choices=recorder.GroupedMetricRecorder.supported_metrics,
    )
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--save-results", action="store_true")
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
    model = model_class(num_frames=1, pretrained=cfg.pretrained)
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
