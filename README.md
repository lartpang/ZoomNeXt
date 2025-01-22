# ZoomNeXt: A Unified Collaborative Pyramid Network for Camouflaged Object Detection (TPAMI 2024)

<div align="center">
  <img src="https://github.com/lartpang/ZoomNeXt/assets/26847524/f43f773b-a81f-4c64-a809-9764b53dd52c" alt="Logo">
</div>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/zoomnext-a-unified-collaborative-pyramid/camouflaged-object-segmentation-on-camo)](https://paperswithcode.com/sota/camouflaged-object-segmentation-on-camo?p=zoomnext-a-unified-collaborative-pyramid) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/zoomnext-a-unified-collaborative-pyramid/camouflaged-object-segmentation-on-chameleon)](https://paperswithcode.com/sota/camouflaged-object-segmentation-on-chameleon?p=zoomnext-a-unified-collaborative-pyramid) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/zoomnext-a-unified-collaborative-pyramid/camouflaged-object-segmentation-on-cod)](https://paperswithcode.com/sota/camouflaged-object-segmentation-on-cod?p=zoomnext-a-unified-collaborative-pyramid) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/zoomnext-a-unified-collaborative-pyramid/camouflaged-object-segmentation-on-nc4k)](https://paperswithcode.com/sota/camouflaged-object-segmentation-on-nc4k?p=zoomnext-a-unified-collaborative-pyramid) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/zoomnext-a-unified-collaborative-pyramid/camouflaged-object-segmentation-on-moca-mask)](https://paperswithcode.com/sota/camouflaged-object-segmentation-on-moca-mask?p=zoomnext-a-unified-collaborative-pyramid) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/zoomnext-a-unified-collaborative-pyramid/camouflaged-object-segmentation-on)](https://paperswithcode.com/sota/camouflaged-object-segmentation-on?p=zoomnext-a-unified-collaborative-pyramid)


```bibtex
@ARTICLE {ZoomNeXt,
    title   = {ZoomNeXt: A Unified Collaborative Pyramid Network for Camouflaged Object Detection},
    author  ={Youwei Pang and Xiaoqi Zhao and Tian-Zhu Xiang and Lihe Zhang and Huchuan Lu},
    journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
    year    = {2024},
    doi     = {10.1109/TPAMI.2024.3417329},
}
```

## Weights and Results

See [Google Drive](https://drive.google.com/drive/folders/1Hp3GIqossOrJYs3bRzJICujMKbZy4WxO?usp=drive_link).

### Performance

| Backbone        | CAMO-TE |                      |       | CHAMELEON |                      |       | COD10K-TE |                      |       | NC4K  |                      |       |
| --------------- | ------- | -------------------- | ----- | --------- | -------------------- | ----- | --------- | -------------------- | ----- | ----- | -------------------- | ----- |
|                 | $S_m$   | $F^{\omega}_{\beta}$ | MAE   | $S_m$     | $F^{\omega}_{\beta}$ | MAE   | $S_m$     | $F^{\omega}_{\beta}$ | MAE   | $S_m$ | $F^{\omega}_{\beta}$ | MAE   |
| ResNet-50       | 0.833   | 0.774                | 0.065 | 0.908     | 0.858                | 0.021 | 0.861     | 0.768                | 0.026 | 0.874 | 0.816                | 0.037 |
| EfficientNet-B1 | 0.848   | 0.803                | 0.056 | 0.916     | 0.870                | 0.020 | 0.863     | 0.773                | 0.024 | 0.876 | 0.823                | 0.036 |
| EfficientNet-B4 | 0.867   | 0.824                | 0.046 | 0.911     | 0.865                | 0.020 | 0.875     | 0.797                | 0.021 | 0.884 | 0.837                | 0.032 |
| PVTv2-B2        | 0.874   | 0.839                | 0.047 | 0.922     | 0.884                | 0.017 | 0.887     | 0.818                | 0.019 | 0.892 | 0.852                | 0.030 |
| PVTv2-B3        | 0.885   | 0.854                | 0.042 | 0.927     | 0.898                | 0.017 | 0.895     | 0.829                | 0.018 | 0.900 | 0.861                | 0.028 |
| PVTv2-B4        | 0.888   | 0.859                | 0.040 | 0.925     | 0.897                | 0.016 | 0.898     | 0.838                | 0.017 | 0.900 | 0.865                | 0.028 |
| PVTv2-B5        | 0.889   | 0.857                | 0.041 | 0.924     | 0.885                | 0.018 | 0.898     | 0.827                | 0.018 | 0.903 | 0.863                | 0.028 |

| Backbone       | CAD   |                      |       |       |       | MoCA-Mask-TE |                      |       |       |       |
| -------------- | ----- | -------------------- | ----- | ----- | ----- | ------------ | -------------------- | ----- | ----- | ----- |
|                | $S_m$ | $F^{\omega}_{\beta}$ | MAE   | mDice | mIoU  | $S_m$        | $F^{\omega}_{\beta}$ | MAE   | mDice | mIoU  |
| PVTv2-B5 (T=5) | 0.757 | 0.593                | 0.020 | 0.599 | 0.510 | 0.734        | 0.476                | 0.010 | 0.497 | 0.422 |

## Prepare Data

> Based on the following data setup, the performance of the VCOD dataset evaluated directly using the training script is now consistent with the paper.

Set all dataset information to the `dataset.yaml` as follows.

<details>
<summary>
Example of the config file (dataset.yaml):
</summary>

```yaml
# VCOD Datasets
moca_mask_tr:
  {
    root: "YOUR-VCOD-DATASETS-ROOT/MoCA-Mask/MoCA_Video/TrainDataset_per_sq",
    image: { path: "*/Imgs", suffix: ".jpg" },
    mask: { path: "*/GT", suffix: ".png" },
    start_idx: 0,
    end_idx: 0
  }
moca_mask_te:
  {
    root: "YOUR-VCOD-DATASETS-ROOT/MoCA-Mask/MoCA_Video/TestDataset_per_sq",
    image: { path: "*/Imgs", suffix: ".jpg" },
    mask: { path: "*/GT", suffix: ".png" },
    start_idx: 0,
    end_idx: -2
  }
cad:
  {
    root: "YOUR-VCOD-DATASETS-ROOT/CamouflagedAnimalDataset",
    image: { path: "original_data/*/frames", suffix: ".png" },
    mask: { path: "converted_mask/*/groundtruth", suffix: ".png" },
    start_idx: 0,
    end_idx: 0
  }

# ICOD Datasets
cod10k_tr:
  {
    root: "YOUR-ICOD-DATASETS-ROOT/Train/COD10K-TR",
    image: { path: "Image", suffix: ".jpg" },
    mask: { path: "Mask", suffix: ".png" },
  }
camo_tr:
  {
    root: "YOUR-ICOD-DATASETS-ROOT/Train/CAMO-TR",
    image: { path: "Image", suffix: ".jpg" },
    mask: { path: "Mask", suffix: ".png" },
  }
cod10k_te:
  {
    root: "YOUR-ICOD-DATASETS-ROOT/Test/COD10K-TE",
    image: { path: "Image", suffix: ".jpg" },
    mask: { path: "Mask", suffix: ".png" },
  }
camo_te:
  {
    root: "YOUR-ICOD-DATASETS-ROOT/Test/CAMO-TE",
    image: { path: "Image", suffix: ".jpg" },
    mask: { path: "Mask", suffix: ".png" },
  }
chameleon:
  {
    root: "YOUR-ICOD-DATASETS-ROOT/Test/CHAMELEON",
    image: { path: "Image", suffix: ".jpg" },
    mask: { path: "Mask", suffix: ".png" },
  }
nc4k:
  {
    root: "YOUR-ICOD-DATASETS-ROOT/Test/NC4K",
    image: { path: "Imgs", suffix: ".jpg" },
    mask: { path: "GT", suffix: ".png" },
  }
```

</details>

## Install Requirements

* torch==2.1.2
* torchvision==0.16.2
* Others: `pip install -r requirements.txt`

## Evaluation

```shell
# ICOD
python main_for_image.py --config configs/icod_train.py --model-name <MODEL_NAME> --evaluate --load-from <TRAINED_WEIGHT>
# VCOD
python main_for_video.py --config configs/vcod_finetune.py --model-name <MODEL_NAME> --evaluate --load-from <TRAINED_WEIGHT>
```

## Training

### Image Camouflaged Object Detection

```shell
python main_for_image.py --config configs/icod_train.py --pretrained --model-name EffB1_ZoomNeXt
python main_for_image.py --config configs/icod_train.py --pretrained --model-name EffB4_ZoomNeXt
python main_for_image.py --config configs/icod_train.py --pretrained --model-name PvtV2B2_ZoomNeXt
python main_for_image.py --config configs/icod_train.py --pretrained --model-name PvtV2B3_ZoomNeXt
python main_for_image.py --config configs/icod_train.py --pretrained --model-name PvtV2B4_ZoomNeXt
python main_for_image.py --config configs/icod_train.py --pretrained --model-name PvtV2B5_ZoomNeXt
python main_for_image.py --config configs/icod_train.py --pretrained --model-name RN50_ZoomNeXt
```

### Video Camouflaged Object Detection

1. Pretrain on COD10K-TR: `python main_for_image.py --config configs/icod_pretrain.py --info pretrain --model-name PvtV2B5_ZoomNeXt --pretrained`
2. Finetune on MoCA-Mask-TR: `python main_for_video.py --config configs/vcod_finetune.py --info finetune --model-name videoPvtV2B5_ZoomNeXt --load-from <PRETAINED_WEIGHT>`

> [!note]
> If you meets the OOM problem, you can try to reduce the batch size or switch on the `--use-checkpoint` flag:
> `python main_for_image.py/main_for_video.py <your config> --use-checkpoint`
