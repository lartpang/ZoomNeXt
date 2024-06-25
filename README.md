# ZoomNeXt: A Unified Collaborative Pyramid Network for Camouflaged Object Detection (TPAMI 2024)

<div align="center">
  <img src="https://github.com/lartpang/ZoomNeXt/assets/26847524/f43f773b-a81f-4c64-a809-9764b53dd52c" alt="Logo">
</div>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/zoomnext-a-unified-collaborative-pyramid/camouflaged-object-segmentation-on-camo)](https://paperswithcode.com/sota/camouflaged-object-segmentation-on-camo?p=zoomnext-a-unified-collaborative-pyramid) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/zoomnext-a-unified-collaborative-pyramid/camouflaged-object-segmentation-on-chameleon)](https://paperswithcode.com/sota/camouflaged-object-segmentation-on-chameleon?p=zoomnext-a-unified-collaborative-pyramid) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/zoomnext-a-unified-collaborative-pyramid/camouflaged-object-segmentation-on-cod)](https://paperswithcode.com/sota/camouflaged-object-segmentation-on-cod?p=zoomnext-a-unified-collaborative-pyramid) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/zoomnext-a-unified-collaborative-pyramid/camouflaged-object-segmentation-on-nc4k)](https://paperswithcode.com/sota/camouflaged-object-segmentation-on-nc4k?p=zoomnext-a-unified-collaborative-pyramid) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/zoomnext-a-unified-collaborative-pyramid/camouflaged-object-segmentation-on-moca-mask)](https://paperswithcode.com/sota/camouflaged-object-segmentation-on-moca-mask?p=zoomnext-a-unified-collaborative-pyramid) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/zoomnext-a-unified-collaborative-pyramid/camouflaged-object-segmentation-on)](https://paperswithcode.com/sota/camouflaged-object-segmentation-on?p=zoomnext-a-unified-collaborative-pyramid)


```bibtex
@misc{ZoomNeXt,
      title={ZoomNeXt: A Unified Collaborative Pyramid Network for Camouflaged Object Detection},
      author={Youwei Pang and Xiaoqi Zhao and Tian-Zhu Xiang and Lihe Zhang and Huchuan Lu},
      year={2023},
      eprint={2310.20208},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Weights and Results

### Results

| Datasets                            | Links                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CAMO-TE, CHAMELEON, COD10K-TE, NC4K | [ResNet-50](https://github.com/lartpang/ZoomNeXt/releases/download/prediction-v0.1/zoomnext_res50.7z), [EfficientNet-B4](https://github.com/lartpang/ZoomNeXt/releases/download/prediction-v0.1/zoomnext_efficientb4.7z), [PVTv2-B2](https://github.com/lartpang/ZoomNeXt/releases/download/prediction-v0.1/zoomnext_pvtv2b2.7z), [PVTv2-B3](https://github.com/lartpang/ZoomNeXt/releases/download/prediction-v0.1/zoomnext_pvtv2b3.7z), [PVTv2-B4](https://github.com/lartpang/ZoomNeXt/releases/download/prediction-v0.1/zoomnext_pvtv2b4.7z), [PVTv2-B5](https://github.com/lartpang/ZoomNeXt/releases/download/prediction-v0.1/zoomnext_pvtv2b5.7z) |
| CAD, MoCA-Mask-TE                   | [PVTv2-B5](https://github.com/lartpang/ZoomNeXt/releases/download/prediction-v0.1/zoomnext_pvtv2b5_video.7z)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |

### Weights

| Backbone         | CAMO-TE |                      |       | CHAMELEON |                      |       | COD10K-TE |                      |       | NC4K  |                      |       | Links                                                                                                   |
| ---------------- | ------- | -------------------- | ----- | --------- | -------------------- | ----- | --------- | -------------------- | ----- | ----- | -------------------- | ----- | ------------------------------------------------------------------------------------------------------- |
|                  | $S_m$   | $F^{\omega}_{\beta}$ | MAE   | $S_m$     | $F^{\omega}_{\beta}$ | MAE   | $S_m$     | $F^{\omega}_{\beta}$ | MAE   | $S_m$ | $F^{\omega}_{\beta}$ | MAE   |                                                                                                         |
| ResNet-50        | 0.833   | 0.774                | 0.065 | 0.908     | 0.858                | 0.021 | 0.861     | 0.768                | 0.026 | 0.874 | 0.816                | 0.037 | [Weight](https://github.com/lartpang/ZoomNeXt/releases/download/weights-v0.2/resnet50-zoomnext.pth)     |
| EfficientNet-B4  | 0.867   | 0.824                | 0.046 | 0.911     | 0.865                | 0.020 | 0.875     | 0.797                | 0.021 | 0.884 | 0.837                | 0.032 | [Weight](https://github.com/lartpang/ZoomNeXt/releases/download/weights-v0.2/eff-b4-zoomnext.pth)       |
| PVTv2-B2         | 0.874   | 0.839                | 0.047 | 0.922     | 0.884                | 0.017 | 0.887     | 0.818                | 0.019 | 0.892 | 0.852                | 0.030 | [Weight](https://github.com/lartpang/ZoomNeXt/releases/download/weights-v0.2/pvtv2-b2-zoomnext.pth)     |
| PVTv2-B3         | 0.885   | 0.854                | 0.042 | 0.927     | 0.898                | 0.017 | 0.895     | 0.829                | 0.018 | 0.900 | 0.861                | 0.028 | [Weight](https://github.com/lartpang/ZoomNeXt/releases/download/weights-v0.2/pvtv2-b3-zoomnext.pth)     |
| PVTv2-B4         | 0.888   | 0.859                | 0.040 | 0.925     | 0.897                | 0.016 | 0.898     | 0.838                | 0.017 | 0.900 | 0.865                | 0.028 | [Weight](https://github.com/lartpang/ZoomNeXt/releases/download/weights-v0.2/pvtv2-b4-zoomnext.pth)     |
| PVTv2-B5         | 0.889   | 0.857                | 0.041 | 0.924     | 0.885                | 0.018 | 0.898     | 0.827                | 0.018 | 0.903 | 0.863                | 0.028 | [Weight](https://github.com/lartpang/ZoomNeXt/releases/download/weights-v0.2/pvtv2-b5-zoomnext.pth)     |
| EfficientNet-B1  |         |                      |       |           |                      |       |           |                      |       |       |                      |       | [Weight](https://github.com/lartpang/ZoomNeXt/releases/download/weights-v0.2/eff-b1-zoomnext.pth)       |
| ConvNeXtV2-Large |         |                      |       |           |                      |       |           |                      |       |       |                      |       | [Weight](https://github.com/lartpang/ZoomNeXt/releases/download/weights-v0.2/convnextv2-l-zoomnext.pth) |

| Backbone        | CAD   |                      |       |       |       | MoCA-Mask-TE |                      |       |       |       | Links                                                                                                      |
| --------------- | ----- | -------------------- | ----- | ----- | ----- | ------------ | -------------------- | ----- | ----- | ----- | ---------------------------------------------------------------------------------------------------------- |
|                 | $S_m$ | $F^{\omega}_{\beta}$ | MAE   | mDice | mIoU  | $S_m$        | $F^{\omega}_{\beta}$ | MAE   | mDice | mIoU  |                                                                                                            |
| PVTv2-B5  (T=5) | 0.757 | 0.593                | 0.020 | 0.599 | 0.510 | 0.734        | 0.476                | 0.010 | 0.497 | 0.422 | [Weight](https://github.com/lartpang/ZoomNeXt/releases/download/weights-v0.2/pvtv2-b5-5frame-zoomnext.pth) |
