# ZoomNeXt

ZoomNeXt: A Unified Collaborative Pyramid Network for Camouflaged Object Detection (preprint)

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

## Results

| Datasets                            |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CAMO-TE, CHAMELEON, COD10K-TE, NC4K | [ResNet-50](https://github.com/lartpang/ZoomNeXt/releases/download/prediction-v0.1/zoomnext_res50.7z), [EfficientNet-B4](https://github.com/lartpang/ZoomNeXt/releases/download/prediction-v0.1/zoomnext_efficientb4.7z), [PVTv2-B2](https://github.com/lartpang/ZoomNeXt/releases/download/prediction-v0.1/zoomnext_pvtv2b2.7z), [PVTv2-B3](https://github.com/lartpang/ZoomNeXt/releases/download/prediction-v0.1/zoomnext_pvtv2b3.7z), [PVTv2-B4](https://github.com/lartpang/ZoomNeXt/releases/download/prediction-v0.1/zoomnext_pvtv2b4.7z), [PVTv2-B5](https://github.com/lartpang/ZoomNeXt/releases/download/prediction-v0.1/zoomnext_pvtv2b5.7z) |
| CAD, MoCA-Mask-TE                   | [PVTv2-B5](https://github.com/lartpang/ZoomNeXt/releases/download/prediction-v0.1/zoomnext_pvtv2b5_video.7z)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |

### CAMO-TE

| Backbone        | $S_m$ | $F^{\omega}_{\beta}$ | MAE   |
| --------------- | ----- | -------------------- | ----- |
| ResNet-50       | 0.833 | 0.774                | 0.065 |
| EfficientNet-B4 | 0.867 | 0.824                | 0.046 |
| PVTv2-B2        | 0.874 | 0.839                | 0.047 |
| PVTv2-B3        | 0.885 | 0.854                | 0.042 |
| PVTv2-B4        | 0.888 | 0.859                | 0.040 |
| PVTv2-B5        | 0.889 | 0.857                | 0.041 |

### CHAMELEON

| Backbone        | $S_m$ | $F^{\omega}_{\beta}$ | MAE   |
| --------------- | ----- | -------------------- | ----- |
| ResNet-50       | 0.908 | 0.858                | 0.021 |
| EfficientNet-B4 | 0.911 | 0.865                | 0.020 |
| PVTv2-B2        | 0.922 | 0.884                | 0.017 |
| PVTv2-B3        | 0.927 | 0.898                | 0.017 |
| PVTv2-B4        | 0.925 | 0.897                | 0.016 |
| PVTv2-B5        | 0.924 | 0.885                | 0.018 |

### COD10K-TE

| Backbone        | $S_m$ | $F^{\omega}_{\beta}$ | MAE   |
| --------------- | ----- | -------------------- | ----- |
| ResNet-50       | 0.861 | 0.768                | 0.026 |
| EfficientNet-B4 | 0.875 | 0.797                | 0.021 |
| PVTv2-B2        | 0.887 | 0.818                | 0.019 |
| PVTv2-B3        | 0.895 | 0.829                | 0.018 |
| PVTv2-B4        | 0.898 | 0.838                | 0.017 |
| PVTv2-B5        | 0.898 | 0.827                | 0.018 |

### NC4K

| Backbone        | $S_m$ | $F^{\omega}_{\beta}$ | MAE   |
| --------------- | ----- | -------------------- | ----- |
| ResNet-50       | 0.874 | 0.816                | 0.037 |
| EfficientNet-B4 | 0.884 | 0.837                | 0.032 |
| PVTv2-B2        | 0.892 | 0.852                | 0.030 |
| PVTv2-B3        | 0.900 | 0.861                | 0.028 |
| PVTv2-B4        | 0.900 | 0.865                | 0.028 |
| PVTv2-B5        | 0.903 | 0.863                | 0.028 |

### CAD

| Backbone | $S_m$ | $F^{\omega}_{\beta}$ | MAE   | mDice | mIoU  |
| -------- | ----- | -------------------- | ----- | ----- | ----- |
| PVTv2-B5 | 0.757 | 0.593                | 0.020 | 0.599 | 0.510 |

### MoCA-Mask-TE

| Backbone | $S_m$ | $F^{\omega}_{\beta}$ | MAE   | mDice | mIoU  |
| -------- | ----- | -------------------- | ----- | ----- | ----- |
| PVTv2-B5 | 0.734 | 0.476                | 0.010 | 0.497 | 0.422 |
