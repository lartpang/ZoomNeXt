import abc
import logging

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone.efficientnet import EfficientNet
from ..backbone.pvt_v2_eff import pvt_v2_eff_b2, pvt_v2_eff_b3, pvt_v2_eff_b4, pvt_v2_eff_b5
from .layers import MHSIU, RGPU, SimpleASPP
from .ops import ConvBNReLU, PixelNormalizer, resize_to

LOGGER = logging.getLogger("main")


class _ZoomNeXt_Base(nn.Module):
    @staticmethod
    def get_coef(iter_percentage=1, method="cos", milestones=(0, 1)):
        min_point, max_point = min(milestones), max(milestones)
        min_coef, max_coef = 0, 1

        ual_coef = 1.0
        if iter_percentage < min_point:
            ual_coef = min_coef
        elif iter_percentage > max_point:
            ual_coef = max_coef
        else:
            if method == "linear":
                ratio = (max_coef - min_coef) / (max_point - min_point)
                ual_coef = ratio * (iter_percentage - min_point)
            elif method == "cos":
                perc = (iter_percentage - min_point) / (max_point - min_point)
                normalized_coef = (1 - np.cos(perc * np.pi)) / 2
                ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
        return ual_coef

    @abc.abstractmethod
    def body(self):
        pass

    def forward(self, data, iter_percentage=1, **kwargs):
        logits = self.body(data=data)

        if self.training:
            mask = data["mask"]
            prob = logits.sigmoid()

            losses = []
            loss_str = []

            sod_loss = F.binary_cross_entropy_with_logits(input=logits, target=mask, reduction="mean")
            losses.append(sod_loss)
            loss_str.append(f"bce: {sod_loss.item():.5f}")

            ual_coef = self.get_coef(iter_percentage=iter_percentage, method="cos", milestones=(0, 1))
            ual_loss = ual_coef * (1 - (2 * prob - 1).abs().pow(2)).mean()
            losses.append(ual_loss)
            loss_str.append(f"powual_{ual_coef:.5f}: {ual_loss.item():.5f}")
            return dict(vis=dict(sal=prob), loss=sum(losses), loss_str=" ".join(loss_str))
        else:
            return logits

    def get_grouped_params(self):
        param_groups = {"pretrained": [], "fixed": [], "retrained": []}
        for name, param in self.named_parameters():
            if name.startswith("encoder.patch_embed1."):
                param.requires_grad = False
                param_groups["fixed"].append(param)
            elif name.startswith("encoder."):
                param_groups["pretrained"].append(param)
            else:
                if "clip." in name:
                    param.requires_grad = False
                    param_groups["fixed"].append(param)
                else:
                    param_groups["retrained"].append(param)
        LOGGER.info(
            f"Parameter Groups:{{"
            f"Pretrained: {len(param_groups['pretrained'])}, "
            f"Fixed: {len(param_groups['fixed'])}, "
            f"ReTrained: {len(param_groups['retrained'])}}}"
        )
        return param_groups


class RN50_ZoomNeXt(_ZoomNeXt_Base):
    def __init__(
        self, pretrained=True, num_frames=1, input_norm=True, mid_dim=64, siu_groups=4, hmu_groups=6, **kwargs
    ):
        super().__init__()
        self.encoder = timm.create_model(
            model_name="resnet50", features_only=True, out_indices=range(5), pretrained=False
        )
        if pretrained:
            params = torch.hub.load_state_dict_from_url(
                url="https://github.com/lartpang/ZoomNeXt/releases/download/weights-v0.1/resnet50-timm.pth",
                map_location="cpu",
            )
            self.encoder.load_state_dict(params, strict=False)

        self.tra_5 = SimpleASPP(in_dim=2048, out_dim=mid_dim)
        self.siu_5 = MHSIU(mid_dim, siu_groups)
        self.hmu_5 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_4 = ConvBNReLU(1024, mid_dim, 3, 1, 1)
        self.siu_4 = MHSIU(mid_dim, siu_groups)
        self.hmu_4 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_3 = ConvBNReLU(512, mid_dim, 3, 1, 1)
        self.siu_3 = MHSIU(mid_dim, siu_groups)
        self.hmu_3 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_2 = ConvBNReLU(256, mid_dim, 3, 1, 1)
        self.siu_2 = MHSIU(mid_dim, siu_groups)
        self.hmu_2 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_1 = ConvBNReLU(64, mid_dim, 3, 1, 1)
        self.siu_1 = MHSIU(mid_dim, siu_groups)
        self.hmu_1 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.normalizer = PixelNormalizer() if input_norm else nn.Identity()
        self.predictor = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(64, 32, 3, 1, 1),
            nn.Conv2d(32, 1, 1),
        )

    def normalize_encoder(self, x):
        x = self.normalizer(x)
        c1, c2, c3, c4, c5 = self.encoder(x)
        return c1, c2, c3, c4, c5

    def body(self, data):
        l_trans_feats = self.normalize_encoder(data["image_l"])
        m_trans_feats = self.normalize_encoder(data["image_m"])
        s_trans_feats = self.normalize_encoder(data["image_s"])

        l, m, s = (
            self.tra_5(l_trans_feats[4]),
            self.tra_5(m_trans_feats[4]),
            self.tra_5(s_trans_feats[4]),
        )
        lms = self.siu_5(l=l, m=m, s=s)
        x = self.hmu_5(lms)

        l, m, s = (
            self.tra_4(l_trans_feats[3]),
            self.tra_4(m_trans_feats[3]),
            self.tra_4(s_trans_feats[3]),
        )
        lms = self.siu_4(l=l, m=m, s=s)
        x = self.hmu_4(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        l, m, s = (
            self.tra_3(l_trans_feats[2]),
            self.tra_3(m_trans_feats[2]),
            self.tra_3(s_trans_feats[2]),
        )
        lms = self.siu_3(l=l, m=m, s=s)
        x = self.hmu_3(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        l, m, s = (
            self.tra_2(l_trans_feats[1]),
            self.tra_2(m_trans_feats[1]),
            self.tra_2(s_trans_feats[1]),
        )
        lms = self.siu_2(l=l, m=m, s=s)
        x = self.hmu_2(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        l, m, s = (
            self.tra_1(l_trans_feats[0]),
            self.tra_1(m_trans_feats[0]),
            self.tra_1(s_trans_feats[0]),
        )
        lms = self.siu_1(l=l, m=m, s=s)
        x = self.hmu_1(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        return self.predictor(x)


class PvtV2B2_ZoomNeXt(_ZoomNeXt_Base):
    def __init__(
        self,
        pretrained=True,
        num_frames=1,
        input_norm=True,
        mid_dim=64,
        siu_groups=4,
        hmu_groups=6,
        use_checkpoint=False,
    ):
        super().__init__()
        self.set_backbone(pretrained=pretrained, use_checkpoint=use_checkpoint)

        self.embed_dims = self.encoder.embed_dims
        self.tra_5 = SimpleASPP(self.embed_dims[3], out_dim=mid_dim)
        self.siu_5 = MHSIU(mid_dim, siu_groups)
        self.hmu_5 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_4 = ConvBNReLU(self.embed_dims[2], mid_dim, 3, 1, 1)
        self.siu_4 = MHSIU(mid_dim, siu_groups)
        self.hmu_4 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_3 = ConvBNReLU(self.embed_dims[1], mid_dim, 3, 1, 1)
        self.siu_3 = MHSIU(mid_dim, siu_groups)
        self.hmu_3 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_2 = ConvBNReLU(self.embed_dims[0], mid_dim, 3, 1, 1)
        self.siu_2 = MHSIU(mid_dim, siu_groups)
        self.hmu_2 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), ConvBNReLU(64, mid_dim, 3, 1, 1)
        )

        self.normalizer = PixelNormalizer() if input_norm else nn.Identity()
        self.predictor = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(64, 32, 3, 1, 1),
            nn.Conv2d(32, 1, 1),
        )

    def set_backbone(self, pretrained: bool, use_checkpoint: bool):
        self.encoder = pvt_v2_eff_b2(pretrained=pretrained, use_checkpoint=use_checkpoint)

    def normalize_encoder(self, x):
        x = self.normalizer(x)
        features = self.encoder(x)
        c2 = features["reduction_2"]
        c3 = features["reduction_3"]
        c4 = features["reduction_4"]
        c5 = features["reduction_5"]
        return c2, c3, c4, c5

    def body(self, data):
        l_trans_feats = self.normalize_encoder(data["image_l"])
        m_trans_feats = self.normalize_encoder(data["image_m"])
        s_trans_feats = self.normalize_encoder(data["image_s"])

        l, m, s = self.tra_5(l_trans_feats[3]), self.tra_5(m_trans_feats[3]), self.tra_5(s_trans_feats[3])
        lms = self.siu_5(l=l, m=m, s=s)
        x = self.hmu_5(lms)

        l, m, s = self.tra_4(l_trans_feats[2]), self.tra_4(m_trans_feats[2]), self.tra_4(s_trans_feats[2])
        lms = self.siu_4(l=l, m=m, s=s)
        x = self.hmu_4(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        l, m, s = self.tra_3(l_trans_feats[1]), self.tra_3(m_trans_feats[1]), self.tra_3(s_trans_feats[1])
        lms = self.siu_3(l=l, m=m, s=s)
        x = self.hmu_3(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        l, m, s = self.tra_2(l_trans_feats[0]), self.tra_2(m_trans_feats[0]), self.tra_2(s_trans_feats[0])
        lms = self.siu_2(l=l, m=m, s=s)
        x = self.hmu_2(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        x = self.tra_1(x)
        return self.predictor(x)


class PvtV2B3_ZoomNeXt(PvtV2B2_ZoomNeXt):
    def set_backbone(self, pretrained: bool, use_checkpoint: bool):
        self.encoder = pvt_v2_eff_b3(pretrained=pretrained, use_checkpoint=use_checkpoint)


class PvtV2B4_ZoomNeXt(PvtV2B2_ZoomNeXt):
    def set_backbone(self, pretrained: bool, use_checkpoint: bool):
        self.encoder = pvt_v2_eff_b4(pretrained=pretrained, use_checkpoint=use_checkpoint)


class PvtV2B5_ZoomNeXt(PvtV2B2_ZoomNeXt):
    def set_backbone(self, pretrained: bool, use_checkpoint: bool):
        self.encoder = pvt_v2_eff_b5(pretrained=pretrained, use_checkpoint=use_checkpoint)


class videoPvtV2B5_ZoomNeXt(PvtV2B5_ZoomNeXt):
    def get_grouped_params(self):
        param_groups = {"pretrained": [], "fixed": [], "retrained": []}
        for name, param in self.named_parameters():
            if name.startswith("encoder.patch_embed1."):
                param.requires_grad = False
                param_groups["fixed"].append(param)
            elif name.startswith("encoder."):
                param_groups["pretrained"].append(param)
            else:
                if "temperal_proj" in name:
                    param_groups["retrained"].append(param)
                else:
                    param_groups["pretrained"].append(param)

        LOGGER.info(
            f"Parameter Groups:{{"
            f"Pretrained: {len(param_groups['pretrained'])}, "
            f"Fixed: {len(param_groups['fixed'])}, "
            f"ReTrained: {len(param_groups['retrained'])}}}"
        )
        return param_groups


class EffB1_ZoomNeXt(_ZoomNeXt_Base):
    def __init__(self, pretrained, num_frames=1, input_norm=True, mid_dim=64, siu_groups=4, hmu_groups=6, **kwargs):
        super().__init__()
        self.set_backbone(pretrained)

        self.tra_5 = SimpleASPP(self.embed_dims[4], out_dim=mid_dim)
        self.siu_5 = MHSIU(mid_dim, siu_groups)
        self.hmu_5 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_4 = ConvBNReLU(self.embed_dims[3], mid_dim, 3, 1, 1)
        self.siu_4 = MHSIU(mid_dim, siu_groups)
        self.hmu_4 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_3 = ConvBNReLU(self.embed_dims[2], mid_dim, 3, 1, 1)
        self.siu_3 = MHSIU(mid_dim, siu_groups)
        self.hmu_3 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_2 = ConvBNReLU(self.embed_dims[1], mid_dim, 3, 1, 1)
        self.siu_2 = MHSIU(mid_dim, siu_groups)
        self.hmu_2 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_1 = ConvBNReLU(self.embed_dims[0], mid_dim, 3, 1, 1)
        self.siu_1 = MHSIU(mid_dim, siu_groups)
        self.hmu_1 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.normalizer = PixelNormalizer() if input_norm else nn.Identity()
        self.predictor = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(64, 32, 3, 1, 1),
            nn.Conv2d(32, 1, 1),
        )

    def set_backbone(self, pretrained):
        self.encoder = EfficientNet.from_pretrained("efficientnet-b1", pretrained=pretrained)
        self.embed_dims = [16, 24, 40, 112, 320]

    def normalize_encoder(self, x):
        x = self.normalizer(x)
        features = self.encoder.extract_endpoints(x)
        c1 = features["reduction_1"]
        c2 = features["reduction_2"]
        c3 = features["reduction_3"]
        c4 = features["reduction_4"]
        c5 = features["reduction_5"]
        return c1, c2, c3, c4, c5

    def body(self, data):
        l_trans_feats = self.normalize_encoder(data["image_l"])
        m_trans_feats = self.normalize_encoder(data["image_m"])
        s_trans_feats = self.normalize_encoder(data["image_s"])

        l, m, s = self.tra_5(l_trans_feats[4]), self.tra_5(m_trans_feats[4]), self.tra_5(s_trans_feats[4])
        lms = self.siu_5(l=l, m=m, s=s)
        x = self.hmu_5(lms)

        l, m, s = self.tra_4(l_trans_feats[3]), self.tra_4(m_trans_feats[3]), self.tra_4(s_trans_feats[3])
        lms = self.siu_4(l=l, m=m, s=s)
        x = self.hmu_4(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        l, m, s = self.tra_3(l_trans_feats[2]), self.tra_3(m_trans_feats[2]), self.tra_3(s_trans_feats[2])
        lms = self.siu_3(l=l, m=m, s=s)
        x = self.hmu_3(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        l, m, s = self.tra_2(l_trans_feats[1]), self.tra_2(m_trans_feats[1]), self.tra_2(s_trans_feats[1])
        lms = self.siu_2(l=l, m=m, s=s)
        x = self.hmu_2(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        l, m, s = self.tra_1(l_trans_feats[0]), self.tra_1(m_trans_feats[0]), self.tra_1(s_trans_feats[0])
        lms = self.siu_1(l=l, m=m, s=s)
        x = self.hmu_1(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        return self.predictor(x)


class EffB4_ZoomNeXt(EffB1_ZoomNeXt):
    def set_backbone(self, pretrained):
        self.encoder = EfficientNet.from_pretrained("efficientnet-b4", pretrained=pretrained)
        self.embed_dims = [24, 32, 56, 160, 448]
