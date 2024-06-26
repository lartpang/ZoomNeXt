import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .ops import ConvBNReLU, resize_to


class SimpleASPP(nn.Module):
    def __init__(self, in_dim, out_dim, dilation=3):
        """A simple ASPP variant.

        Args:
            in_dim (int): Input channels.
            out_dim (int): Output channels.
            dilation (int, optional): Dilation of the convolution operation. Defaults to 3.
        """
        super().__init__()
        self.conv1x1_1 = ConvBNReLU(in_dim, 2 * out_dim, 1)
        self.conv1x1_2 = ConvBNReLU(out_dim, out_dim, 1)
        self.conv3x3_1 = ConvBNReLU(out_dim, out_dim, 3, dilation=dilation, padding=dilation)
        self.conv3x3_2 = ConvBNReLU(out_dim, out_dim, 3, dilation=dilation, padding=dilation)
        self.conv3x3_3 = ConvBNReLU(out_dim, out_dim, 3, dilation=dilation, padding=dilation)
        self.fuse = nn.Sequential(ConvBNReLU(5 * out_dim, out_dim, 1), ConvBNReLU(out_dim, out_dim, 3, 1, 1))

    def forward(self, x):
        y = self.conv1x1_1(x)
        y1, y5 = y.chunk(2, dim=1)

        # dilation branch
        y2 = self.conv3x3_1(y1)
        y3 = self.conv3x3_2(y2)
        y4 = self.conv3x3_3(y3)

        # global branch
        y0 = torch.mean(y5, dim=(2, 3), keepdim=True)
        y0 = self.conv1x1_2(y0)
        y0 = resize_to(y0, tgt_hw=x.shape[-2:])
        return self.fuse(torch.cat([y0, y1, y2, y3, y4], dim=1))


class DifferenceAwareOps(nn.Module):
    def __init__(self, num_frames):
        super().__init__()
        self.num_frames = num_frames

        self.temperal_proj_norm = nn.LayerNorm(num_frames, elementwise_affine=False)
        self.temperal_proj_kv = nn.Linear(num_frames, 2 * num_frames, bias=False)
        self.temperal_proj = nn.Sequential(
            nn.Conv2d(num_frames, num_frames, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(num_frames, num_frames, 3, 1, 1, bias=False),
        )
        for t in self.parameters():
            nn.init.zeros_(t)

    def forward(self, x):
        if self.num_frames == 1:
            return x

        unshifted_x_tmp = rearrange(x, "(b t) c h w -> b c h w t", t=self.num_frames)
        B, C, H, W, T = unshifted_x_tmp.shape
        shifted_x_tmp = torch.roll(unshifted_x_tmp, shifts=1, dims=-1)
        diff_q = shifted_x_tmp - unshifted_x_tmp  # B,C,H,W,T
        diff_q = self.temperal_proj_norm(diff_q)  # normalization along the time

        # merge all channels
        diff_k, diff_v = self.temperal_proj_kv(diff_q).chunk(2, dim=-1)
        diff_qk = torch.einsum("bxhwt, byhwt -> bxyt", diff_q, diff_k) * (H * W) ** -0.5
        temperal_diff = torch.einsum("bxyt, byhwt -> bxhwt", diff_qk.softmax(dim=2), diff_v)

        temperal_diff = rearrange(temperal_diff, "b c h w t -> (b c) t h w")
        shifted_x_tmp = self.temperal_proj(temperal_diff)  # combine different time step
        shifted_x_tmp = rearrange(shifted_x_tmp, "(b c) t h w -> (b t) c h w", c=x.shape[1])
        return x + shifted_x_tmp


class RGPU(nn.Module):
    def __init__(self, in_c, num_groups=6, hidden_dim=None, num_frames=1):
        super().__init__()
        self.num_groups = num_groups

        hidden_dim = hidden_dim or in_c // 2
        expand_dim = hidden_dim * num_groups
        self.expand_conv = ConvBNReLU(in_c, expand_dim, 1)
        self.gate_genator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(num_groups * hidden_dim, hidden_dim, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, num_groups * hidden_dim, 1),
            nn.Softmax(dim=1),
        )

        self.interact = nn.ModuleDict()
        self.interact["0"] = ConvBNReLU(hidden_dim, 3 * hidden_dim, 3, 1, 1)
        for group_id in range(1, num_groups - 1):
            self.interact[str(group_id)] = ConvBNReLU(2 * hidden_dim, 3 * hidden_dim, 3, 1, 1)
        self.interact[str(num_groups - 1)] = ConvBNReLU(2 * hidden_dim, 2 * hidden_dim, 3, 1, 1)

        self.fuse = nn.Sequential(
            DifferenceAwareOps(num_frames=num_frames),
            ConvBNReLU(num_groups * hidden_dim, in_c, 3, 1, 1, act_name=None),
        )
        self.final_relu = nn.ReLU(True)

    def forward(self, x):
        xs = self.expand_conv(x).chunk(self.num_groups, dim=1)

        outs = []
        gates = []

        group_id = 0
        curr_x = xs[group_id]
        branch_out = self.interact[str(group_id)](curr_x)
        curr_out, curr_fork, curr_gate = branch_out.chunk(3, dim=1)
        outs.append(curr_out)
        gates.append(curr_gate)

        for group_id in range(1, self.num_groups - 1):
            curr_x = torch.cat([xs[group_id], curr_fork], dim=1)
            branch_out = self.interact[str(group_id)](curr_x)
            curr_out, curr_fork, curr_gate = branch_out.chunk(3, dim=1)
            outs.append(curr_out)
            gates.append(curr_gate)

        group_id = self.num_groups - 1
        curr_x = torch.cat([xs[group_id], curr_fork], dim=1)
        branch_out = self.interact[str(group_id)](curr_x)
        curr_out, curr_gate = branch_out.chunk(2, dim=1)
        outs.append(curr_out)
        gates.append(curr_gate)

        out = torch.cat(outs, dim=1)
        gate = self.gate_genator(torch.cat(gates, dim=1))
        out = self.fuse(out * gate)
        return self.final_relu(out + x)


class MHSIU(nn.Module):
    def __init__(self, in_dim, num_groups=4):
        super().__init__()
        self.conv_l_pre = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s_pre = ConvBNReLU(in_dim, in_dim, 3, 1, 1)

        self.conv_l = ConvBNReLU(in_dim, in_dim, 3, 1, 1)  # intra-branch
        self.conv_m = ConvBNReLU(in_dim, in_dim, 3, 1, 1)  # intra-branch
        self.conv_s = ConvBNReLU(in_dim, in_dim, 3, 1, 1)  # intra-branch

        self.conv_lms = ConvBNReLU(3 * in_dim, 3 * in_dim, 1)  # inter-branch
        self.initial_merge = ConvBNReLU(3 * in_dim, 3 * in_dim, 1)  # inter-branch

        self.num_groups = num_groups
        self.trans = nn.Sequential(
            ConvBNReLU(3 * in_dim // num_groups, in_dim // num_groups, 1),
            ConvBNReLU(in_dim // num_groups, in_dim // num_groups, 3, 1, 1),
            nn.Conv2d(in_dim // num_groups, 3, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, l, m, s):
        tgt_size = m.shape[2:]

        l = self.conv_l_pre(l)
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        s = self.conv_s_pre(s)
        s = resize_to(s, tgt_hw=m.shape[2:])

        l = self.conv_l(l)
        m = self.conv_m(m)
        s = self.conv_s(s)
        lms = torch.cat([l, m, s], dim=1)  # BT,3C,H,W

        attn = self.conv_lms(lms)  # BT,3C,H,W
        attn = rearrange(attn, "bt (nb ng d) h w -> (bt ng) (nb d) h w", nb=3, ng=self.num_groups)
        attn = self.trans(attn)  # BTG,3,H,W
        attn = attn.unsqueeze(dim=2)  # BTG,3,1,H,W

        x = self.initial_merge(lms)
        x = rearrange(x, "bt (nb ng d) h w -> (bt ng) nb d h w", nb=3, ng=self.num_groups)
        x = (attn * x).sum(dim=1)
        x = rearrange(x, "(bt ng) d h w -> bt (ng d) h w", ng=self.num_groups)
        return x
