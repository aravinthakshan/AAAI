"""
Implementation of Prof-of-Concept Network: StarNet.

We make StarNet as simple as possible [to show the key contribution of element-wise multiplication]:
    - like NO layer-scale in network design,
    - and NO EMA during training,
    - which would improve the performance further.

Created by: Xu Ma (Email: ma.xu1@northeastern.edu)
Modified Date: Mar/29/2024

------------------------------------------------------------------
Local-feature refinement (new)
------------------------------------------------------------------
`StarNetEncoder` now optionally inserts a `LocalDetailRefine` module
(LDSConv-based — see Model/lap_utils.py) *after* selected stage outputs,
rather than modifying anything inside `Block`. This is deliberate:

  • `load_starnet_pretrained` (Model/star_utils.py) copies ImageNet
    weights into `StarNetEncoder` by exact key name + shape match.
    Swapping the depthwise conv inside `Block` for LDSConv would change
    those layers' parameter shapes, silently dropping pretrained init
    for every block touched — undermining Fix 4, which was already
    identified as the single biggest source of metric improvement.
    A post-stage hook leaves every `Block` (and its pretrained weights)
    completely untouched.

  • `LocalDetailRefine` is gated by a learnable scalar passed through
    `tanh`, initialized at 0. At the start of fine-tuning the encoder
    is mathematically identical to the pretrained one — the refinement
    only phases in as training pulls the gate open. No information is
    lost relative to the pretrained checkpoint at init, and there's no
    risk of compounding with other zero-initialized branches downstream
    (StarDeBlock.cross_g, LDSConv.offset_conv) the way an always-on
    in-block replacement would.

  • Default `local_enhance_stages=(2,)`: stage 2 output only (the
    higher-resolution of the two mid stages). Texture/edge detail
    relevant to camouflage discrimination is still rich there, but the
    spatial map is small enough that LDSConv's gather-based sampling
    isn't prohibitively expensive. Stage 1 is supported but off by
    default (largest spatial map, costliest); stages 3/4 are already
    semantically abstracted, where extra local adaptivity buys little
    for camouflage detection specifically.
"""
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

from Model.lap_utils import LDSConv

model_urls = {
    "starnet_s1": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s1.pth.tar",
    "starnet_s2": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s2.pth.tar",
    "starnet_s3": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s3.pth.tar",
    "starnet_s4": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s4.pth.tar",
}


class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


class StarNet(nn.Module):
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1000, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = 32
        # stem layer
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6())
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # stochastic depth
        # build stages
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
        # head
        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.in_channel, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = torch.flatten(self.avgpool(self.norm(x)), 1)
        return self.head(x)


class LocalDetailRefine(nn.Module):
    """
    Post-stage local-feature refinement.

    Wraps LDSConv (deformable sampling + LKP/SKA spatially-adaptive
    weighting) around a stage output. Gated by a learnable scalar passed
    through tanh and initialized at 0, so this module is the identity
    function at init — the pretrained backbone's behaviour is preserved
    exactly until fine-tuning pulls the gate open.
    """
    def __init__(self, channels: int, num_param: int = 9, groups: int = 8, lks: int = 7):
        super().__init__()
        self.refine = LDSConv(channels, num_param=num_param, groups=groups, lks=lks)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LDSConv.forward already returns bn(out) + x, so subtracting x
        # isolates the refinement delta that gets gated.
        delta = self.refine(x) - x
        return x + torch.tanh(self.gate) * delta


class StarNetEncoder(nn.Module):
    def __init__(self, variant="starnet_s1", local_enhance_stages=(2,),
                 local_num_param: int = 9, local_groups: int = 8, local_lks: int = 7,
                 **kwargs):
        super().__init__()
        configs = {
            "starnet_s050": (16, [1, 1, 3, 1], 3),
            "starnet_s100": (20, [1, 2, 4, 1], 4),
            "starnet_s150": (24, [1, 2, 4, 2], 3),
            "starnet_s1": (24, [2, 2, 8, 3], 4),
            "starnet_s2": (32, [1, 2, 6, 2], 4),
            "starnet_s3": (32, [2, 2, 8, 4], 4),
            "starnet_s4": (32, [3, 3, 12, 5], 4),
        }
        if variant not in configs:
            raise ValueError(f"Unsupported StarNet encoder variant: {variant}")

        base_dim, depths, mlp_ratio = configs[variant]
        model = StarNet(base_dim=base_dim, depths=depths, mlp_ratio=mlp_ratio, **kwargs)
        self.base_dim = base_dim
        self.stem = model.stem
        self.stages = model.stages

        # stage_channels[i] for i in 1..4 is the output width of self.stages[i-1]
        stage_channels = [32, base_dim, base_dim * 2, base_dim * 4, base_dim * 8]

        # Post-stage local refinement (does NOT touch Block / pretrained weights).
        # Keys live under `local_refine.<stage_idx>` so load_starnet_pretrained's
        # key-matching against a classification checkpoint simply reports them
        # as "missing" (new capacity, randomly/gate-initialized) rather than
        # clobbering anything.
        self.local_enhance_stages = set(local_enhance_stages)
        self.local_refine = nn.ModuleDict({
            str(i): LocalDetailRefine(stage_channels[i], num_param=local_num_param,
                                       groups=local_groups, lks=local_lks)
            for i in self.local_enhance_stages
        })

    def forward(self, x):
        out0 = self.stem(x)
        features = [out0]
        for i, stage in enumerate(self.stages, start=1):
            feat = stage(features[-1])
            if i in self.local_enhance_stages:
                feat = self.local_refine[str(i)](feat)
            features.append(feat)
        return tuple(features)

    def get_stage_channels(self):
        return [32, self.base_dim, self.base_dim * 2, self.base_dim * 4, self.base_dim * 8]


@register_model
def starnet_s1(pretrained=False, **kwargs):
    model = StarNet(24, [2, 2, 8, 3], **kwargs)
    if pretrained:
        url = model_urls['starnet_s1']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model


@register_model
def starnet_s2(pretrained=False, **kwargs):
    model = StarNet(32, [1, 2, 6, 2], **kwargs)
    if pretrained:
        url = model_urls['starnet_s2']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model


@register_model
def starnet_s3(pretrained=False, **kwargs):
    model = StarNet(32, [2, 2, 8, 4], **kwargs)
    if pretrained:
        url = model_urls['starnet_s3']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model


@register_model
def starnet_s4(pretrained=False, **kwargs):
    model = StarNet(32, [3, 3, 12, 5], **kwargs)
    if pretrained:
        url = model_urls['starnet_s4']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model


# very small networks #
@register_model
def starnet_s050(pretrained=False, **kwargs):
    return StarNet(16, [1, 1, 3, 1], 3, **kwargs)


@register_model
def starnet_s100(pretrained=False, **kwargs):
    return StarNet(20, [1, 2, 4, 1], 4, **kwargs)


@register_model
def starnet_s150(pretrained=False, **kwargs):
    return StarNet(24, [1, 2, 4, 2], 3, **kwargs)