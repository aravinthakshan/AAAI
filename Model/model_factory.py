BACKBONE_CHOICES = [
    "efficientb0",
    "tinynet-a",
    "starnet_s050",
    "starnet_s100",
    "starnet_s150",
    "starnet_s1",
    "starnet_s2",
    "starnet_s3",
    "starnet_s4",
    "demonet_d12_w32_sum",
    "demonet_d12_w32_mul",
    "demonet_d12_w64_sum",
    "demonet_d12_w64_mul",
    "demonet_d12_w96_sum",
    "demonet_d12_w96_mul",
    "demonet_d12_w128_sum",
    "demonet_d12_w128_mul",
    "demonet_d12_w160_sum",
    "demonet_d12_w160_mul",
    "demonet_d12_w192_sum",
    "demonet_d12_w192_mul",
    "demonet_d12_w224_sum",
    "demonet_d12_w224_mul",
    "demonet_d12_w256_sum",
    "demonet_d12_w256_mul",
    "demonet_d12_w288_sum",
    "demonet_d12_w288_mul",
    "demonet_d12_w320_sum",
    "demonet_d12_w320_mul",
    "demonet_d12_w352_sum",
    "demonet_d12_w352_mul",
    "demonet_d12_w384_sum",
    "demonet_d12_w384_mul",
    "demonet_d12_w416_sum",
    "demonet_d12_w416_mul",
    "demonet_d12_w448_sum",
    "demonet_d12_w448_mul",
    "demonet_d6_w192_sum",
    "demonet_d6_w192_mul",
    "demonet_d8_w192_sum",
    "demonet_d8_w192_mul",
    "demonet_d10_w192_sum",
    "demonet_d10_w192_mul",
    "demonet_d14_w192_sum",
    "demonet_d14_w192_mul",
    "demonet_d16_w192_sum",
    "demonet_d16_w192_mul",
    "demonet_d18_w192_sum",
    "demonet_d18_w192_mul",
    "demonet_d20_w192_sum",
    "demonet_d20_w192_mul",
    "demonet_d22_w192_sum",
    "demonet_d22_w192_mul",
    "demonet_d24_w192_sum",
    "demonet_d24_w192_mul",
]

MODEL_CHOICES = ["FINet", "LAFinet", "LaFINet", "LaplacianFINet"]


def normalize_model_name(model_name):
    if model_name in ["LAFinet", "LaFINet", "LaplacianFINet"]:
        return "LAFinet"
    if model_name == "FINet":
        return "FINet"
    raise ValueError(f"Unsupported model: {model_name}")


def build_model(model_name, backbone="efficientb0", channels=(8, 24, 32, 64)):
    model_name = normalize_model_name(model_name)
    if model_name == "FINet":
        if backbone not in ["efficientb0", "tinynet-a"]:
            raise ValueError("FINet currently supports only efficientb0 and tinynet-a backbones.")
        from Model.FINet import FINet
        return FINet(backbone=backbone, channels=channels)
    from Model.LAFinet import LaplacianFINet
    return LaplacianFINet(backbone=backbone, channels=channels)
