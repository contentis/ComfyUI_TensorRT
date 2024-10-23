import torch
from .plugin_calib import PercentileCalibrator
from .utils import filter_func

FP8_BF16_DEFAULT_CONFIG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "trt_high_precision_dtype": "BFloat16",
        },
        "*input_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "trt_high_precision_dtype": "BFloat16",
        },
        "*output_quantizer": {"enable": False},
        "*q_bmm_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "trt_high_precision_dtype": "BFloat16",
        },
        "*k_bmm_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "trt_high_precision_dtype": "BFloat16",
        },
        "*v_bmm_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "trt_high_precision_dtype": "BFloat16",
        },
        "*softmax_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "trt_high_precision_dtype": "BFloat16",
        },
        "default": {"enable": False},
    },
    "algorithm": "max",
}

FP8_FP16_DEFAULT_CONFIG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Half"},
        "*input_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Half"},
        "*output_quantizer": {"enable": False},
        "*q_bmm_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Half"},
        "*k_bmm_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Half"},
        "*v_bmm_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Half"},
        "*softmax_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "trt_high_precision_dtype": "Half",
        },
        "default": {"enable": False},
    },
    "algorithm": "max",
}

FP8_FP32_DEFAULT_CONFIG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "trt_high_precision_dtype": "Float",
        },
        "*input_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Float"},
        "*output_quantizer": {"enable": False},
        "*q_bmm_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Float"},
        "*k_bmm_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Float"},
        "*v_bmm_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Float"},
        "*softmax_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "trt_high_precision_dtype": "Float",
        },
        "default": {"enable": False},
    },
    "algorithm": "max",
}


def get_int8_config(
    model,
    quant_level=3,
    alpha=0.8,
    percentile=1.0,
    num_inference_steps=20,
    collect_method="global_min",
):
    quant_config = {
        "quant_cfg": {
            "*output_quantizer": {"enable": False},
            "default": {"enable": False},
        },
        "algorithm": {"method": "smoothquant", "alpha": alpha},
    }
    for name, module in model.named_modules():
        w_name = f"{name}*weight_quantizer"
        i_name = f"{name}*input_quantizer"

        if w_name in quant_config["quant_cfg"].keys() or i_name in quant_config["quant_cfg"].keys():
            continue
        if filter_func(name):
            continue
        if isinstance(module, torch.nn.Linear):
            if (
                (quant_level >= 2 and "ff.net" in name)
                or (quant_level >= 2.5 and ("to_q" in name or "to_k" in name or "to_v" in name))
                or quant_level == 3
            ):
                quant_config["quant_cfg"][w_name] = {
                    "num_bits": 8,
                    "axis": 0,
                    "trt_high_precision_dtype": "Half",
                }
                quant_config["quant_cfg"][i_name] = {
                    "num_bits": 8,
                    "axis": -1,
                    "trt_high_precision_dtype": "Half",
                }
        elif isinstance(module, torch.nn.Conv2d):
            quant_config["quant_cfg"][w_name] = {
                "num_bits": 8,
                "axis": 0,
                "trt_high_precision_dtype": "Half",
            }
            quant_config["quant_cfg"][i_name] = {
                "num_bits": 8,
                "axis": None,
                "trt_high_precision_dtype": "Half",
                "calibrator": (
                    PercentileCalibrator,
                    (),
                    {
                        "num_bits": 8,
                        "axis": None,
                        "percentile": percentile,
                        "total_step": num_inference_steps,
                        "collect_method": collect_method,
                    },
                ),
            }
    return quant_config


def get_fp4_config(model, fp4_linear_only=False):
    """fp4 for linear, optionally fp8 for conv"""

    quant_config = {
        "quant_cfg": {},
        "algorithm": "max",
    }
    for name, module in model.named_modules():
        w_name = f"{name}*weight_quantizer"
        i_name = f"{name}*input_quantizer"

        if (
            w_name in quant_config["quant_cfg"].keys()  # type: ignore
            or i_name in quant_config["quant_cfg"].keys()  # type: ignore
        ):
            continue
        if isinstance(module, torch.nn.Linear):
            quant_config["quant_cfg"][w_name] = {  # type: ignore
                "num_bits": (2, 1),
                "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
                "axis": None,
            }
            quant_config["quant_cfg"][i_name] = {  # type: ignore
                "num_bits": (2, 1),
                "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
                "axis": None,
            }
        elif isinstance(module, torch.nn.Conv2d):
            if fp4_linear_only:
                quant_config["quant_cfg"][w_name] = {"enable": False}  # type: ignore
                quant_config["quant_cfg"][i_name] = {"enable": False}  # type: ignore
            else:
                # fp8 for conv
                quant_config["quant_cfg"][w_name] = {"num_bits": (4, 3), "axis": None}  # type: ignore
                quant_config["quant_cfg"][i_name] = {"num_bits": (4, 3), "axis": None}  # type: ignore
    return quant_config


def set_stronglytyped_precision(quant_config, precision: str = "Half"):
    for key in quant_config["quant_cfg"].keys():
        if "trt_high_precision_dtype" in quant_config["quant_cfg"][key].keys():
            quant_config["quant_cfg"][key]["trt_high_precision_dtype"] = precision
