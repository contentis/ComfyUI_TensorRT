import os
import re
import torch

from .attention_plugin import attn_cls


def filter_func(name):
    pattern = re.compile(
        r".*(emb_layers|time_embed|input_blocks.0.0|out.2|skip_connection|label_emb.0|x_embedder|pos_embed|t_embedder|y_embedder|context_embedder|final_layer.adaLN_modulation|final_layer.linear).*"
    )
    return pattern.match(name) is not None


def quantize_lvl(unet, quant_level=2.5, linear_only=False):
    """
    We should disable the unwanted quantizer when exporting the onnx
    Because in the current modelopt setting, it will load the quantizer amax for all the layers even
    if we didn't add that unwanted layer into the config during the calibration
    """
    for name, module in unet.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            if linear_only:
                module.input_quantizer.disable()
                module.weight_quantizer.disable()
            else:
                module.input_quantizer.enable()
                module.weight_quantizer.enable()
        elif isinstance(module, torch.nn.Linear):
            if (
                (quant_level >= 2 and "ff.net" in name)
                or (
                    quant_level >= 2.5
                    and ("to_q" in name or "to_k" in name or "to_v" in name)
                )
                or quant_level >= 3
            ):
                module.input_quantizer.enable()
                module.weight_quantizer.enable()
            else:
                module.input_quantizer.disable()
                module.weight_quantizer.disable()
        elif isinstance(module, attn_cls):
            if quant_level >= 4:
                module.q_bmm_quantizer.enable()
                module.k_bmm_quantizer.enable()
                module.v_bmm_quantizer.enable()
                module.softmax_quantizer.enable()
            else:
                module.q_bmm_quantizer.disable()
                module.k_bmm_quantizer.disable()
                module.v_bmm_quantizer.disable()
                module.softmax_quantizer.disable()
        elif "Attention" in module.__class__.__name__:
            print("DEBUG")


def load_calib_prompts(batch_size, calib_data_path="./calib_prompts.txt"):
    if not os.path.exists(calib_data_path):
        raise FileNotFoundError
    with open(calib_data_path, "r", encoding="utf8") as file:
        lst = [line.rstrip("\n") for line in file]
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]
