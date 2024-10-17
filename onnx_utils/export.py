import torch
from enum import Enum
from typing import List
import os

import onnx
from onnx.external_data_helper import _get_all_tensors, ExternalDataInfo
from onnxmltools.utils.float16_converter import convert_float_to_float16
import onnx_graphsurgeon as gs

import comfy

from ..mo_utils.attention_plugin import attn_cls
from .fp8_onnx_graphsurgeon import (
    cast_fp8_mha_io,
    cast_resize_io,
    convert_fp16_io,
    convert_zp_fp8,
)
import folder_paths
import time


def _get_onnx_external_data_tensors(model: onnx.ModelProto) -> List[str]:
    """
    Gets the paths of the external data tensors in the model.
    Note: make sure you load the model with load_external_data=False.
    """
    model_tensors = _get_all_tensors(model)
    model_tensors_ext = [
        ExternalDataInfo(tensor).location
        for tensor in model_tensors
        if tensor.HasField("data_location")
        and tensor.data_location == onnx.TensorProto.EXTERNAL
    ]
    return model_tensors_ext


class ModelType(Enum):
    SD1x = "SD1.x"
    SD2x768v = "SD2.x-768v"
    SDXL_BASE = "SDXL-Base"
    SDXL_REFINER = "SDXL-Refiner"
    SVD = "SVD"
    SD3 = "SD3"
    AuraFlow = "AuraFlow"
    FLUX_DEV = "FLUX-Dev"
    FLUX_SCHNELL = "FLUX-Schnell"
    UNKNOWN = "Unknown"

    def __eq__(self, value: object) -> bool:
        return self.value == value

    @classmethod
    def detect_version(cls, model):
        if isinstance(model.model, comfy.model_base.SD3):
            return cls.SD3
        elif isinstance(model.model, comfy.model_base.AuraFlow):
            return cls.AuraFlow
        elif isinstance(model.model, comfy.model_base.Flux):
            if model.model.model_config.unet_config["guidance_embed"]:
                return cls.FLUX_DEV
            else:
                return cls.FLUX_SCHNELL

        if model.model.model_config.unet_config.get("use_temporal_resblock", False):
            return cls.SVD

        context_dim = model.model.model_config.unet_config.get("context_dim", None)
        y_dim = model.model.adm_channels

        if context_dim == 768:
            return cls.SD1x
        elif context_dim == 1024:
            return cls.SD2x768v
        elif context_dim == 2048:
            if y_dim == 2560:
                return cls.SDXL_REFINER
            elif y_dim == 2816:
                return cls.SDXL_BASE

        return cls.UNKNOWN

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))

    @classmethod
    def list_mo_support(cls):
        return [
            cls.SD1x.value,
            cls.SD2x768v.value,
            cls.SDXL_BASE.value,
            cls.SD3.value,
            cls.FLUX_DEV.value,
            cls.FLUX_SCHNELL.value,
            # cls.AuraFlow.value
        ]


def get_io_names(y_dim: int = None, extra_input: dict = {}):
    input_names = ["x", "timesteps", "context"]
    output_names = ["h"]
    dynamic_axes = {
        "x": {0: "batch", 2: "height", 3: "width"},
        "timesteps": {0: "batch"},
        "context": {0: "batch", 1: "num_embeds"},
    }

    if y_dim:
        input_names.append("y")
        dynamic_axes["y"] = {0: "batch"}

    for k in extra_input:
        input_names.append(k)
        dynamic_axes[k] = {0: "batch"}

    return input_names, output_names, dynamic_axes


def get_shape(
    model,
    model_type: ModelType,
    batch_size: int,
    width: int,
    height: int,
    context_multiplier: int = 1,
    num_video_frames: int = 12,
    y_dim: int = None,
    extra_input: dict = {},  # TODO batch_size*=2?
):
    context_len = 77
    context_dim = model.model.model_config.unet_config.get("context_dim", None)
    if model_type == ModelType.AuraFlow:
        context_len = 256
        context_dim = 2048
    elif model_type in (ModelType.FLUX_DEV, ModelType.FLUX_SCHNELL):
        context_len = 256
        context_dim = model.model.model_config.unet_config.get("context_in_dim", None)
    elif model_type == ModelType.SD3:
        context_embedder_config = model.model.model_config.unet_config.get(
            "context_embedder_config", None
        )
        if context_embedder_config is not None:
            context_dim = context_embedder_config.get("params", {}).get(
                "in_features", None
            )
            context_len = 154  # NOTE: SD3 can have 77 or 154 depending on which text encoders are used, this is why context_len_min stays 77
    elif model_type == ModelType.SVD:
        batch_size = batch_size * num_video_frames
        context_len = 1

    assert context_dim is not None

    input_channels = model.model.model_config.unet_config.get("in_channels", 4)
    inputs_shapes = (
        (batch_size, input_channels, height // 8, width // 8),
        (batch_size,),
        (batch_size, context_len * context_multiplier, context_dim),
    )
    if y_dim > 0:
        inputs_shapes += ((batch_size, y_dim),)

    for k in extra_input:
        inputs_shapes += ((batch_size,) + extra_input[k],)

    return inputs_shapes


def get_sample_input(input_shapes: tuple, dtype: torch.dtype, device: torch.device):
    inputs = ()
    for shape in input_shapes:
        inputs += (
            torch.zeros(
                shape,
                device=device,
                dtype=dtype,
            ),
        )

    return inputs


def get_dtype(model_type: ModelType):
    if model_type in (ModelType.FLUX_DEV, ModelType.FLUX_SCHNELL):
        return torch.bfloat16
    return torch.float16


def get_backbone(model, model_type, input_names, num_video_frames):
    unet = model.model.diffusion_model
    transformer_options = model.model_options["transformer_options"].copy()

    if model_type == ModelType.SVD:

        class UNET(torch.nn.Module):
            def forward(self, x, timesteps, context, y):
                return self.unet(
                    x,
                    timesteps,
                    context,
                    y,
                    num_video_frames=self.num_video_frames,
                    transformer_options=self.transformer_options,
                )

        svd_unet = UNET()
        svd_unet.num_video_frames = num_video_frames
        svd_unet.unet = unet
        svd_unet.transformer_options = transformer_options
        unet = svd_unet
    else:

        class UNET(torch.nn.Module):
            def forward(self, x, timesteps, context, *args):
                extras = input_names[3:]
                extra_args = {}
                for i in range(len(extras)):
                    extra_args[extras[i]] = args[i]
                return self.unet(
                    x,
                    timesteps,
                    context,
                    transformer_options=self.transformer_options,
                    **extra_args,
                )

        _unet = UNET()
        _unet.unet = unet
        _unet.transformer_options = transformer_options
        unet = _unet

    return unet


def get_extra_input(model, model_type):
    y_dim = model.model.adm_channels
    extra_input = {}
    if model_type in (ModelType.FLUX_DEV, ModelType.FLUX_SCHNELL):
        y_dim = model.model.model_config.unet_config.get("vec_in_dim", None)
        extra_input = {"guidance": ()}
    return y_dim, extra_input


def get_io_names_onnx(model: onnx.ModelProto):
    input_names = [i.name for i in model.graph.input]
    return input_names, None, None


def flux_convert_rope_weight_type(onnx_graph):
    graph = gs.import_onnx(onnx_graph)
    for node in graph.nodes:
        if node.op == "Einsum":
            node.inputs[1].dtype == "float32"
            print(node.name)
    return gs.export_onnx(graph)


def generate_fp8_scales(backbone):
    # temporary solution due to a known bug in torch.onnx._dynamo_export
    for _, module in backbone.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)) and (
            hasattr(module.input_quantizer, "_amax")
            and module.input_quantizer is not None
        ):
            module.input_quantizer._num_bits = 8
            module.weight_quantizer._num_bits = 8
            module.input_quantizer._amax = module.input_quantizer._amax * (127 / 448.0)
            module.weight_quantizer._amax = module.weight_quantizer._amax * (
                127 / 448.0
            )
        elif isinstance(module, attn_cls) and (
           hasattr(module.q_bmm_quantizer, "_amax")
           and module.q_bmm_quantizer is not None
        ):
            module.q_bmm_quantizer._num_bits = 8
            module.q_bmm_quantizer._amax = module.q_bmm_quantizer._amax * (127 / 448.0)
            module.k_bmm_quantizer._num_bits = 8
            module.k_bmm_quantizer._amax = module.k_bmm_quantizer._amax * (127 / 448.0)
            module.v_bmm_quantizer._num_bits = 8
            module.v_bmm_quantizer._amax = module.v_bmm_quantizer._amax * (127 / 448.0)
            module.softmax_bmm_quantizer._num_bits = 8
            module.softmax_bmm_quantizer._amax = module.softmax_bmm_quantizer._amax * (
                127 / 448.0
            )


def flux_convert_rope_weight_type(onnx_graph):
    graph = gs.import_onnx(onnx_graph)
    for node in graph.nodes:
        if node.op == "Einsum":
            node.inputs[1].dtype == "float32"
            print(node.name)
    return gs.export_onnx(graph)


def export_onnx(
    model,
    path,
    batch_size: int = 1,
    height: int = 512,
    width: int = 512,
    num_video_frames: int = 12,
    context_multiplier: int = 1,
    fp8: bool = False,
):
    model_type = ModelType.detect_version(model)
    if model_type == ModelType.UNKNOWN:
        raise Exception("ERROR: model not supported.")

    y_dim, extra_input = get_extra_input(model, model_type)
    input_names, output_names, dynamic_axes = get_io_names(y_dim, extra_input)
    dtype = get_dtype(model_type)
    device = comfy.model_management.get_torch_device()
    input_shapes = get_shape(
        model,
        model_type,
        batch_size,
        width,
        height,
        context_multiplier,
        num_video_frames,
        y_dim,
        extra_input,
    )
    inputs = get_sample_input(input_shapes, dtype, device)
    backbone = get_backbone(model, model_type, input_names, num_video_frames)

    dir, name = os.path.split(path)
    temp_path = os.path.join(folder_paths.get_temp_directory(), "{}".format(time.time()))
    onnx_temp = os.path.normpath(
        os.path.join(temp_path, name)
    )

    if not os.path.exists(temp_path):
        os.makedirs(temp_path)

    torch.onnx.export(
        backbone,
        inputs,
        onnx_temp,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        dynamic_axes=dynamic_axes,
    )

    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()

    onnx_model = onnx.load(onnx_temp, load_external_data=True)
    tensors_paths = _get_onnx_external_data_tensors(onnx_model)

    if tensors_paths:
        for tensor in tensors_paths:
            os.remove(os.path.join(onnx_temp, tensor))

    if fp8:
        onnx_model = convert_zp_fp8(onnx_model)
        if model_type not in (ModelType.FLUX_DEV, ModelType.FLUX_SCHNELL):
            onnx_model = convert_float_to_float16(
                onnx_model, keep_io_types=True, disable_shape_infer=True
            )
            graph = gs.import_onnx(onnx_model)
            cast_resize_io(graph)
            convert_fp16_io(graph)
            cast_fp8_mha_io(graph)
            onnx_model = gs.export_onnx(graph)
        else:
            flux_convert_rope_weight_type(onnx_model)

    onnx.save(
        onnx_model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=name + "_data",
        size_threshold=1024,
    )
