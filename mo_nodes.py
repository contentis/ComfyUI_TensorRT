import os
from dataclasses import dataclass

import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.supported_models
import folder_paths

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq

from .mo_utils.utils import load_calib_prompts, filter_func, quantize_lvl
from .mo_utils.config import (
    SD_FP8_FP16_DEFAULT_CONFIG,
    SD_FP8_FP32_DEFAULT_CONFIG,
    get_int8_config,
    set_stronglytyped_precision,
)
from .mo_utils.ksampler_t2i import DiffusionPipe
from .mo_utils.flux_sampler_t2i import DiffusionPipe as FluxDiffusionPipe
from .mo_utils.attention_plugin import register_quant_modules

from .onnx_utils.export import ModelType, generate_fp8_scales

MAX_RESOLUTION = 16384


# add output directory to modelopt search path<<<<<<
if "modelopt" in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["modelopt"][0].append(
        os.path.join(folder_paths.models_dir, "modelopt")
    )
    folder_paths.folder_names_and_paths["modelopt"][1].add(".pt")
else:
    folder_paths.folder_names_and_paths["modelopt"] = (
        [os.path.join(folder_paths.models_dir, "modelopt")],
        {".pt"},
    )

if not os.path.exists(folder_paths.folder_names_and_paths["modelopt"][0][0]):
    os.makedirs(folder_paths.folder_names_and_paths["modelopt"][0][0])


def do_calibrate(pipe, calibration_prompts, **kwargs):
    for i_th, prompts in enumerate(calibration_prompts):
        if i_th >= kwargs["calib_size"]:
            return
        pipe(
            positive_prompt=prompts[0],
            num_inference_steps=kwargs["n_steps"],
            negative_prompt="normal quality, low quality, worst quality, low res, blurry, nsfw, nude",
        )


@dataclass
class MOConfig:
    width: int
    height: int
    seed: int
    steps: int
    cfg: float
    sampler_name: str
    scheduler: comfy.samplers.KSampler.SAMPLERS
    denoise: comfy.samplers.KSampler.SCHEDULERS
    percentile: float
    alpha: float
    calib_size: int
    collect_method: str
    quant_level: float


# TODO validate the default confgis
DEFAULT_CONFIGS = {
    ModelType.SD1x.value: {
        "int8": MOConfig(512, 512, 42, 30, 7.5, "euler", "normal", 1.0, 1.0, 1.0, 128, "default", 2),
        "fp8": MOConfig(512, 512, 42, 30, 7.5, "euler", "normal", 1.0, 1.0, 1.0, 256, "default", 2),
    },
    ModelType.SD2x768v.value: {
        "int8": MOConfig(768, 768, 42, 30, 7.5, "euler", "normal", 1.0, 1.0, 0.8, 64, "min-mean", 2.5),
        "fp8": MOConfig(768, 768, 42, 30, 7.5, "euler", "normal", 1.0, 1.0, 1.0, 128, "default", 2.5),     
    },
    ModelType.SDXL_BASE.value: {
        "int8": MOConfig(1024, 1024, 42, 30, 7.5, "euler", "normal", 1.0, 1.0, 0.8, 64, "min-mean", 2.5),
        "fp8": MOConfig(1024, 1024, 42, 30, 7.5, "euler", "normal", 1.0, 1.0, 1.0, 128, "default", 3),       
    },
    ModelType.SD3.value: {
        "int8": MOConfig(1024, 1024, 42, 30, 5.5, "euler", "sgm_uniform", 1.0, 1.0, 0.8, 64, "min-mean", 2.5),
        "fp8": MOConfig(1024, 1024, 42, 30, 5.5, "euler", "sgm_uniform", 1.0, 1.0, 1.0, 128, "default", 3),            
    },
    ModelType.FLUX_DEV.value: {
        "int8": MOConfig(1024, 1024, 42, 30, 3.5, "euler", "simple", 1.0, 1.0, 0.8, 64, "min-mean", 2.5),
        "fp8": MOConfig(1024, 1024, 42, 30, 3.5, "euler", "simple", 1.0, 1.0, 1.0, 128, "default", 3),       
    },
    ModelType.FLUX_SCHNELL.value: {
        "int8": MOConfig(1024, 1024, 42, 4, 3.5, "euler", "simple", 1.0, 1.0, 0.8, 64, "min-mean", 2.5),
        "fp8": MOConfig(1024, 1024, 42, 4, 3.5, "euler", "simple", 1.0, 1.0, 1.0, 128, "default", 3),       
    },
    ModelType.AuraFlow.value: {
        "int8": MOConfig(1024, 1024, 42, 30, 3.5, "euler", "sgm_uniform", 1.0, 1.0, 0.8, 64, "min-mean", 2.5),
        "fp8": MOConfig(1024, 1024, 42, 30, 3.5, "euler", "sgm_uniform", 1.0, 1.0, 1.0, 128, "default", 3),       
    },
}


class BaseQuantizer:
    @classmethod
    def INPUT_TYPES(s):
        raise NotImplementedError

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "quantize"
    OUTPUT_NODE = False
    CATEGORY = "TensorRT"

    def _quantize(
        self,
        name,
        model,
        clip,
        model_type,
        format,
        width,
        height,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        percentile,
        alpha,
        calib_size,
        collect_method,
        quant_level,
        calib_prompts_path="default",
    ):
        comfy.model_management.unload_all_models()
        comfy.model_management.load_models_gpu([model], force_patch_weights=True, force_full_load=True)
        backbone = model.model.diffusion_model
        device = comfy.model_management.get_torch_device()

        # This is a list of prompts
        if calib_prompts_path == "default":
            calib_prompts_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "mo_utils/calib_prompts.txt",
            )

        calib_prompts = load_calib_prompts(1, calib_prompts_path)
        quant_level = float(quant_level)

        if quant_level == 4.0:
            assert format != "int8", "We only support fp8 for Level 4 Quantization"

        extra_step = (
            1 if model_type == ModelType.SD1x or model_type == ModelType.SD2x768v else 0
        )  # Depending on the scheduler. some schedulers will do n+1 steps

        if format == "int8":
            # Making sure to use global_min in the calibrator for SD 1.5 or SD 2.1
            if collect_method == "default":
                collect_method = "min-mean"
            if model_type == ModelType.SD1x or model_type == ModelType.SD2x768v:
                collect_method = "global_min"
            quant_config = get_int8_config(
                backbone,
                quant_level,
                alpha,
                percentile,
                steps + extra_step,
                collect_method=collect_method,
            )
        elif format == "fp8":
            if collect_method == "default":
                quant_config = (
                    SD_FP8_FP32_DEFAULT_CONFIG
                    if model_type == ModelType.SD2x768v
                    else SD_FP8_FP16_DEFAULT_CONFIG
                )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        if model_type in (ModelType.FLUX_DEV, ModelType.FLUX_SCHNELL):
            pipe = FluxDiffusionPipe(
                model,
                clip,
                1,
                height,
                width,
                seed,
                cfg,
                sampler_name,
                scheduler,
                denoise,
                device=device,
            )
        else:
            pipe = DiffusionPipe(
                model,
                clip,
                1,
                width,
                height,
                seed,
                cfg,
                sampler_name,
                scheduler,
                denoise,
                device=device,
                is_sd3=model_type in (ModelType.SD3,),
            )

        def forward_loop(backbone):
            pipe.model.model.diffusion_model = backbone
            do_calibrate(
                pipe=pipe,
                calibration_prompts=calib_prompts,
                calib_size=calib_size,
                n_steps=steps,
            )

        if model_type in (ModelType.FLUX_DEV, ModelType.FLUX_SCHNELL):
            set_stronglytyped_precision(quant_config, "BFloat16")

        register_quant_modules()
        mtq.quantize(backbone, quant_config, forward_loop)

        out_path = os.path.join(
            folder_paths.folder_names_and_paths["modelopt"][0][0],
            "{}_{}.pt".format(name, format),
        )
        print(out_path)
        mto.save(backbone, out_path)

        quantize_lvl(backbone, quant_level)
        mtq.disable_quantizer(backbone, filter_func)

        if format == "fp8":
            generate_fp8_scales(backbone)

        return (model,)


class ModelOptEzQuantizer(BaseQuantizer):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "name": ("STRING", {"default": "mo_quantization"}),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "model_type": (ModelType.list_mo_support(), {}),
                "format": (["int8", "fp8"],),
            },
            "optional": {
                "calib_prompts_path": (
                    "STRING",
                    {"forceInput": True, "default": "default"},
                ),
            },
        }

    def quantize(
        self, name, model, clip, model_type, format, calib_prompts_path="default"
    ):
        config = DEFAULT_CONFIGS[model_type][format]
        print(f"INFO: Running quantization with following config: {config}")
        return super()._quantize(
            name,
            model,
            clip,
            model_type,
            format,
            calib_prompts_path=calib_prompts_path,
            **config.__dict__,
        )


class ModelOptQuantizer(BaseQuantizer):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "name": ("STRING", {"default": "mo_quantization"}),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "model_type": (ModelType.list_mo_support(), {}),
                "format": (["int8", "fp8"],),
                "quant_level": ([4.0, 3.0, 2.5, 2.0, 1.0], {"default": 2.5}),
                "width": (
                    "INT",
                    {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8},
                ),
                "height": (
                    "INT",
                    {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "percentile": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "alpha": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "calib_size": ("INT", {"default": 128, "min": 1, "max": 10000}),
                "collect_method": (
                    ["min-mean", "min-max" "mean-max", "global_min", "default"],
                    {"default": "default"},
                ),
            },
            "optional": {
                "calib_prompts_path": (
                    "STRING",
                    {"forceInput": True, "default": "default"},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "quantize"
    OUTPUT_NODE = False
    CATEGORY = "TensorRT"

    def quantize(
        self,
        name,
        model,
        clip,
        model_type,
        format,
        width,
        height,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        percentile,
        alpha,
        calib_size,
        collect_method,
        quant_level,
        calib_prompts_path="default",
    ):
        return super()._quantize(
            name,
            model,
            clip,
            model_type,
            format,
            width,
            height,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            denoise,
            percentile,
            alpha,
            calib_size,
            collect_method,
            quant_level,
            calib_prompts_path,
        )


class ModelOptLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "quantized_ckpt": (folder_paths.get_filename_list("modelopt"),),
                "quant_level": ([4.0, 3.0, 2.5, 2.0, 1.0], {"default": 2.5}),
                "format": (["int8", "fp8"],),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    OUTPUT_NODE = True
    CATEGORY = "TensorRT"

    def load(self, model, quantized_ckpt, quant_level, format):
        _state_key = "_modelopt_state"

        quantized_ckpt_path = folder_paths.get_full_path("modelopt", quantized_ckpt)
        if not os.path.isfile(quantized_ckpt_path):
            raise FileNotFoundError(f"File {quantized_ckpt_path} does not exist")

        comfy.model_management.unload_all_models()
        comfy.model_management.load_models_gpu([model], force_patch_weights=True, force_full_load=True)
        backbone = model.model.diffusion_model
        register_quant_modules()

        # Lets restore the quantized model
        if hasattr(model, _state_key):
            print("INFO: Model already has ModeOpt State loaded. Skipping...")
        else:
            mto.restore(backbone, quantized_ckpt_path)
        quantize_lvl(backbone, quant_level)
        mtq.disable_quantizer(backbone, filter_func)

        if format == "fp8":
            generate_fp8_scales(backbone)

        return (model,)


NODE_CLASS_MAPPINGS = {
    "ModelOptQuantizer": ModelOptQuantizer,
    "ModelOptEzQuantizer": ModelOptEzQuantizer,
    "ModelOptLoader": ModelOptLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelOptQuantizer": "ModelOpt Advanced Quantizer",
    "ModelOptEzQuantizer": "ModelOpt Ez Quantizer",
    "ModelOptLoader": "Model OptLoader",
}
