from nodes import CLIPTextEncode
from comfy_extras.nodes_custom_sampler import (
    BasicGuider,
    SamplerCustomAdvanced,
    BasicScheduler,
    RandomNoise,
    KSamplerSelect,
)

from comfy_extras.nodes_flux import FluxGuidance
from comfy_extras.nodes_model_advanced import ModelSamplingFlux
from comfy_extras.nodes_sd3 import EmptySD3LatentImage


class DiffusionPipe:
    def __init__(
        self,
        model,
        clip,
        batch_size,
        width,
        height,
        seed,
        cfg=3.5,
        sampler_name="euler",
        scheduler="simple",
        denoise=1.0,
        max_shift=1.15,
        base_shift=0.5,
        device="cuda",
    ) -> None:
        self.clip = clip
        self.clip_node = CLIPTextEncode()
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.max_shift = max_shift
        self.base_shift = base_shift
        self.device = device

        self.seed = seed
        self.cfg = cfg
        self.scheduler_name = scheduler
        self.denoise = denoise

        (self.model,) = ModelSamplingFlux().patch(
            model, self.max_shift, self.base_shift, self.width, self.height
        )
        (self.ksampler,) = KSamplerSelect().get_sampler(sampler_name)
        self.latent_node = EmptySD3LatentImage()
        self.guidance = FluxGuidance()
        self.sampler = SamplerCustomAdvanced()
        self.scheduler_node = BasicScheduler()
        self.guider = BasicGuider()
        self.noise_generator = RandomNoise()

    def __call__(self, num_inference_steps, positive_prompt, *args, **kwargs):
        (positive,) = self.clip_node.encode(self.clip, positive_prompt)
        (latent_image,) = self.latent_node.generate(
            self.width, self.height, self.batch_size
        )
        (noise,) = self.noise_generator.get_noise(self.seed)

        (conditioning,) = self.guidance.append(positive, self.cfg)
        (sigmas,) = self.scheduler_node.get_sigmas(
            self.model, self.scheduler_name, num_inference_steps, self.denoise
        )
        (guider,) = self.guider.get_guider(self.model, conditioning)

        out, denoised_out = self.sampler.sample(
            noise, guider, self.ksampler, sigmas, latent_image
        )

        return out["samples"]