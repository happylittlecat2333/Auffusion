# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import warnings
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass

import torch
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    logging,
    randn_tensor,
    replace_example_docstring,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from huggingface_hub import snapshot_download
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, PNDMScheduler
from transformers import PretrainedConfig, AutoTokenizer
import torch.nn as nn
import os, json, PIL
import numpy as np
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from diffusers.utils.outputs import BaseOutput
import matplotlib.pyplot as plt


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name



def json_dump(data_json, json_save_path):
    with open(json_save_path, 'w') as f:
        json.dump(data_json, f, indent=4)
        f.close()


def json_load(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        f.close()
    return data                


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    if "t5" in model_class.lower():
        from transformers import T5EncoderModel
        return T5EncoderModel
    if "clap" in model_class.lower():
        from transformers import ClapTextModelWithProjection
        return ClapTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


class ConditionAdapter(nn.Module):
    def __init__(self, config):
        super(ConditionAdapter, self).__init__()
        self.config = config
        self.proj = nn.Linear(self.config["condition_dim"], self.config["cross_attention_dim"])
        self.norm = torch.nn.LayerNorm(self.config["cross_attention_dim"])
        print(f"INITIATED: ConditionAdapter: {self.config}")

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        ckpt_path = os.path.join(pretrained_model_name_or_path, "condition_adapter.pt")
        config = json.loads(open(config_path).read())
        instance = cls(config)
        instance.load_state_dict(torch.load(ckpt_path))
        print(f"LOADED: ConditionAdapter from {pretrained_model_name_or_path}")
        return instance

    def save_pretrained(self, pretrained_model_name_or_path):
        os.makedirs(pretrained_model_name_or_path, exist_ok=True)
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        ckpt_path = os.path.join(pretrained_model_name_or_path, "condition_adapter.pt")        
        json_dump(self.config, config_path)
        torch.save(self.state_dict(), ckpt_path)
        print(f"SAVED: ConditionAdapter {self.config['model_name']} to {pretrained_model_name_or_path}")


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg



LRELU_SLOPE = 0.1
MAX_WAV_VALUE = 32768.0


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_config(config_path):
    config = json.loads(open(config_path).read())
    config = AttrDict(config)
    return config

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)



class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        # self.conv_pre = weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))
        self.conv_pre = weight_norm(Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3)) # change: 80 --> 512
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            if (k-u) % 2 == 0:
                self.ups.append(weight_norm(
                    ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                    k, u, padding=(k-u)//2)))
            else:
                self.ups.append(weight_norm(
                    ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                    k, u, padding=(k-u)//2+1, output_padding=1)))
            
            # self.ups.append(weight_norm(
            #     ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
            #                     k, u, padding=(k-u)//2)))
            

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, subfolder=None):
        if subfolder is not None:
            pretrained_model_name_or_path = os.path.join(pretrained_model_name_or_path, subfolder)
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        ckpt_path = os.path.join(pretrained_model_name_or_path, "vocoder.pt")

        config = get_config(config_path)
        vocoder = cls(config)

        state_dict_g = torch.load(ckpt_path)
        vocoder.load_state_dict(state_dict_g["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        return vocoder
    
    @torch.no_grad()
    def inference(self, mels, lengths=None):
        self.eval()
        with torch.no_grad():
            wavs = self(mels).squeeze(1)

        wavs = (wavs.cpu().numpy() * MAX_WAV_VALUE).astype("int16")

        if lengths is not None:
            wavs = wavs[:, :lengths]

        return wavs



def normalize_spectrogram(
    spectrogram: torch.Tensor,
    max_value: float = 200, 
    min_value: float = 1e-5, 
    power: float = 1., 
) -> torch.Tensor:
    
    # Rescale to 0-1
    max_value = np.log(max_value) # 5.298317366548036
    min_value = np.log(min_value) # -11.512925464970229
    spectrogram = torch.clamp(spectrogram, min=min_value, max=max_value)
    data = (spectrogram - min_value) / (max_value - min_value)
    # Apply the power curve
    data = torch.pow(data, power)
    # 1D -> 3D
    data = data.repeat(3, 1, 1)
    # Flip Y axis: image origin at the top-left corner, spectrogram origin at the bottom-left corner
    data = torch.flip(data, [1])

    return data


def denormalize_spectrogram(
    data: torch.Tensor,
    max_value: float = 200, 
    min_value: float = 1e-5, 
    power: float = 1, 
) -> torch.Tensor:
    
    assert len(data.shape) == 3, "Expected 3 dimensions, got {}".format(len(data.shape))

    max_value = np.log(max_value)
    min_value = np.log(min_value)
    # Flip Y axis: image origin at the top-left corner, spectrogram origin at the bottom-left corner
    data = torch.flip(data, [1])    
    if data.shape[0] == 1:
        data = data.repeat(3, 1, 1)        
    assert data.shape[0] == 3, "Expected 3 channels, got {}".format(data.shape[0])
    data = data[0]
    # Reverse the power curve
    data = torch.pow(data, 1 / power)
    # Rescale to max value
    spectrogram = data * (max_value - min_value) + min_value

    return spectrogram

@staticmethod
def pt_to_numpy(images: torch.FloatTensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy image.
    """
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    return images

@staticmethod
def numpy_to_pil(images: np.ndarray) -> PIL.Image.Image:
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [PIL.Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [PIL.Image.fromarray(image) for image in images]

    return pil_images


def image_add_color(spec_img):
    cmap = plt.get_cmap('viridis')
    cmap_r = cmap.reversed()
    image = cmap(np.array(spec_img)[:,:,0])[:, :, :3]  # 省略透明度通道
    image = (image - image.min()) / (image.max() - image.min())
    image = PIL.Image.fromarray(np.uint8(image*255))
    return image


@dataclass
class PipelineOutput(BaseOutput):
    """
    Output class for audio pipelines.

    Args:
        audios (`np.ndarray`)
            List of denoised audio samples of a NumPy array of shape `(batch_size, num_channels, sample_rate)`.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    spectrograms: Union[List[np.ndarray], np.ndarray]
    audios: Union[List[np.ndarray], np.ndarray]



class AuffusionPipeline(DiffusionPipeline):

    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    In addition the pipeline inherits the following loading methods:
        - *Textual-Inversion*: [`loaders.TextualInversionLoaderMixin.load_textual_inversion`]
        - *LoRA*: [`loaders.LoraLoaderMixin.load_lora_weights`]
        - *Ckpt*: [`loaders.FromSingleFileMixin.from_single_file`]

    as well as the following saving methods:
        - *LoRA*: [`loaders.LoraLoaderMixin.save_lora_weights`]

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor", "text_encoder_list", "tokenizer_list", "adapter_list", "vocoder"]

    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,        
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        text_encoder_list: Optional[List[Callable]] = None,
        tokenizer_list: Optional[List[Callable]] = None,
        vocoder: Generator = None,
        requires_safety_checker: bool = False,        
        adapter_list: Optional[List[Callable]] = None,
        tokenizer_model_max_length: Optional[int] = 77, # 77 is the default value for the CLIPTokenizer(and set for other models)
    ):
        super().__init__()

        self.text_encoder_list = text_encoder_list
        self.tokenizer_list = tokenizer_list
        self.vocoder = vocoder
        self.adapter_list = adapter_list
        self.tokenizer_model_max_length = tokenizer_model_max_length

        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)


    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str = "auffusion/auffusion",
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        if not os.path.isdir(pretrained_model_name_or_path):
            pretrained_model_name_or_path = snapshot_download(pretrained_model_name_or_path) 
        
        vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
        feature_extractor = CLIPImageProcessor.from_pretrained(pretrained_model_name_or_path, subfolder="feature_extractor")
        scheduler = PNDMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")

        vocoder = Generator.from_pretrained(pretrained_model_name_or_path, subfolder="vocoder").to(device, dtype)

        text_encoder_list, tokenizer_list, adapter_list = [], [], []
        
        condition_json_path = os.path.join(pretrained_model_name_or_path, "condition_config.json")
        condition_json_list = json.loads(open(condition_json_path).read())
        
        for i, condition_item in enumerate(condition_json_list):
            
            # Load Condition Adapter
            text_encoder_path = os.path.join(pretrained_model_name_or_path, condition_item["text_encoder_name"])
            tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
            tokenizer_list.append(tokenizer)
            text_encoder_cls = import_model_class_from_model_name_or_path(text_encoder_path)
            text_encoder = text_encoder_cls.from_pretrained(text_encoder_path).to(device, dtype)
            text_encoder_list.append(text_encoder)
            print(f"LOADING CONDITION ENCODER {i}")

            # Load Condition Adapter
            adapter_path = os.path.join(pretrained_model_name_or_path, condition_item["condition_adapter_name"])
            adapter = ConditionAdapter.from_pretrained(adapter_path).to(device, dtype)
            adapter_list.append(adapter)
            print(f"LOADING CONDITION ADAPTER {i}")


        pipeline = cls(
            vae=vae,
            unet=unet,
            text_encoder_list=text_encoder_list,
            tokenizer_list=tokenizer_list,
            vocoder=vocoder,
            adapter_list=adapter_list,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=feature_extractor,
        )
        pipeline = pipeline.to(device, dtype)

        return pipeline
    

    def to(self, device, dtype=None):
        super().to(device, dtype)

        self.vocoder.to(device, dtype)

        for text_encoder in self.text_encoder_list:
            text_encoder.to(device, dtype)
       
        if self.adapter_list is not None:
            for adapter in self.adapter_list:
                adapter.to(device, dtype)

        return self


    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding.

        When this option is enabled, the VAE will split the input tensor into tiles to compute decoding and encoding in
        several steps. This is useful to save a large amount of memory and to allow the processing of larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        Note that offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.14.0"):
            from accelerate import cpu_offload
        else:
            raise ImportError("`enable_sequential_cpu_offload` requires `accelerate v0.14.0` or higher")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            cpu_offload(cpu_offloaded_model, device)

        if self.safety_checker is not None:
            cpu_offload(self.safety_checker, execution_device=device, offload_buffers=True)

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        hook = None
        for cpu_offloaded_model in [self.text_encoder, self.unet, self.vae]:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        if self.safety_checker is not None:
            _, hook = cpu_offload_with_hook(self.safety_checker, device, prev_module_hook=hook)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):

        assert len(self.text_encoder_list) == len(self.tokenizer_list), "Number of text_encoders must match number of tokenizers"
        if self.adapter_list is not None:
            assert len(self.text_encoder_list) == len(self.adapter_list), "Number of text_encoders must match number of adapters"

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        def get_prompt_embeds(prompt_list, device):
            if isinstance(prompt_list, str):
                prompt_list = [prompt_list]

            prompt_embeds_list = []
            for prompt in prompt_list:
                encoder_hidden_states_list = []

                # Generate condition embedding
                for j in range(len(self.text_encoder_list)):
                    # get condition embedding using condition encoder
                    input_ids = self.tokenizer_list[j](prompt, return_tensors="pt").input_ids.to(device)            
                    cond_embs = self.text_encoder_list[j](input_ids).last_hidden_state # [bz, text_len, text_dim]
                    # padding to max_length
                    if cond_embs.shape[1] < self.tokenizer_model_max_length: 
                        cond_embs = torch.functional.F.pad(cond_embs, (0, 0, 0, self.tokenizer_model_max_length - cond_embs.shape[1]), value=0)
                    else:
                        cond_embs = cond_embs[:, :self.tokenizer_model_max_length, :]

                    # use condition adapter
                    if self.adapter_list is not None:
                        cond_embs = self.adapter_list[j](cond_embs)
                        encoder_hidden_states_list.append(cond_embs)

                prompt_embeds = torch.cat(encoder_hidden_states_list, dim=1)
                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
            return prompt_embeds


        if prompt_embeds is None:           
            prompt_embeds = get_prompt_embeds(prompt, device)

        prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:

            if negative_prompt is None:
                negative_prompt_embeds = torch.zeros_like(prompt_embeds).to(dtype=prompt_embeds.dtype, device=device)

            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                negative_prompt_embeds = get_prompt_embeds(negative_prompt, device)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.unet.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds
            

    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    def decode_latents(self, latents):
        warnings.warn(
            "The decode_latents method is deprecated and will be removed in a future version. Please"
            " use VaeImageProcessor instead",
            FutureWarning,
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )



    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = 256,
        width: Optional[int] = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pt",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        duration: Optional[float] = 10,
    ):
       
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        audio_length = int(duration * 16000)


        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )
       

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds
        )
        
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()


        # Generate audio
        spectrograms, audios = [], []
        for img in image:
            spectrogram = denormalize_spectrogram(img)
            audio = self.vocoder.inference(spectrogram, lengths=audio_length)[0]
            audios.append(audio)
            spectrograms.append(spectrogram)

        # Convert to PIL
        images = pt_to_numpy(image)    
        images = numpy_to_pil(images)
        images = [image_add_color(image) for image in images]

        if not return_dict:
            return (images, audios, spectrograms)
                    

        return PipelineOutput(images=images, audios=audios, spectrograms=spectrograms)




