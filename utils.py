import os, json
import math, random
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from transformers import CLIPTextModel
from transformers import PretrainedConfig


def pad_spec(spec, spec_length, pad_value=0, random_crop=True): # spec: [3, mel_dim, spec_len]
    assert spec_length % 8 == 0, "spec_length must be divisible by 8"
    if spec.shape[-1] < spec_length:
        # pad spec to spec_length
        spec = F.pad(spec, (0, spec_length - spec.shape[-1]), value=pad_value)
    else:
        # random crop
        if random_crop:
            start = random.randint(0, spec.shape[-1] - spec_length)
            spec = spec[:, :, start:start+spec_length]
        else:
            spec = spec[:, :, :spec_length]
    return spec


def load_spec(spec_path):
    if spec_path.endswith(".pt"):
        spec = torch.load(spec_path, map_location="cpu")
    elif spec_path.endswith(".npy"):
        spec = torch.from_numpy(np.load(spec_path))
    else:
        raise ValueError(f"Unknown spec file type {spec_path}")
    assert len(spec.shape) == 3, f"spec shape must be [3, mel_dim, spec_len], got {spec.shape}"
    if spec.size(0) == 1:
        spec = spec.repeat(3, 1, 1)
    return spec


def random_crop_spec(spec, target_spec_length, pad_value=0, frame_per_sec=100, time_step=5): # spec: [3, mel_dim, spec_len]
    assert target_spec_length % 8 == 0, "spec_length must be divisible by 8"

    spec_length = spec.shape[-1]
    full_s = math.ceil(spec_length / frame_per_sec / time_step) * time_step # get full seconds(ceil)
    start_s = random.randint(0, math.floor(spec_length / frame_per_sec / time_step)) * time_step # random get start seconds

    end_s = min(start_s + math.ceil(target_spec_length / frame_per_sec), full_s) # get end seconds

    spec = spec[:, :, start_s * frame_per_sec : end_s * frame_per_sec] # get spec in seconds(crop more than target_spec_length because ceiling)

    if spec.shape[-1] < target_spec_length:
        spec = F.pad(spec, (0, target_spec_length - spec.shape[-1]), value=pad_value) # pad to target_spec_length
    else:     
        spec = spec[:, :, :target_spec_length] # crop to target_spec_length

    return spec, int(start_s), int(end_s), int(full_s)



def load_condion_embed(text_embed_path):
    if text_embed_path.endswith(".pt"):
        text_embed_list = torch.load(text_embed_path, map_location="cpu")
    elif text_embed_path.endswith(".npy"):
        text_embed_list = torch.from_numpy(np.load(text_embed_path))
    else:
        raise ValueError(f"Unknown text embedding file type {text_embed_path}")
    if type(text_embed_list) == list:
        text_embed = random.choice(text_embed_list)
    if len(text_embed.shape) == 3: # [1, text_len, text_dim]
        text_embed = text_embed.squeeze(0) # random choice and return text_emb: [text_len, text_dim]
    return text_embed.detach().cpu()
    

def process_condition_embed(cond_emb, max_length): # [text_len, text_dim], Padding 0 and random drop by CFG
    if cond_emb.shape[0] < max_length:
        cond_emb = F.pad(cond_emb, (0, 0, 0, max_length - cond_emb.shape[0]), value=0)
    else:
        cond_emb = cond_emb[:max_length, :]
    return cond_emb
                

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
    


def str2bool(string):
    str2val = {"True": True, "False": False, "true": True, "false": False, "none": False, "None": False}
    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")
    

def str2str(string):
    if string.lower() == "none" or string.lower() == "null" or string.lower() == "false" or string == "":
        return None
    else:
        return string    


def json_dump(data_json, json_save_path):
    with open(json_save_path, 'w') as f:
        json.dump(data_json, f, indent=4)
        f.close()


def json_load(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        f.close()
    return data


def load_json_list(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f.readlines()]
    

def save_json_list(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d) + '\n')
    

def multiprocess_function(func, func_args, n_jobs=32):  
    with Pool(processes=n_jobs) as p:
            with tqdm(total=len(func_args)) as pbar:
                for i, _ in enumerate(p.imap_unordered(func, func_args)):
                    pbar.update()


def image_add_color(spec_img):
    cmap = plt.get_cmap('viridis')
    cmap_r = cmap.reversed()
    image = cmap(np.array(spec_img)[:,:,0])[:, :, :3]  # 省略透明度通道
    image = (image - image.min()) / (image.max() - image.min())
    image = Image.fromarray(np.uint8(image*255))
    return image


@staticmethod
def pt_to_numpy(images: torch.FloatTensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy image.
    """
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    return images


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

### CODE FOR INPAITING ###
def normalize(images):
    """
    Normalize an image array to [-1,1].
    """
    if images.min() >= 0:
        return 2.0 * images - 1.0
    else:
        return images

def denormalize(images):
    """
    Denormalize an image array to [0,1].
    """
    if images.min() < 0:
        return (images / 2 + 0.5).clamp(0, 1)
    else:
        return images.clamp(0, 1)     
    

def prepare_mask_and_masked_image(image, mask):
    """
    Prepare a binary mask and the masked image.
    
    Parameters:
    - image (torch.Tensor): The input image tensor of shape [3, height, width] with values in the range [0, 1].
    - mask (torch.Tensor): The input mask tensor of shape [1, height, width].
    
    Returns:
    - tuple: A tuple containing the binary mask and the masked image.
    """
    # Noralize image to [0,1]
    if image.max() > 1:
        image = (image - image.min()) / (image.max() - image.min())
    # Normalize image from [0,1] to [-1,1]
    if image.min() >= 0:
        image = normalize(image)    
    # Apply the mask to the image
    masked_image = image * (mask < 0.5)
    
    return mask, masked_image


def torch_to_pil(image):
    """
    Convert a torch tensor to a PIL image.
    """
    if image.min() < 0:
        image = denormalize(image)

    return transforms.ToPILImage()(image.cpu().detach().squeeze())



# class TextEncoderAdapter(nn.Module):    
#     def __init__(self, hidden_size, cross_attention_dim=768):
#         super(TextEncoderAdapter, self).__init__()
#         self.hidden_size = hidden_size
#         self.cross_attention_dim = cross_attention_dim
#         self.proj = nn.Linear(self.hidden_size, self.cross_attention_dim)
#         self.norm = torch.nn.LayerNorm(self.cross_attention_dim)

#     def forward(self, x):
#         x = self.proj(x)
#         x = self.norm(x)
#         return x
    
#     def save_pretrained(self, save_directory, subfolder=""):
#         if subfolder:
#             save_directory = os.path.join(save_directory, subfolder)
#         os.makedirs(save_directory, exist_ok=True)
#         ckpt_path = os.path.join(save_directory, "adapter.pt")
#         config_path = os.path.join(save_directory, "config.json")
#         config = {"hidden_size": self.hidden_size, "cross_attention_dim": self.cross_attention_dim}
#         json_dump(config, config_path)
#         torch.save(self.state_dict(), ckpt_path)
#         print(f"Saving adapter model to {ckpt_path}")

#     @classmethod
#     def from_pretrained(cls, load_directory, subfolder=""):
#         if subfolder:
#             load_directory = os.path.join(load_directory, subfolder)
#         ckpt_path = os.path.join(load_directory, "adapter.pt")
#         config_path = os.path.join(load_directory, "config.json")
#         config = json_load(config_path)
#         instance = cls(**config)
#         instance.load_state_dict(torch.load(ckpt_path))
#         print(f"Loading adapter model from {ckpt_path}")
#         return instance          



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
        config = json_load(config_path)
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
        print(f"SAVED: ConditionAdapter {self.config['condition_adapter_name']} to {pretrained_model_name_or_path}")


# class TextEncoderWrapper(CLIPTextModel):
#     def __init__(self, text_encoder, text_encoder_adapter):
#         super().__init__(text_encoder.config)
#         self.text_encoder = text_encoder
#         self.adapter = text_encoder_adapter

#     def forward(self, input_ids, **kwargs):
#         outputs = self.text_encoder(input_ids, **kwargs)
#         adapted_output = self.adapter(outputs[0])
#         return [adapted_output] # to compatible with last_hidden_state

