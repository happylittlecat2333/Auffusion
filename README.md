# Auffusion: Leveraging the Power of Diffusion and Large Language Models for Text-to-Audio Generation


[Paper](https://arxiv.org/pdf/2401.01044) | [Model](https://huggingface.co/auffusion/auffusion) | [Website and Examples](https://auffusion.github.io) | [Audio Manipulation Notebooks](https://github.com/happylittlecat2333/Auffusion/tree/main/notebooks/README.md) | [Hugging Face Models](https://huggingface.co/auffusion) | [Google Colab](https://colab.research.google.com/drive/1JEPHT_AvHZxvlaZAsetkBnMrzCGMRKaf?usp=sharing)


## Description

**Auffusion** is a latent diffusion model (LDM) for text-to-audio (TTA) generation. **Auffusion** can generate realistic audios including human sounds, animal sounds, natural and artificial sounds and sound effects from textual prompts. We introduce Auffusion, a TTA system adapting T2I model frameworks to TTA task, by effectively leveraging their inherent generative strengths and precise cross-modal alignment. Our objective and subjective evaluations demonstrate that Auffusion surpasses previous TTA approaches using limited data and computational resource. We release our model, inference code, and pre-trained checkpoints for the research community.


<p align="center">
  <img src=img/overview.png />
</p>


## 🚀 News
- **2024/01/02**: 📣 [Colab notebooks](notebooks/README.md) for audio manipulation is released. Feel free to try!

- **2023/12/31**: 📣 Auffusion release.[Demo website](https://auffusion.github.io/) and 3 models are released in [Hugging Face](https://huggingface.co/auffusion). 


## Auffusion Model Family

| Model Name                 | Model Path                                                                                                              |
|----------------------------|------------------------------------------------------------------------------------------------------------------------ |
| Auffusion                  | [https://huggingface.co/auffusion/auffusion](https://huggingface.co/auffusion/auffusion)                                |
| Auffusion-Full             | [https://huggingface.co/auffusion/auffusion-full](https://huggingface.co/auffusion/auffusion-full)                      |
| Auffusion-Full-no-adapter  | [https://huggingface.co/auffusion/auffusion-full-no-adapter](https://huggingface.co/auffusion/auffusion-full-no-adapter)|



## 📀 Prerequisites

Our code is built on pytorch version 2.0.1. We mention `torch==2.0.1` in the requirements file but you might need to install a specific cuda version of torch depending on your GPU device type. We also depend on `diffusers==0.18.2`.

Install `requirements.txt`.

```bash
git clone https://github.com/happylittlecat2333/Auffusion/
cd Auffusion
pip install -r requirements.txt
```

You might also need to install `libsndfile1` for soundfile to work properly in linux:

```bash
(sudo) apt-get install libsndfile1
```


## ⭐ Quickstart Guide

Download the **Auffusion** model and generate audio from a text prompt:

```python
import IPython, torch
import soundfile as sf
from auffusion_pipeline import AuffusionPipeline

pipeline = AuffusionPipeline.from_pretrained("auffusion/auffusion")

prompt = "Birds singing sweetly in a blooming garden"
output = pipeline(prompt=prompt)
audio = output.audios[0]
sf.write(f"{prompt}.wav", audio, samplerate=16000)
IPython.display.Audio(data=audio, rate=16000)
```

The auffusion model will be automatically downloaded from Hugging Face and saved in cache. Subsequent runs will load the model directly from cache.

The `generate` function uses 100 steps and 7.5 guidance_scale by default to sample from the latent diffusion model. You can also vary parameters for different results.

```python
prompt = "Rolling thunder with lightning strikes"
output = pipeline(prompt=prompt, num_inference_steps=100, guidance_scale=7.5)
audio = output.audios[0]
IPython.display.Audio(data=audio, rate=16000)
```


More generated samples are shown [here](https://auffusion.github.io). You can also try out the [colab notebook](https://colab.research.google.com/drive/1JEPHT_AvHZxvlaZAsetkBnMrzCGMRKaf?usp=sharing) to generate your own audio samples.


## 🐍 How to make inferences?

### From our released checkpoints in Hugging Face Hub

To perform audio generation in AudioCaps test set from our Hugging Face checkpoints:

```bash
python inference.py \
--pretrained_model_name_or_path="auffusion/auffusion" \
--test_data_dir="./data/test_audiocaps.raw.json" \
--output_dir="./output/auffusion_hf" \
--enable_xformers_memory_efficient_attention \
```

### Note

We use the evaluation tools from [https://github.com/haoheliu/audioldm_eval](https://github.com/haoheliu/audioldm_eval) to evaluate our models, and we adopt [https://huggingface.co/laion/clap-htsat-unfused](https://huggingface.co/laion/clap-htsat-unfused) to compute CLAP score.

Some data instances originally released in AudioCaps have since been removed from YouTube and are no longer available. We thus evaluated our models on all the instances which were available as June, 2023.

## Audio Manipulation

We show some examples of audio manipulation using Auffusion. Current audio manipulation methods include:

- Text-to-audio generation: [notebook](notebooks/text_to_audio.ipynb) or [colab](https://colab.research.google.com/drive/1JEPHT_AvHZxvlaZAsetkBnMrzCGMRKaf?usp=sharing)
- Text-guided style transfer: [notebook](notebooks/img2img.ipynb) or [colab](https://colab.research.google.com/drive/1VjgryIz7kSXDzgCClqtqVgoDXIECeG0M?usp=sharing)
- Audio inpainting: [notebook](notebooks/inpainting.ipynb) or [colab](https://colab.research.google.com/drive/1NsqeiutoAynhtaZnlhzBdTXtZ27tQxVc?usp=sharing)
- attention-based word swap control: [notebook](notebooks/word_swap.ipynb) or [colab](https://colab.research.google.com/drive/18CtUoBMsPbgzeI-o0wHDYTtaErnq9KoI?usp=sharing)
- attention-based reweight control: [notebook](notebooks/reweight.ipynb) or [colab](https://colab.research.google.com/drive/18CtUoBMsPbgzeI-o0wHDYTtaErnq9KoI?usp=sharing)

The audio manipulation code examples can all be found in [notebooks](notebooks/README.md).

# TODO

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/happylittlecat)

- [x] Publish demo website and arxiv link.
- [x] Publish Auffusion and Auffusion-Full checkpoints.
- [x] Add text-guided style transfer.
- [x] Add audio-to-audio generation.
- [x] Add audio inpainting.
- [x] Add word_swap and reweight prompt2prompt-based control.
- [ ] Add audio super-resolution.
- [ ] Build Gradio web application.
- [ ] Add audio-to-audio, inpainting into Gradio web application.
- [ ] Add style-transfer into Gradio web application.
- [ ] Add audio super-resolution into Gradio web application.
- [ ] Add prompt2prompt-based control into Gradio web application.
- [ ] Add data preprocess and training code.



## 📚 Citation
Please consider citing the following article if you found our work useful:

```bibtex
@article{xue2024auffusion,
  title={Auffusion: Leveraging the Power of Diffusion and Large Language Models for Text-to-Audio Generation}, 
  author={Jinlong Xue and Yayue Deng and Yingming Gao and Ya Li},
  journal={arXiv preprint arXiv:2401.01044},
  year={2024}
}
```

## 🙏 Acknowledgement
Part of the code is borrowed from the following repos. We would like to thank the authors of these repos for their contribution.

- https://github.com/huggingface/diffusers

- https://github.com/huggingface/transformers

- https://github.com/google/prompt-to-prompt

- https://github.com/declare-lab/tango

- https://github.com/riffusion/riffusion

-  https://github.com/haoheliu/audioldm_eval


## Contact

If you have any problems regarding the paper, code, models, or the project itself, please feel free to open an issue or contact [Jinlong Xue](mailto:jinlong_xue@bupt.edu.cn) directly :)
