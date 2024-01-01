
import os
import argparse
import torch
from diffusers.utils.import_utils import is_xformers_available
from datasets import load_dataset
from tqdm.auto import tqdm
from scipy.io.wavfile import write
from auffusion_pipeline import AuffusionPipeline




def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a inference script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="auffusion/auffusion",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )    
    parser.add_argument(
        "--test_data_dir",
        type=str,
        default="./data/test_audiocaps.raw.json",
        help="Path to test dataset in json file",
    )    
    parser.add_argument(
        "--audio_column", type=str, default="audio_path", help="The column of the dataset containing an audio."
    )
    parser.add_argument(
        "--caption_column", type=str, default="text", help="The column of the dataset containing a caption."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/auffusion_hf",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--sample_rate", type=int, default=16000, help="The sample rate of audio."
    )
    parser.add_argument(
        "--duration", type=int, default=10, help="The duration(s) of audio."
    ) 
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible inference.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )    
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=7.5, help="The scale of guidance."
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=100, help="Number of inference steps to perform."
    )
    parser.add_argument(
        "--width", type=int, default=1024, help="Width of the image."
    )
    parser.add_argument(
        "--height", type=int, default=256, help="Height of the image."
    ) 
    args = parser.parse_args()
    
    return args


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)   

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32

    pipeline = AuffusionPipeline.from_pretrained(args.pretrained_model_name_or_path)
    pipeline = pipeline.to(device, weight_dtype)
    pipeline.set_progress_bar_config(disable=True)
    
    if is_xformers_available() and args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    generator = torch.Generator(device=device).manual_seed(args.seed)

    # load dataset
    audio_column, caption_column = args.audio_column, args.caption_column
    data_files = {"test": args.test_data_dir}
    dataset = load_dataset("json", data_files=data_files, split="test")

    # output dir
    audio_output_dir = os.path.join(args.output_dir, "audios")
    os.makedirs(audio_output_dir, exist_ok=True)

    # generating    
    audio_length = args.sample_rate * args.duration
    for i in tqdm(range(len(dataset)), desc="Generating"):

        prompt = dataset[i][caption_column]        
        audio_name = os.path.basename(dataset[i][audio_column])

        audio_path = os.path.join(audio_output_dir, audio_name)

        if os.path.exists(audio_path):
            continue

        with torch.autocast("cuda"):
            output = pipeline(prompt=prompt, num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale, generator=generator, width=args.width, height=args.height)        

        audio = output.audios[0][:audio_length]

        write(audio_path, args.sample_rate, audio)


if __name__ == "__main__":
    main()




