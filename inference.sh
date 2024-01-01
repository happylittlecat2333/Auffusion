
MODEL_NAME="/home/xjl/Project/Audio/T2A/Auffusion/huggingface_checkpoint/auffusion"
MODEL_NAME="auffusion/auffusion"
test_data_dir="./data/test_audiocaps.raw.json"  
output_dir="./output/auffusion"
audio_column="spec_path"
caption_column="text"
num_inference_steps=100 
guidance_scale=7.5


training_params="--pretrained_model_name_or_path=$MODEL_NAME \
    --test_data_dir=$test_data_dir \
    --output_dir=$output_dir \
    --audio_column=$audio_column \
    --caption_column=$caption_column \
    --sample_rate=16000 \
    --duration=10 \
    --num_inference_steps=$num_inference_steps \
    --guidance_scale=$guidance_scale \
    --mixed_precision="fp16" \
    --enable_xformers_memory_efficient_attention \
    "

python inference.py $training_params
