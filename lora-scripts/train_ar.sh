#!/bin/bash
# LoRA train script by @Akegarasu

# Train data path | 设置训练用模型、图片
model_train_type="sdxl-lora"
network_module="networks.lora" # 在这里将会设置训练的网络种类，默认为 networks.lora 也就是 LoRA 训练。如果你想训练 LyCORIS（LoCon、LoHa） 等，则修改这个值为 lycoris.kohya
pretrained_model="/data02/models/checkpoints/sd_xl_base_1.0.safetensors"
vae="/data02/models/vae/SDXL_vae.safetensors"
v2=false
data_path="$1"
train_data_dir="/data02/hyf/train-backend/train_data/$data_path"
prior_loss_weight=1
resolution="1024,1024"
reg_data_dir=""   
noise_offset="0" 
enable_bucket=true
min_bucket_reso=256
max_bucket_reso=5000
bucket_reso_steps=64
save_model_as="safetensors"
save_precision="bf16"
save_every_n_epochs=20
max_train_epochs=20
batch_size=1
gradient_checkpointing=false
network_train_unet_only=false
network_train_text_encoder_only=false
lr=1                           # learning rate | 学习率，在分别设置下方 U-Net 和 文本编码器 的学习率时，该参数失效
unet_lr=1   
text_encoder_lr=1
lr_scheduler="cosine_with_restarts"
lr_warmup_steps=0
lr_restart_cycles=1   
optimizer_type="Prodigy"
network_dim=32
network_alpha=16
log_with="tensorboard"
logging_dir="./logs"
caption_extension=".txt"
shuffle_caption=true
no_metadata=true
keep_tokens=1
max_token_length=255
multires_noise_iterations=6
multires_noise_discount=0.3
seed=1234
clip_skip=2
mixed_precision="bf16"
xformers=true
lowram=false
cache_latents=false
cache_latents_to_disk=true
persistent_data_loader_workers=true
output_name="$data_path"
# output_dir="./output/A-R/$data_path"
output_dir="/data02/models/photoG/$data_path"

# ============= DO NOT MODIFY CONTENTS BELOW | 请勿修改下方内容 =====================
export HF_HOME="huggingface"
export TF_CPP_MIN_LOG_LEVEL=3

extArgs=()
launchArgs=()

if [[ $multi_gpu == 1 ]]; then
  launchArgs+=("--multi_gpu")
  launchArgs+=("--num_processes=2")
fi

if [[ $is_v2_model == 1 ]]; then
  extArgs+=("--v2")
else
  extArgs+=("--clip_skip $clip_skip")
fi

if [[ $parameterization == 1 ]]; then extArgs+=("--v_parameterization"); fi

if [[ $train_unet_only == 1 ]]; then extArgs+=("--network_train_unet_only"); fi

if [[ $train_text_encoder_only == 1 ]]; then extArgs+=("--network_train_text_encoder_only"); fi

if [[ $network_weights ]]; then extArgs+=("--network_weights $network_weights"); fi

if [[ $reg_data_dir ]]; then extArgs+=("--reg_data_dir $reg_data_dir"); fi

if [[ $optimizer_type ]]; then extArgs+=("--optimizer_type $optimizer_type"); fi

if [[ $optimizer_type == "DAdaptation" ]]; then extArgs+=("--optimizer_args decouple=True"); fi

if [[ $save_state == 1 ]]; then extArgs+=("--save_state"); fi

if [[ $resume ]]; then extArgs+=("--resume $resume"); fi

if [[ $persistent_data_loader_workers == 1 ]]; then extArgs+=("--persistent_data_loader_workers"); fi

if [[ $network_module == "lycoris.kohya" ]]; then
  extArgs+=("--network_args conv_dim=$conv_dim conv_alpha=$conv_alpha algo=$algo dropout=$dropout")
fi

if [[ $stop_text_encoder_training -ne 0 ]]; then extArgs+=("--stop_text_encoder_training $stop_text_encoder_training"); fi

if [[ $noise_offset != "0" ]]; then extArgs+=("--noise_offset $noise_offset"); fi

if [[ $min_snr_gamma -ne 0 ]]; then extArgs+=("--min_snr_gamma $min_snr_gamma"); fi

if [[ $use_wandb == 1 ]]; then
  extArgs+=("--log_with=all")
else
  extArgs+=("--log_with=tensorboard")
fi

if [[ $wandb_api_key ]]; then extArgs+=("--wandb_api_key $wandb_api_key"); fi

if [[ $log_tracker_name ]]; then extArgs+=("--log_tracker_name $log_tracker_name"); fi

if [[ $lowram ]]; then extArgs+=("--lowram"); fi

echo $train_data_dir

python -m accelerate.commands.launch ${launchArgs[@]} --num_cpu_threads_per_process=8 "./lora-scripts/scripts/sdxl_train_network.py" \
  --enable_bucket \
  --pretrained_model_name_or_path=$pretrained_model \
  --train_data_dir=$train_data_dir \
  --output_dir=$output_dir \
  --logging_dir="./logs" \
  --log_prefix=$output_name \
  --resolution=$resolution \
  --network_module=$network_module \
  --max_train_epochs=$max_train_epochs \
  --learning_rate=$lr \
  --unet_lr=$unet_lr \
  --text_encoder_lr=$text_encoder_lr \
  --lr_scheduler=$lr_scheduler \
  --lr_warmup_steps=$lr_warmup_steps \
  --lr_scheduler_num_cycles=$lr_restart_cycles \
  --network_dim=$network_dim \
  --network_alpha=$network_alpha \
  --output_name=$output_name \
  --train_batch_size=$batch_size \
  --save_every_n_epochs=$save_every_n_epochs \
  --mixed_precision="bf16" \
  --save_precision="bf16" \
  --seed=$seed \
  --cache_latents_to_disk \
  --prior_loss_weight=1 \
  --max_token_length=225 \
  --caption_extension=".txt" \
  --save_model_as=$save_model_as \
  --min_bucket_reso=$min_bucket_reso \
  --max_bucket_reso=$max_bucket_reso \
  --keep_tokens=$keep_tokens \
  --xformers --shuffle_caption ${extArgs[@]} \
  --multires_noise_iterations=6 \
  --multires_noise_discount=0.3 \
  --optimizer_args "decouple=True" "weight_decay=0.01" "use_bias_correction=True" "d_coef=2.0"
