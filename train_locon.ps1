# LoRA train script by @Akegarasu modify by @bdsqlsz

# Train data path | 设置训练用模型、图片
$pretrained_model = "/content/drive/MyDrive/sd/stable-diffusion-webui/models/Stable-diffusion/anythingV3V5Anything_anythingV5PrtRE.safetensors" # base model path | 底模路径
$train_data_dir = "/content/drive/MyDrive/public/sucai/Yimilu_chenshan/" # train dataset path | 训练数据集路径
$reg_data_dir = "./train/tarot/reg"	# reg dataset path | 正则数据集化路径
$network_weights = "" # pretrained weights for LoRA network | 若需要从已有的 LoRA 模型上继续训练，请填写 LoRA 模型路径。
$training_comment = "Yimilu" # training_comment | 训练介绍，可以写作者名或者使用触发关键词

# Train related params | 训练相关参数
$resolution = "768,768" # image resolution w,h. 图片分辨率，宽,高。支持非正方形，但必须是 64 倍数。
$batch_size = 3 # batch size 一次性训练图片批处理数量，根据显卡质量对应调高。
$max_train_epoches = 15 # max train epoches | 最大训练 epoch
$save_every_n_epochs = 2 # save every n epochs | 每 N 个 epoch 保存一次
$network_dim = 64 # network dim | 常用 4~128，不是越大越好
$network_alpha = 64 # network alpha | 常用与 network_dim 相同的值或者采用较小的值，如 network_dim的一半 防止下溢。默认值为 1，使用较小的 alpha 需要提升学习率。
$clip_skip = 2 # clip skip | 玄学 一般用 2
$train_unet_only = 0 # train U-Net only | 仅训练 U-Net，开启这个会牺牲效果大幅减少显存使用。6G显存可以开启
$train_text_encoder_only = 0 # train Text Encoder only | 仅训练 文本编码器
$seed = 1026 # reproducable seed | 设置跑测试用的种子，输入一个prompt和这个种子大概率得到训练图。可以用来试触发关键词
$noise_offset = 0.1 # help allow SD to gen better blacks and whites，(0-1) | 帮助SD更好分辨黑白，推荐0.1
$mixed_precision = "fp16" # bf16效果更好，默认fp16
$network_module = "locon.locon_kohya"
$conv_dim = 8
$conv_alpha = 4

# Learning rate | 学习率
$lr = "1"
$unet_lr = "1"
$text_encoder_lr = "0.5"
$lr_scheduler = "cosine_with_restarts" # "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup" | PyTorch自带6种动态学习率函数
# constant，常量不变, constant_with_warmup 线性增加后保持常量不变, linear 线性增加线性减少, polynomial 线性增加后平滑衰减, cosine 余弦波曲线, cosine_with_restarts 余弦波硬重启，瞬间最大值。
# 推荐默认cosine_with_restarts或者polynomial，配合输出多个epoch结果更玄学
$lr_warmup_steps = 0 # warmup steps | 仅在 lr_scheduler 为 constant_with_warmup 时需要填写这个值
$lr_scheduler_num_cycles = 1 # restarts nums | 仅在 lr_scheduler 为 cosine_with_restarts 时需要填写这个值
$optimizer_type = "DAdaptation" # "adaFactor","AdamW","AdamW8bit","Lion","SGDNesterov","SGDNesterov8bit","DAdaptation",  推荐 新优化器Lion。推荐学习率unetlr=lr=6e-5,tenclr=7e-6


# Output settings | 输出设置
$output_name = "Yimilu_chenshan" # output model name | 模型保存名称
$save_model_as = "safetensors" # model save ext | 模型保存格式 ckpt, pt, safetensors

# 其他设置
$min_bucket_reso = 448 # arb min resolution | arb 最小分辨率
$max_bucket_reso = 768 # arb max resolution | arb 最大分辨率
$persistent_workers = 1 # makes workers persistent, further reduces/eliminates the lag in between epochs. however it may increase memory usage | 跑的更快，吃内存。大概能提速2.5倍


# ============= DO NOT MODIFY CONTENTS BELOW | 请勿修改下方内容 =====================
# Activate python venv
source venv/bin/activate

$Env:HF_HOME = "huggingface"
$ext_args = [System.Collections.ArrayList]::new()
$network_args = [System.Collections.ArrayList]::new()
$optimizer_args = [System.Collections.ArrayList]::new()

if ($train_unet_only) {
  [void]$ext_args.Add("--network_train_unet_only")
}

if ($train_text_encoder_only) {
  [void]$ext_args.Add("--network_train_text_encoder_only")
}

if ($persistent_workers) {
  [void]$ext_args.Add("--persistent_data_loader_workers")
}

if ($network_module) {
  #[void]$ext_args.Add("--network_module=" + $network_module)
  [void]$network_args.Add("conv_dim=1")
  [void]$network_args.Add("conv_alpha=1")
  #[void]$ext_args.Add("--network_args=" + $network_args)
}


if($optimizer_type -ieq "adafactor"){
	[void]$ext_args.Add("--optimizer_type=" + $optimizer_type)
	[void]$optimizer_args.Add("scale_parameter=True")
	[void]$optimizer_args.Add("warmup_init=True")
	[void]$ext_args.Add("--optimizer_args=" + $optimizer_args)
}

if($optimizer_type -ieq "DAdaptation"){
	[void]$ext_args.Add("--optimizer_type=" + $optimizer_type)
	[void]$optimizer_args.Add("decouple=True")
	$lr = "1"
	$unet_lr = "1"
	$text_encoder_lr = "0.5"
	[void]$ext_args.Add("--optimizer_args=" + $optimizer_args)
}

if($optimizer_type -ieq "Lion"){
	$optimizer_type=""
	[void]$ext_args.Add("--use_lion_optimizer")
}

if($optimizer_type -ieq "AdamW8bit"){
	$optimizer_type=""
	[void]$ext_args.Add("--use_8bit_adam")
}

if ($network_weights) {
  [void]$ext_args.Add("--network_weights=" + $network_weights)
}

if ($reg_data_dir) {
  [void]$ext_args.Add("--reg_data_dir=" + $reg_data_dir)
}


# run train
accelerate launch --num_cpu_threads_per_process=8 "./sd-scripts/train_network.py" `
  --network_module="locon.locon_kohya" `
  --network_args $network_args `
  --network_dim=$network_dim `
  --network_alpha=$network_alpha `
  --enable_bucket `
  --pretrained_model_name_or_path=$pretrained_model `
  --train_data_dir=$train_data_dir `
  --output_dir="/content/drive/MyDrive/public/out/Yimilu_chenshan" `
  --logging_dir="/content/drive/MyDrive/public/log" `
  --resolution=$resolution `
  --network_module=networks.lora `
  --max_train_epochs=$max_train_epoches `
  --learning_rate=$lr `
  --unet_lr=$unet_lr `
  --text_encoder_lr=$text_encoder_lr `
  --lr_scheduler=$lr_scheduler `
  --lr_scheduler_num_cycles=$lr_scheduler_num_cycles `
  --output_name=$output_name `
  --train_batch_size=$batch_size `
  --save_every_n_epochs=$save_every_n_epochs `
  --mixed_precision=$mixed_precision `
  --save_precision="fp16" `
  --seed=$seed  `
  --cache_latents `
  --clip_skip=$clip_skip `
  --prior_loss_weight=1 `
  --max_token_length=225 `
  --caption_extension=".txt" `
  --save_model_as=$save_model_as `
  --min_bucket_reso=$min_bucket_reso `
  --max_bucket_reso=$max_bucket_reso `
  --training_comment=$training_comment `
  --noise_offset=$noise_offset `
  --network_args $network_args `
  --xformers $ext_args

Write-Output "Train finished"
Read-Host | Out-Null ;