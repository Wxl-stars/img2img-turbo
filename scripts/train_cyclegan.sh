su root
rm -rf /root/.cache
ln -s /gpfs/public-shared/fileset-groups/wheeljack/wuxiaolei/.cache ~/.cache
source ~/.zshrc
conda activate opensora
cd /gpfs/public-shared/fileset-groups/wheeljack/wuxiaolei/projs/img2img-turbo

accelerate launch --multi_gpu --num_processes 8 --main_process_port 29501 src/train_cyclegan_turbo.py   --pretrained_model_name_or_path="stabilityai/sd-turbo" \
    --output_dir="output/cyclegan_turbo/day2night" \
    --dataset_folder "s3://sdagent-shard-bj-baiducloud/wheeljack/wuxiaolei/img_translation/data/e171_day_7322278_night_1706606_2024-12-18_allgood_night_1013194.json" \
    --train_img_prep "resize_286_randomcrop_256x256_hflip" --val_img_prep "no_resize" \
    --learning_rate="1e-5" --max_train_steps=2500000 \
    --train_batch_size=4 --gradient_accumulation_steps=1 \
    --report_to "wandb" --tracker_project_name "unpaired_day2night_cycle_bsl" \
    --enable_xformers_memory_efficient_attention --validation_steps 2500 --viz_freq 100\
    --lambda_gan 0.5 --lambda_idt 1 --lambda_cycle 1 

accelerate launch --multi_gpu --num_processes 32 --num_machines 4 --machine_rank ${RANK} --main_process_port ${MASTER_PORT} --main_process_ip ${MASTER_ADDR} src/train_cyclegan_turbo.py   --pretrained_model_name_or_path="stabilityai/sd-turbo" \
    --output_dir="output/cyclegan_turbo/day2night" \
    --dataset_folder "s3://sdagent-shard-bj-baiducloud/wheeljack/wuxiaolei/img_translation/data/e171_day_7322278_night_1706606_2024-12-18_allgood_night_1013194.json" \
    --train_img_prep "resized_crop_512" --val_img_prep "no_resize" \
    --learning_rate="1e-5" --max_train_steps=2500000 \
    --train_batch_size=1 --gradient_accumulation_steps=1 \
    --report_to "wandb" --tracker_project_name "unpaired_day2night_cycle_bsl" \
    --enable_xformers_memory_efficient_attention --validation_steps 2500 --viz_freq 100\
    --lambda_gan 0.5 --lambda_idt 1 --lambda_cycle 1 
