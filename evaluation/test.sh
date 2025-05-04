accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=Qwen/Qwen2.5-VL-3B-Instruct,max_pixels=448*448,use_flash_attention_2=True,device_map=auto\
    --tasks multidocvqa_val \
    --batch_size 1 \
    --output_path /home/sem/lmms-eval/outputs/Qwen2_5_VL_multidocvqa_val.json