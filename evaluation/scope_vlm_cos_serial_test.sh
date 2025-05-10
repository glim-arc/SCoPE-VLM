#export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:/home/sem/SCoPE-VLM
accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model scope_vlm_cos_serial \
    --model_args="pretrained=Gyubeum/SCOPE-VLM-3B-SFT-Qwen2.5-VL,max_pixels=12845056,use_flash_attention_2=True" \
    --tasks m3docvqa_val \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix scope \
    --limit 5 \
    --output_path /home/sem/SCoPE-VLM/evaluation/outputs/SCoPe_serial_m3docvqa_val.json

#for lora
    # --model_args="pretrained=Gyubeum/SCOPE-VLM-3B-SFT-Qwen2.5-VL,lora=lorar/path,max_pixels=12845056,use_flash_attention_2=True" \
