from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import random
import torch
from typing import Optional
from prompt import scroll_llava_prompt, answer_llava_prompt
import random
from vllm import LLM, SamplingParams
from typing import Dict, Optional, Sequence, List, Tuple
import os
from PIL import Image

random.seed(42)
torch.manual_seed(42)

def CoS_generate(model, processor=None, images=None, question=None, max_new_tokens=512):
    assert processor is not None
    assert images is not None
    assert question is not None

    answer = None
    scroll_num = 0
    notes = ""
    step_note = ""
    current_page = 0
    reading_history = [False] * len(images)
    reading_history[0] = True
    max_steps = len(images)

    for _ in range(max_steps):
        cur_prompt, image, notes, current_page, reading_history = transition_function(question, images, current_page, scroll_num, notes, step_note, reading_history)

        print(cur_prompt)

        messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": cur_prompt},
                    ],
                }
            ]

        if image != None:
            messages[0]["content"] = [{"type": "image"}] + messages[0]["content"]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        input = {
            "prompt": text,
            "multi_modal_data": {"image": image},
        },

        sampling_params = SamplingParams(
                    temperature=0.9,
                    top_p=0.9,
                    top_k=50,
                    max_tokens=max_new_tokens,
                )
        
        outputs = llm.generate(input, sampling_params)

        step_outputs = outputs[0].outputs[0].text

        print(step_outputs)

        step_think = extract_string_between(step_outputs, "<think>", "</think>")
        step_note = extract_string_between(step_outputs, "<note>", "</note>")
        scroll_num = extract_string_between(step_outputs, "<scroll>", "</scroll>")
        answer = extract_string_between(step_outputs, "<answer>", "</answer>")

        if answer != "":
            break
    
    return answer

def extract_string_between(text: str, start_string: str, end_string: str) -> Optional[str]:
    try:
        start_index = text.index(start_string) + len(start_string)
        end_index = text.index(end_string, start_index)
        return text[start_index:end_index]
    except ValueError:
        return ""

def transition_function(question, images, current_page, scroll_num, notes, step_note, reading_history):
    print("tranition_function")

    if current_page != 0:
        notes

    min_page = 0
    max_page = len(images)

    random_flag = False

    try:
        next_page = current_page + int(scroll_num)
    except:
        next_page = -1
        random_flag = True

    if next_page < min_page or next_page >= max_page or random_flag or reading_history[next_page]:
        unvisited_indices = [i for i, visited in enumerate(reading_history) if not visited]

        if len(unvisited_indices) > 0:
            next_page = random.choice(unvisited_indices)
        else:
            next_page = -1

    if next_page >= 0:
        if next_page != 0:
            notes += f" Page {current_page}: " + step_note.replace(f"Page {str(current_page)}:", "")

        reading_history[next_page] = True
        image = images[next_page]

        print(next_page, reading_history)

        cur_prompt = scroll_llava_prompt.replace("[Question]", question)
        cur_prompt = cur_prompt.replace("[Previous_Note]", notes)
        cur_prompt = cur_prompt.replace("[Current_page_num]", str(next_page))
        cur_prompt = cur_prompt.replace("[Total_page_num]", str(max_page))
    else:
        image = None
        cur_prompt = answer_llava_prompt.replace("[Question]", question)
        cur_prompt = cur_prompt.replace("[Previous_Note]", notes)
        cur_prompt = cur_prompt.replace("[Total_page_num]", str(max_page))
    
    return cur_prompt, image, notes, next_page, reading_history

def sort_key(s):
    parts = s.split('_p')
    try:
        return int(parts[-1].split('.')[0])
    except ValueError:
        return s
        
if __name__ == "__main__":
    device_num = 0
    model_path = "Gyubeum/SCOPE-VLM-3B-SFT"
    image_dir = "./sample_imgs"

    if "Qwen" not in model_path:
        model_path+= "-Qwen2.5-VL"

    llm = LLM(model=model_path, gpu_memory_utilization=0.70, max_model_len=8096, enable_prefix_caching=True, device=f"cuda:{device_num}")

    min_pixels = 256*28*28
    max_pixels = 2560*28*28
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True, min_pixels=min_pixels, max_pixels=max_pixels, truncation=True)

    image_dir = "/mnt/a/research/SCoPE/sample_imgs"

    question = "What is the corporation name?"

    while True:
        image_filenames = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        image_paths = [os.path.join(image_dir, filename) for filename in image_filenames if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'))]
        image_paths.sort(key=sort_key)

        print(image_paths)

        images = []

        for image_path in image_paths:
            img = Image.open(image_path)
            images.append(img)

        output_text = CoS_generate(llm, processor=processor, images=images, question=question)

        print(output_text)

        question = input("Question: ")
        image_dir = input("Image_dir: ")
