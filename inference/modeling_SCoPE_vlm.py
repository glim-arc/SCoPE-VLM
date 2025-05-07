from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import random
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from qwen_vl_utils import process_vision_info
from prompt import scroll_llava_prompt, answer_llava_prompt
import random
import os
import re
from PIL import Image

random.seed(42)
torch.manual_seed(42)

class SCoPEVLMForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config, max_steps=99):
        super().__init__(config)
        self.max_steps = max_steps
        self.INIT_SCROLL = -1
        self.map= None

    def CoS_generate(self, processor=None, images=None, question=None, max_new_tokens=1024,return_pages = False, **kwargs):
        assert processor is not None
        assert images is not None
        assert question is not None

        answer = None
        scroll_num = 0
        notes = ""
        step_note = ""
        current_page = 0
        reading_history = [False] * len(images)

        max_steps = min(self.max_steps, len(images))


        for _ in range(max_steps):
            cur_prompt, image, notes, current_page, reading_history = self.transition_function(question, images, current_page, scroll_num, notes, step_note, reading_history)
            
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

            inputs = processor(
                text=[text],
                images=image,
                padding=True,
                return_tensors="pt",
            )

            inputs.to(self.device)

            generated_ids = super().generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)

            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            step_outputs = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            step_outputs = step_outputs[0]

            step_think = self.extract_string_between(step_outputs, "<think>", "</think>")
            step_note = self.extract_string_between(step_outputs, "<note>", "</note>")
            scroll_num = self.extract_string_between(step_outputs, "<scroll>", "</scroll>")
            answer = self.extract_string_between(step_outputs, "<answer>", "</answer>")

            print(step_outputs)

            if answer != "":
                break

        pages_visited = sum(reading_history)
        
        if return_pages:
            return answer, pages_visited         # <- 두 값 반환
        return answer

    def extract_string_between(self, text: str, start_string: str, end_string: str) -> Optional[str]:
        try:
            start_index = text.index(start_string) + len(start_string)
            end_index = text.index(end_string, start_index)
            return text[start_index:end_index]
        except ValueError:
            return ""

    def transition_function(self, question, images, current_page, scroll_num, notes, step_note, reading_history):

        if current_page != 0:
            notes

        min_page = 0
        max_page = len(images)

        random_flag = False

        try:
            next_page = current_page + int(scroll_num)
        except:
            random_flag = True
            next_page = -1

        if next_page < min_page or next_page >= max_page or random_flag or reading_history[next_page]:
            unvisited_indices = [i for i, visited in enumerate(reading_history) if not visited]
            if unvisited_indices:
                next_page = random.choice(unvisited_indices)
            else:
                next_page = -1

        if next_page >= 0:
            if next_page != 0:
                notes += f" Page {current_page}: " + step_note.replace(f"Page {str(current_page)}:", "")
            
            reading_history[next_page] = True
            image = images[next_page]

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

if __name__ == "__main__":
    device_num = 1
    model_path = "Gyubeum/SCOPE-VLM-3B-SFT"
    image_dir = "./sample_imgs"

    if "Qwen" not in model_path:
        model_path+= "-Qwen2.5-VL"

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Load model (now called `model`)
    model = SCoPEVLMForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_steps=20,
    )

    # Load processor
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        model_path,
        use_fast=True,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    question = "What is the percentage of the total contracted sum?"

    # Sort key for filenames
    def sort_key(path: str):
        fname = os.path.basename(path)
        m = re.search(r"(\d+)", fname)
        return int(m.group(1)) if m else fname.lower()

    # Interactive loop
    while True:
        # 1) Gather & sort image paths
        image_paths = sorted(
            [
                os.path.join(image_dir, f)
                for f in os.listdir(image_dir)
                if f.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")
                )
            ],
            key=sort_key,
        )

        # 2) Load & convert to RGB
        images = [Image.open(p).convert("RGB") for p in image_paths]

        # 3) Run CoS generation
        output_text = model.CoS_generate(
            processor=processor,
            images=images,
            question=question,
        )

        print(f"\nAnswer: {output_text}\n")

        # 4) Get next question (empty → exit)
        question = input("Question (press Enter to exit): ").strip()
        if not question:
            break

        # 5) Optionally change folder
        new_dir = input("Image directory (press Enter to reuse current folder): ").strip()
        if new_dir:
            image_dir = new_dir