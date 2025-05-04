import ast
import json
import os
import re
from PIL import Image
from loguru import logger as eval_logger
from datasets import load_dataset
from lmms_eval.api.metrics import levenshtein_distance
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from huggingface_hub import hf_hub_download

def DUDE_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    return f"{pre_prompt}{question}{post_prompt}"


def DUDE_doc_to_visual(doc):
    # 싱글톤 패턴으로 데이터셋을 한 번만 로드 (성능 최적화)
    if not hasattr(DUDE_doc_to_visual, "_images_cache"):
        print("Initializing image cache for DUDE dataset...")
        DUDE_doc_to_visual._images_cache = {}
    
    repo_id = "YeMoKoo/DUDE"  # 새로운 리포지터리 이름
    folder_path = "dude_dev"   # 이미지가 저장된 폴더 경로
    images = []
    
    for img_name in doc["image_paths"]:
        # 이미 캐시에 있는 경우 재사용
        if img_name in DUDE_doc_to_visual._images_cache:
            images.append(DUDE_doc_to_visual._images_cache[img_name])
            continue
        
        try:
            # 폴더 경로와 이미지 이름을 결합하여 전체 경로 생성
            img_path = os.path.join(folder_path, img_name)
            
            # 허깅페이스 허브에서 이미지 다운로드
            local_path = hf_hub_download(
                repo_id=repo_id, 
                filename=img_path,
                revision="main",
                repo_type="dataset"
            )
            
            # 이미지 로드 및 RGB로 변환
            image = Image.open(local_path).convert("RGB")
            
            # 캐시에 저장
            DUDE_doc_to_visual._images_cache[img_name] = image
            images.append(image)
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            # 오류 발생 시 None을 추가하거나 대체 이미지를 사용할 수 있음
            # 여기서는 오류 메시지만 출력
   
    return images



def DUDE_process_results(doc, results):
    pred_answer = results[0]
    answer = doc["answer"]
    if isinstance(answer, list):
        pass  # 이미 리스트면 그대로 사용
    elif isinstance(answer, str):
        answer = [answer]
    else:
        answer = [str(answer)]

    return {"anls": {"answer": answer, "pred_answer": pred_answer}, "accuracy": {"answer": answer, "pred_answer": pred_answer}}


def DUDE_aggregate_results_anls(results):
    keys = {k for result in results for k in result}
    results = {key: [result.get(key, None) for result in results] for key in keys}
    evaluator = Evaluator(case_sensitive=False)
    metric = evaluator.get_metrics(results["answer"], results["pred_answer"])

    return sum(metric["anls"]) / len(metric["anls"])


def DUDE_aggregate_results_accuracy(results):
    keys = {k for result in results for k in result}
    results = {key: [result.get(key, None) for result in results] for key in keys}
    evaluator = Evaluator(case_sensitive=False)
    metric = evaluator.get_metrics(results["answer"], results["pred_answer"])

    return sum(metric["accuracy"]) / len(metric["accuracy"])




##################
# Helper functions
##################


class Evaluator:
    def __init__(self, case_sensitive=False):
        self.case_sensitive = case_sensitive
        self.get_edit_distance = levenshtein_distance
        self.anls_threshold = 0.5

    def get_metrics(self, gt_answers, preds):
        batch_accuracy = []
        batch_anls = []
        for batch_idx in range(len(preds)):
            gt = [self._preprocess_str(gt_elm) for gt_elm in gt_answers[batch_idx]]
            pred = self._preprocess_str(preds[batch_idx])
            batch_accuracy.append(self._calculate_accuracy(gt, pred))
            batch_anls.append(self._calculate_anls(gt, pred))

        return {"accuracy": batch_accuracy, "anls": batch_anls}

    def _preprocess_str(self, string):
        if not self.case_sensitive:
            string = string.lower()

        return string.strip()

    def _calculate_accuracy(self, gt, pred):
        if pred == "none":
            return 0

        for gt_elm in gt:
            if gt_elm == pred:
                return 1

        return 0

    def _calculate_anls(self, gt, pred):
        if len(pred) == 0:
            return 0

        if pred == "none":
            return 0

        answers_similarity = [1 - self.get_edit_distance(gt_elm, pred) / max(len(gt_elm), len(pred)) for gt_elm in gt]
        max_similarity = max(answers_similarity)

        anls = max_similarity if max_similarity >= self.anls_threshold else 0
        return anls


if __name__ == "__main__":
    print("-----------------")
    result = DUDE_aggregate_results_anls([{"questionId": 1, "answer": ["answer"], "pred_answer": "pred_answer"}, {"questionId": 2, "answer": ["nswer"], "pred_answer": "nswer"}])
    print("Aggregate ANLS result:", result)