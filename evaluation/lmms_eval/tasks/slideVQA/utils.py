import ast
import json
import os
import re

from loguru import logger as eval_logger

from lmms_eval.api.metrics import levenshtein_distance
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file


def slideVQA_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    return f"{pre_prompt}{question}{post_prompt}"


def slideVQA_doc_to_visual(doc):
    return [doc[f"page_{i}"].convert("RGB") for i in range(1, 30) if f"page_{i}" in doc and doc[f"page_{i}"] is not None]


def slideVQA_process_results(doc, results):
    pred_answer = results[0]
    
    # 안전하게 answer 필드 처리
    try:
        # Python 리터럴 형식인 경우 (리스트 등)
        answer = ast.literal_eval(doc["answer"])
    except (SyntaxError, ValueError):
        # 일반 텍스트 또는 숫자인 경우
        if isinstance(doc["answer"], (int, float)):
            answer = doc["answer"]  # 숫자는 그대로 유지
        else:
            answer = [doc["answer"]]  # 텍스트는 리스트로 변환
    
    return {"anls": {"questionId": int(doc["qa_id"]), "answer": answer, "pred_answer": pred_answer}, 
            "accuracy": {"questionId": int(doc["qa_id"]), "answer": answer, "pred_answer": pred_answer}}


def slideVQA_aggregate_results_anls(results):
    keys = {k for result in results for k in result}
    results = {key: [result.get(key, None) for result in results] for key in keys}
    evaluator = Evaluator(case_sensitive=False)
    metric = evaluator.get_metrics(results["answer"], results["pred_answer"])

    return sum(metric["anls"]) / len(metric["anls"])


def slideVQA_aggregate_results_accuracy(results):
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

    def _ensure_list(self, obj):
        if isinstance(obj, (list, tuple, set)):
            return obj
        return [obj]


    def get_metrics(self, gt_answers, preds):
        batch_accuracy, batch_anls = [], []

        for gt_raw, pred_raw in zip(gt_answers, preds):
            # 1. 정답을 리스트형으로 통일
            gt_list = self._ensure_list(gt_raw)
            # 2. 문자열 전처리
            gt   = [self._preprocess_str(g) for g in gt_list]
            pred = self._preprocess_str(pred_raw)

            # 3. 메트릭 계산
            batch_accuracy.append(self._calculate_accuracy(gt, pred))
            batch_anls.append(self._calculate_anls(gt, pred))

        return {"accuracy": batch_accuracy, "anls": batch_anls}


    def _preprocess_str(self, string):
        if string is None:
            string = ""

        string = str(string)
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
    slideVQA_aggregate_results_anls([{"questionId": 1, "answer": ["answer"], "pred_answer": "pred_answer"}, {"questionId": 2, "answer": ["nswer"], "pred_answer": "nswer"}])
