from transformers import EvalPrediction, Qwen2VLProcessor
import numpy as np
from dirtyjson import json
from utils import clear_memory


def compute_bbox_category_metrics(predictions: list, labels: list):
    pass


def compute_metrics(eval_pred: EvalPrediction, processor: Qwen2VLProcessor):
    clear_memory()

    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.argmax(predictions, axis=-1)

    ## token-level
    pass

    ## token-level between <im_start>assistant (151644 77091) and <im_end> 151645

    ## text-level (human readable)
    labels[labels == -100] = 0
    label_text = processor.batch_decode(labels, skip_special_tokens=False)

    pred_text = processor.batch_decode(predictions, skip_special_tokens=False)

    labels = json.loads(
        '{"results": ['
        + label_text.split("results")[-1].split("[")[1].split("]")[0]
        + "]}"
    )["results"]
    print(labels)

    predictions = json.loads(
        '{"results": ['
        + label_text.split("results")[-1].split("[")[1].split("]")[0]
        + "]}"
    )["results"]
    print("pred_text:", pred_text)

    print()

    return {
        # "accuracy": accuracy,
        "custom_metric": 0.0
    }
