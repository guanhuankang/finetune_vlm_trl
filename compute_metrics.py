from transformers import EvalPrediction
import numpy as np

from transformers import Qwen2VLProcessor


def compute_metrics(eval_pred: EvalPrediction, processor: Qwen2VLProcessor):
    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.argmax(predictions, axis=-1)

    print()
    labels[labels == -100] = 0
    label_text = processor.batch_decode(
        labels, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("label_text:", label_text)

    pred_text = processor.batch_decode(
        predictions, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("pred_text:", pred_text)

    print()

    return {
        # "accuracy": accuracy,
        "custom_metric": 0.0
    }
