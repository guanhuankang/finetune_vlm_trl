from transformers import EvalPrediction
import numpy as np

def compute_metrics(eval_pred: EvalPrediction):
    predictions, labels = eval_pred
    print(predictions, labels)
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = (predictions == labels).mean()
    
    return {
        "accuracy": accuracy,
        "custom_metric": 0.0
    }