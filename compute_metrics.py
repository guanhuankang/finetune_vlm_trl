from transformers import EvalPrediction
import numpy as np

def compute_metrics(eval_pred: EvalPrediction):
    predictions, labels = eval_pred
    
    print(predictions, "$$",  labels)
    print(type(predictions), type(labels))
    print(type(predictions[0]), type(labels[0]), len(predictions), len(labels))
    print(predictions.shape, len(labels), len(labels[0]))
    
    predictions = np.argmax(predictions, axis=1)
    print(predictions, labels)

    accuracy = (predictions == labels).mean()
    
    print("acc:", accuracy)

    return {
        "accuracy": accuracy,
        "custom_metric": 0.0
    }