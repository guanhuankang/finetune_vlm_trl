from transformers import EvalPrediction
import numpy as np

def compute_metrics(eval_pred: EvalPrediction, processor):
    predictions, labels = eval_pred
    
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.argmax(predictions, axis=-1)
    
    print(predictions.shape, labels[0].shape)
    print(predictions)

    output_text = processor.batch_decode(
        labels, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print(output_text)

    accuracy = (predictions == labels).mean()
    
    print("acc:", accuracy)

    return {
        "accuracy": accuracy,
        "custom_metric": 0.0
    }