from transformers import EvalPrediction
import numpy as np

def compute_metrics(eval_pred: EvalPrediction, processor):
    predictions, labels = eval_pred
    
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.argmax(predictions, axis=-1)
    
    print(predictions.shape, labels[0].shape)
    print(predictions)

    try:
        # Validate token IDs are within bounds
        if np.any(labels < 0) or np.any(labels >= processor.tokenizer.vocab_size):
            invalid_ids = labels[(labels < 0) | (labels >= processor.tokenizer.vocab_size)]
            print(f"Warning: Found {len(invalid_ids)} invalid token IDs (min: {np.min(labels)}, max: {np.max(labels)}, vocab_size: {processor.tokenizer.vocab_size})")
            labels = np.clip(labels, 0, processor.tokenizer.vocab_size - 1)

        output_text = processor.batch_decode(
            labels, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)
    except Exception as e:
        print(f"Error during decoding: {str(e)}")
        output_text = [""] * len(labels)

    accuracy = (predictions == labels).mean()
    
    print("acc:", accuracy)

    return {
        "accuracy": accuracy,
        "custom_metric": 0.0
    }
