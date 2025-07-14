from transformers import EvalPrediction
import numpy as np

def compute_metrics(eval_pred: EvalPrediction, processor):
    predictions, labels = eval_pred
    
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    print(predictions.shape, labels[0].shape)
    print(predictions)
    predictions = np.argmax(predictions, axis=-1)
    
    try:
        # Validate token IDs are within bounds and filter specific invalid tokens
        invalid_tokens = [-100, 151644, 151645] + [151652, 151653, 151655]
        invalid_mask = np.isin(labels, invalid_tokens)
        print(invalid_mask.sum())

        if np.any(invalid_mask):
            invalid_ids = labels[invalid_mask]
            print(f"Warning: Found {len(invalid_ids)} invalid token IDs (min: {np.min(labels)}, max: {np.max(labels)}, vocab_size: {processor.tokenizer.vocab_size})")
            print(f"Specific invalid tokens found: {np.unique(invalid_ids)}")
            # Replace invalid tokens with 0 (or appropriate padding token)
            labels = labels[invalid_mask]

        output_text = processor.batch_decode(
            labels, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)
    except Exception as e:
        print(f"Error during decoding: {str(e)}")
        output_text = [""] * len(labels)

    # accuracy = (predictions == labels).mean()
    
    # print("acc:", accuracy)

    return {
        # "accuracy": accuracy,
        "custom_metric": 0.0
    }
