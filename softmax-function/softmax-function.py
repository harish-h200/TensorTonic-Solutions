import numpy as np

def softmax(x, axis=-1):
    """
    Compute the softmax of vector/matrix x in a numerically stable way.
    Works for 1D arrays and 2D mini-batches.
    """
    # Subtracting the max prevents explosive values (NaN / overflow errors)
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    """
    Compute accuracy, precision, recall, F1 for single-label classification.
    Averages: 'micro' | 'macro' | 'weighted' | 'binary' (uses pos_label).
    Return dict with float values.
    """
    # Convert inputs to standard Python lists for native iteration
    y_true = list(y_true)
    y_pred = list(y_pred)
    n_samples = len(y_true)
    
    if n_samples == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # Calculate global Accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = float(correct / n_samples)
    
    # Binary Classification Mode
    if average == "binary":
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == pos_label and p == pos_label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != pos_label and p == pos_label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == pos_label and p != pos_label)
        
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
        
    # Multi-Class Classification Mode
    # Extract unique classes present across both sets to simulate confusion matrix boundaries
    classes = sorted(list(set(y_true + y_pred)))
    
    tp_dict = {c: 0 for c in classes}
    fp_dict = {c: 0 for c in classes}
    fn_dict = {c: 0 for c in classes}
    support_dict = {c: 0 for c in classes}
    
    # Map out true positives, false positives, and false negatives per class
    for t, p in zip(y_true, y_pred):
        support_dict[t] += 1
        if t == p:
            tp_dict[t] += 1
        else:
            fp_dict[p] += 1
            fn_dict[t] += 1
            
    # Compute metrics based on selected averaging method
    if average == "micro":
        total_tp = sum(tp_dict.values())
        total_fp = sum(fp_dict.values())
        total_fn = sum(fn_dict.values())
        
        precision = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
        recall = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
    elif average in ("macro", "weighted"):
        precisions = {}
        recalls = {}
        f1s = {}
        
        # Calculate base metrics for each unique class
        for c in classes:
            tp, fp, fn = tp_dict[c], fp_dict[c], fn_dict[c]
            precisions[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recalls[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1s[c] = 2 * precisions[c] * recalls[c] / (precisions[c] + recalls[c]) if (precisions[c] + recalls[c]) > 0 else 0.0
            
        if average == "macro":
            precision = float(sum(precisions.values()) / len(classes))
            recall = float(sum(recalls.values()) / len(classes))
            f1 = float(sum(f1s.values()) / len(classes))
            
        else:  # weighted
            total_support = sum(support_dict.values())
            if total_support > 0:
                precision = float(sum(precisions[c] * support_dict[c] for c in classes) / total_support)
                recall = float(sum(recalls[c] * support_dict[c] for c in classes) / total_support)
                f1 = float(sum(f1s[c] * support_dict[c] for c in classes) / total_support)
            else:
                precision, recall, f1 = 0.0, 0.0, 0.0
                
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# --- Example Execution Workflow ---
if __name__ == "__main__":
    # 1. Setup Ground Truth Labels (4 samples, 3 possible classes: 0, 1, 2)
    y_true = np.array([0, 1, 2, 2])
    
    # 2. Simulate raw Model Outputs (Logits) from a neural network layer
    # Shape: (n_samples, n_classes) -> (4, 3)
    mock_logits = np.array([
        [2.0, 0.5, 0.1],  # Model strongly guesses class 0 (Correct)
        [0.2, 1.8, 0.3],  # Model strongly guesses class 1 (Correct)
        [1.5, 0.2, 0.4],  # Model wrongly guesses class 0 (Incorrect, should be 2)
        [0.1, 0.5, 2.2]   # Model strongly guesses class 2 (Correct)
    ])
    
    # 3. Apply Softmax to get class probabilities
    probabilities = softmax(mock_logits, axis=-1)
    print("Class Probabilities:\n", np.round(probabilities, 4))
    
    # 4. Get predicted class indices (argmax picks highest probability)
    y_pred = np.argmax(probabilities, axis=-1)
    print("\nPredicted Labels:", y_pred)
    print("True Labels:     ", y_true)
    
    # 5. Run classification metrics
    metrics = classification_metrics(y_true, y_pred, average="micro")
    print("\nComputed Metrics (Micro):")
    for metric_name, score in metrics.items():
        print(f"  {metric_name.capitalize()}: {score:.4f}")
