import numpy as np
import evaluate

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predicted_labels = np.argmax(predictions, axis=1)
    return accuracy_metric.compute(predictions=predicted_labels, references=labels)