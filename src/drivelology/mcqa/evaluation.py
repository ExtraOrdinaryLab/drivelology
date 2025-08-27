"""
Metrics and evaluation functions for MCQA tasks.
"""

from sklearn import metrics
from rich.console import Console

console = Console()


def calculate_metrics(save_file: str):
    """
    Calculate and display evaluation metrics.
    
    Args:
        save_file: Path to the results file
    """
    y_true = []
    y_pred = []
    
    with open(save_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items = line.split('\t')
                if len(items) >= 8:
                    answer, prediction = items[-2], items[-1]
                    y_true.append(answer)
                    y_pred.append(prediction)
    
    if not y_true:
        console.log("No valid data found for metrics calculation.")
        return
    
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    console.log(f"Accuracy: {accuracy:.6f}")
    console.log(f"Precision: {precision:.6f}")
    console.log(f"Recall: {recall:.6f}")
    console.log(f"F1: {f1:.6f}")
    
    report = metrics.classification_report(y_true, y_pred, zero_division=0)
    console.log(report)
    
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    console.log("Confusion Matrix:")
    console.log(confusion_matrix)