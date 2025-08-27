"""
Evaluation functions for Drivelology detection task.
"""

from typing import Dict, Any
from sklearn import metrics
from rich.console import Console

console = Console()


def evaluate_results(save_file: str) -> Dict[str, Any]:
    """
    Evaluate classification results and return metrics.
    
    Args:
        save_file: Path to the results file
        
    Returns:
        Dictionary containing evaluation metrics
    """
    y_true = []
    y_pred = []
    
    with open(save_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items = line.split('\t')
                if len(items) >= 5:
                    answer, prediction = items[-2], items[-1]
                    y_true.append(answer)
                    y_pred.append(prediction)

    if not y_true:
        return {"error": "No valid results found"}

    accuracy_score = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
    report = metrics.classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)
    
    return {
        "accuracy": accuracy_score,
        "classification_report": report,
        "total_samples": len(y_true)
    }


def display_metrics(results: Dict[str, Any]):
    """
    Display evaluation metrics in a readable format.
    
    Args:
        results: Dictionary containing evaluation metrics
    """
    if "error" in results:
        console.log(f"[ERROR] {results['error']}")
        return
        
    console.log(f"Total samples evaluated: {results['total_samples']}")
    console.log(f"Accuracy Score: {results['accuracy']:.4f}")
    
    # Print classification report
    report = results['classification_report']
    console.log("\nClassification Report:")
    console.log(f"Drivelology - Precision: {report['Drivelology']['precision']:.4f}, "
               f"Recall: {report['Drivelology']['recall']:.4f}, "
               f"F1-score: {report['Drivelology']['f1-score']:.4f}")
    console.log(f"non-Drivelology - Precision: {report['non-Drivelology']['precision']:.4f}, "
               f"Recall: {report['non-Drivelology']['recall']:.4f}, "
               f"F1-score: {report['non-Drivelology']['f1-score']:.4f}")