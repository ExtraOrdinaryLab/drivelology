"""
Evaluation functions for the multi-label tagging task.

This module contains functions for evaluating the performance of
multi-label tagging predictions using various metrics.
"""

import numpy as np
from typing import List
from rich.console import Console
from sklearn import metrics
from drivelology.tagging.models import ALL_CATEGORIES

console = Console()


def mean_average_precision(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Calculate Mean Average Precision (mAP) for multi-label classification.
    
    Args:
        y_true: Binary matrix of true labels (samples x labels)
        y_scores: Matrix of prediction scores (samples x labels)
    
    Returns:
        mAP score
    """
    n_classes = y_true.shape[1]
    ap_scores = []
    
    for i in range(n_classes):
        ap = metrics.average_precision_score(y_true[:, i], y_scores[:, i])
        ap_scores.append(ap)
    
    return np.mean(ap_scores)


def calculate_overall_lwlrap_sklearn(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Calculate Label-Weighted Label-Ranking Average Precision (lwlrap) using sklearn.
    
    Args:
        y_true: Binary matrix of true labels (samples x labels)
        y_scores: Matrix of prediction scores (samples x labels)
    
    Returns:
        lwlrap score
    """
    sample_weight = np.sum(y_true > 0, axis=1)
    nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)
    
    if len(nonzero_weight_sample_indices) == 0:
        return 0.0
        
    overall_lwlrap = metrics.label_ranking_average_precision_score(
        y_true[nonzero_weight_sample_indices, :] > 0, 
        y_scores[nonzero_weight_sample_indices, :], 
        sample_weight=sample_weight[nonzero_weight_sample_indices]
    )
    return overall_lwlrap


def calculate_metrics(save_file: str):
    """
    Calculate and display evaluation metrics for results in a file.
    
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
                if len(items) >= 5:
                    answers, predictions = items[-2], items[-1]
                    
                    # Parse ground truth labels
                    answer_list = [answer.strip().lower() for answer in answers.split(',')]
                    answer_binary = [1 if cat in answer_list else 0 for cat in ALL_CATEGORIES]
                    
                    # Parse predicted labels
                    pred_list = [pred.strip().lower() for pred in predictions.split(',')]
                    pred_binary = [1 if cat in pred_list else 0 for cat in ALL_CATEGORIES]
                    
                    y_true.append(answer_binary)
                    y_pred.append(pred_binary)
    
    if not y_true:
        console.log("No valid data found for metrics calculation.")
        return
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    # Note: Commented out metrics can be enabled if needed
    # map_score = mean_average_precision(y_true, y_pred)
    # lwlrap_score = calculate_overall_lwlrap_sklearn(y_true, y_pred)
    
    # Calculate additional metrics
    hamming_loss = metrics.hamming_loss(y_true, y_pred)
    jaccard_score = metrics.jaccard_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_micro = metrics.f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_weighted = metrics.f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Display metrics
    # console.log(f"mAP score: {map_score:.6f}")
    # console.log(f"lwlrap score: {lwlrap_score:.6f}")
    console.log(f"Hamming Loss: {hamming_loss:.6f}")
    console.log(f"Jaccard Score (macro): {jaccard_score:.6f}")
    console.log(f"F1 Score (macro): {f1_macro:.6f}")
    console.log(f"F1 Score (micro): {f1_micro:.6f}")
    console.log(f"F1 Score (weighted): {f1_weighted:.6f}")
    
    # Detailed report for each category
    report = metrics.classification_report(
        y_true, y_pred, 
        target_names=ALL_CATEGORIES, 
        zero_division=0, 
        digits=4
    )
    console.log("Classification Report:")
    console.log(report)