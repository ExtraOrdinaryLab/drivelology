"""
Evaluation functions for the narrative generation task.

This module contains functions for evaluating generated narratives
using metrics like BERTScore and G-Eval.
"""

from typing import List, Dict, Any, Tuple

from rich.console import Console
from bert_score import score as compute_bert_score

from drivelology.narrative.data import EvaluationResult

console = Console()


def evaluate_with_bertscore(candidates: List[str], 
                           references: List[str],
                           language: str = "en") -> Tuple[List[float], List[float], List[float]]:
    """
    Evaluate narratives using BERTScore.
    
    Args:
        candidates: List of candidate narratives
        references: List of reference narratives
        language: Language code for BERTScore
        
    Returns:
        Tuple of (precision, recall, F1) lists
    """
    console.log(f"Computing BERTScore for {len(candidates)} samples (language: {language})...")
    try:
        P, R, F1 = compute_bert_score(candidates, references, lang=language, verbose=True)
        return P.tolist(), R.tolist(), F1.tolist()
    except Exception as e:
        console.log(f"[ERROR] Failed to compute BERTScore: {e}")
        # Return empty lists of appropriate length
        empty = [0.0] * len(candidates)
        return empty, empty, empty


def update_results_with_bertscore(results: List[EvaluationResult], 
                                 language_code: str,
                                 result_filepath: str):
    """
    Update results with BERTScore metrics and save to file.
    
    Args:
        results: List of evaluation results to update
        language_code: Language code
        result_filepath: Path to the results file
    """
    if not results:
        return
    
    candidates = [r.candidate for r in results]
    references = [r.reference for r in results]
    
    console.log("Computing BERTScore for batch...")
    # For BERTScore, use simplified language code
    bert_lang = "en" if language_code.startswith("en") else "zh"
    bert_p, bert_r, bert_f1 = evaluate_with_bertscore(candidates, references, bert_lang)
    
    # Update results and rewrite file
    for i, result in enumerate(results):
        if i < len(bert_p):
            result.bert_precision = bert_p[i]
            result.bert_recall = bert_r[i]
            result.bert_f1 = bert_f1[i]
            
            # Update line in file, ensuring correct format
            try:
                # Read existing file content
                with open(result_filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Flag whether the corresponding line was found and updated
                updated = False
                
                # Find and update the corresponding line
                for j, line in enumerate(lines):
                    if line.startswith(result.id + '\t'):
                        parts = line.strip().split('\t')
                        # Ensure parts list has sufficient length
                        while len(parts) < 8:
                            parts.append('')
                        
                        # Update BERTScore values
                        parts[5] = str(result.bert_precision)
                        parts[6] = str(result.bert_recall)
                        parts[7] = str(result.bert_f1)
                        lines[j] = '\t'.join(parts) + '\n'
                        updated = True
                        break
                
                # If corresponding line not found, append a new line
                if not updated:
                    console.log(f"Warning: Could not find row for ID {result.id} in file, appending instead")
                    from drivelology.narrative.data import append_result
                    append_result(result_filepath, result, False)
                else:
                    # Write back to file
                    with open(result_filepath, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
                
                console.log(f"Updated BERTScore for ID {result.id}: P={bert_p[i]:.3f}, R={bert_r[i]:.3f}, F1={bert_f1[i]:.3f}")
            
            except Exception as e:
                console.log(f"Error updating BERTScore for ID {result.id}: {str(e)}")


def print_statistics(results: List[EvaluationResult]):
    """
    Print summary statistics for evaluation results.
    
    Args:
        results: List of evaluation results
    """
    geval_scores = [r.geval_score for r in results if r.geval_score is not None]
    bert_f1_scores = [r.bert_f1 for r in results if r.bert_f1 is not None]
    bert_recall_scores = [r.bert_recall for r in results if r.bert_recall is not None]
    
    console.log(f"===== Results Statistics =====")
    console.log(f"Total processed samples: {len(results)}")
    
    if geval_scores:
        console.log(f"G-Eval Statistics:")
        console.log(f"  Mean: {sum(geval_scores) / len(geval_scores):.6f}")
        console.log(f"  Median: {sorted(geval_scores)[len(geval_scores) // 2]}")
        console.log(f"  Max: {max(geval_scores)}")
        console.log(f"  Min: {min(geval_scores)}")
        console.log(f"  Count: {len(geval_scores)}")
    else:
        console.log(f"G-Eval: No scores available")
    
    if bert_f1_scores:
        console.log(f"BERTScore F1 Statistics:")
        console.log(f"  Mean: {sum(bert_f1_scores) / len(bert_f1_scores):.6f}")
        console.log(f"  Max: {max(bert_f1_scores):.6f}")
        console.log(f"  Min: {min(bert_f1_scores):.6f}")
        console.log(f"  Count: {len(bert_f1_scores)}")
    else:
        console.log(f"BERTScore: No scores available")
    
    if bert_recall_scores:
        console.log(f"BERTScore Recall Statistics:")
        console.log(f"  Mean: {sum(bert_recall_scores) / len(bert_recall_scores):.6f}")
        console.log(f"  Max: {max(bert_recall_scores):.6f}")
        console.log(f"  Min: {min(bert_recall_scores):.6f}")
        console.log(f"  Count: {len(bert_recall_scores)}")
    else:
        console.log(f"BERTScore: No scores available")
    
    console.log(f"============================")