"""
Data handling functions for the narrative generation task.

This module contains functions for loading, processing, and storing
data related to narrative generation experiments.
"""

import os
from dataclasses import dataclass
from typing import Set, List, Dict, Any, Optional

import pandas as pd
from rich.console import Console
from datasets import load_dataset

console = Console()


@dataclass
class EvaluationResult:
    """
    Container for narrative evaluation results.
    
    Attributes:
        id: Unique identifier for the text
        text: Original input text
        reference: Reference (gold standard) narrative
        candidate: Generated candidate narrative
        geval_score: G-Eval (LLM as judge) score
        bert_precision: BERTScore precision
        bert_recall: BERTScore recall
        bert_f1: BERTScore F1
    """
    id: str
    text: str
    reference: str
    candidate: str
    geval_score: Optional[int] = None
    bert_precision: Optional[float] = None
    bert_recall: Optional[float] = None
    bert_f1: Optional[float] = None


def load_dataset_from_hub(dataset_name: str = "extraordinarylab/drivel-hub", 
                         config_name: str = "v0618", 
                         split: str = "test") -> pd.DataFrame:
    """
    Load a dataset from Hugging Face Hub.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace
        config_name: Configuration name or version
        split: Dataset split to load
        
    Returns:
        Pandas DataFrame containing the dataset
    """
    try:
        dataset = load_dataset(dataset_name, config_name, split=split)
        console.log(f"Loaded {len(dataset)} samples from {dataset_name}")
        return pd.DataFrame(dataset)
    except Exception as e:
        console.log(f"[ERROR] Failed to load dataset: {e}")
        return pd.DataFrame()


def get_reference_narrative(row: pd.Series, language_code: str) -> str:
    """
    Extract the reference narrative from a dataset row based on language.
    
    Args:
        row: Row from the dataset
        language_code: Language code (en, zh_tw, zh)
        
    Returns:
        Reference narrative text or empty string if not found
    """
    if language_code == "en":
        return row.get('pos_en', '')
    elif language_code == "zh_tw":
        return row.get('pos_tc', '')
    elif language_code == "zh":
        return row.get('pos_sc', '')
    else:
        console.log(f"[WARNING] Unsupported language code: {language_code}")
        return ''


def get_processed_ids(filepath: str) -> Set[str]:
    """
    Get IDs of already processed samples from a results file.
    
    Args:
        filepath: Path to the results file
        
    Returns:
        Set of processed IDs
    """
    processed_ids = set()
    if not os.path.exists(filepath):
        return processed_ids
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Skip header
            next(f)
            for line in f:
                if line.strip():
                    parts = line.strip().split('\t')
                    if parts and len(parts) > 0:
                        processed_ids.add(parts[0])
    except Exception as e:
        console.log(f"[WARNING] Failed to read processed IDs from {filepath}: {str(e)}")
    
    return processed_ids


def load_processed_results(filepath: str) -> List[EvaluationResult]:
    """
    Load evaluation results from an existing file.
    
    Args:
        filepath: Path to the results file
        
    Returns:
        List of EvaluationResult objects
    """
    results = []
    if not os.path.exists(filepath):
        return results
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Skip header
            next(f)
            for line in f:
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:  # at least id, text, reference, candidate
                        result = EvaluationResult(
                            id=parts[0],
                            text=parts[1],
                            reference=parts[2],
                            candidate=parts[3]
                        )
                        
                        # Set G-Eval score if available
                        if len(parts) >= 5 and parts[4].strip():
                            try:
                                result.geval_score = int(parts[4])
                            except ValueError:
                                pass
                        
                        # Set BERTScore if available
                        if len(parts) >= 8:
                            try:
                                if parts[5].strip():
                                    result.bert_precision = float(parts[5])
                                if parts[6].strip():
                                    result.bert_recall = float(parts[6])
                                if parts[7].strip():
                                    result.bert_f1 = float(parts[7])
                            except ValueError:
                                pass
                        
                        results.append(result)
    except Exception as e:
        console.log(f"[WARNING] Failed to load processed results from {filepath}: {str(e)}")
    
    return results


def sanitize_text(text: str) -> str:
    """
    Clean text by removing newlines, returns, and tabs.
    
    Args:
        text: Input text
        
    Returns:
        Sanitized text
    """
    return str(text).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')


def append_result(filepath: str, result: EvaluationResult, write_header: bool = False):
    """
    Append a single result to the results file.
    
    Args:
        filepath: Path to the results file
        result: EvaluationResult to append
        write_header: Whether to write a header row (for new files)
    """
    mode = 'w' if write_header else 'a'
    with open(filepath, mode, encoding='utf-8') as f:
        if write_header:
            # Write header row
            headers = ["id", "text", "reference", "candidate", "geval_score", "bert_precision", "bert_recall", "bert_f1"]
            f.write('\t'.join(headers) + '\n')
        
        # Write data row
        row = [
            result.id,
            sanitize_text(result.text),
            sanitize_text(result.reference),
            sanitize_text(result.candidate),
            str(result.geval_score) if result.geval_score is not None else '',
            str(result.bert_precision) if result.bert_precision is not None else '',
            str(result.bert_recall) if result.bert_recall is not None else '',
            str(result.bert_f1) if result.bert_f1 is not None else ''
        ]
        f.write('\t'.join(row) + '\n')


def save_results(filepath: str, results: List[EvaluationResult]):
    """
    Save all results to a file (overwrites existing file).
    
    Args:
        filepath: Path to the results file
        results: List of EvaluationResult objects to save
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        # Write header row
        headers = ["id", "text", "reference", "candidate", "geval_score", "bert_precision", "bert_recall", "bert_f1"]
        f.write('\t'.join(headers) + '\n')
        
        # Write data rows
        for result in results:
            row = [
                result.id,
                sanitize_text(result.text),
                sanitize_text(result.reference),
                sanitize_text(result.candidate),
                str(result.geval_score) if result.geval_score is not None else '',
                str(result.bert_precision) if result.bert_precision is not None else '',
                str(result.bert_recall) if result.bert_recall is not None else '',
                str(result.bert_f1) if result.bert_f1 is not None else ''
            ]
            f.write('\t'.join(row) + '\n')
    
    console.log(f"Results saved to {filepath}")


def get_result_filepath(output_dir: str, 
                       llm_model: str,
                       generation_version: str, 
                       evaluation_version: str) -> str:
    """
    Generate the path for saving results.
    
    Args:
        output_dir: Directory to save results
        llm_model: Name of the LLM model
        generation_version: Version of generation prompt
        evaluation_version: Version of evaluation prompt
        
    Returns:
        Full path to save results
    """
    llm_name = str(llm_model).split('/')[-1].replace(':', '-')
    filename = f"gen_{llm_name}_prompt_{generation_version}_eval_{evaluation_version}.tsv"
    return os.path.join(output_dir, filename)


def extract_language_code(version: str) -> str:
    """
    Extract language code from a version string.
    
    Args:
        version: Version string (e.g., "v1_en", "v2_zh_tw")
        
    Returns:
        Language code
        
    Raises:
        ValueError: If version format is invalid
    """
    parts = version.split('_')
    if len(parts) < 2:
        raise ValueError(f"Invalid version format: {version}")
    
    if len(parts) == 2:  # e.g., "v1_en"
        return parts[1]
    else:  # e.g., "v1_zh_tw"
        return '_'.join(parts[1:])