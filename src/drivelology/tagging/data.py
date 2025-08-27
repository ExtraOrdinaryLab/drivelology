"""
Data handling functions for the multi-label tagging task.

This module contains functions for loading, processing, and storing
data related to multi-label tagging experiments.
"""

import os
import hashlib
from typing import List, Set, Dict, Any
from rich.console import Console
from datasets import load_dataset

console = Console()


def generate_id_from_text(text: str) -> str:
    """
    Generate a 16-character ID from text using SHA-256 hash.
    
    Args:
        text: Input text
        
    Returns:
        16-character hash ID
    """
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def extract_labels_from_dataset(row: Dict[str, Any]) -> List[str]:
    """
    Extract tagging labels from a dataset row.
    
    Args:
        row: Dataset row dictionary
        
    Returns:
        List of category labels
        
    Raises:
        ValueError: If taggings column is not present
    """
    if 'taggings' in row and row['taggings']:
        return row['taggings']
    else:
        raise ValueError("Current dataset doesn't have `taggings` column.")


def load_existing_results(save_file: str) -> Set[str]:
    """
    Load IDs of already processed samples from a results file.
    
    Args:
        save_file: Path to the results file
        
    Returns:
        Set of processed IDs
    """
    exist_ids = set()
    if os.path.exists(save_file):
        with open(save_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        id_val = line.split('\t')[0]
                        exist_ids.add(id_val)
                    except:
                        console.log(f"Warning: Invalid line format: {line}")
    return exist_ids


def sanitize_text(text: str) -> str:
    """
    Clean text by removing newlines, returns, and tabs.
    
    Args:
        text: Input text
        
    Returns:
        Sanitized text
    """
    return str(text).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')


def save_result(save_file: str, data_row: Dict[str, Any], reason: str, answer: str, prediction: str):
    """
    Append a single result to the results file.
    
    Args:
        save_file: Path to the results file
        data_row: Dictionary containing sample data
        reason: Explanation from the model
        answer: Ground truth categories (comma-separated)
        prediction: Predicted categories (comma-separated)
    """
    with open(save_file, 'a', encoding='utf-8') as f:
        sanitized_text = sanitize_text(data_row['text'])
        sanitized_reason = sanitize_text(reason)
        
        line = (
            f"{data_row['id']}\t{sanitized_text}\t"
            f"{sanitized_reason}\t{answer}\t{prediction}"
        )
        
        f.write(f"{line}\n")