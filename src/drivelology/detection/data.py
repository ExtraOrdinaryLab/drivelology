"""
Data handling functions for Drivelology detection task.
"""

import os
import hashlib
from typing import Set

from rich.console import Console
from datasets import load_dataset

console = Console()


def generate_id_from_text(text: str) -> str:
    """
    Generate a unique ID from text using SHA-256 hash.
    
    Args:
        text: Input text
        
    Returns:
        First 16 characters of SHA-256 hash
    """
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def load_existing_ids(save_file: str) -> Set[str]:
    """
    Load existing processed IDs from save file.
    
    Args:
        save_file: Path to the results file
        
    Returns:
        Set of already processed item IDs
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
                        console.log(f"[WARNING] Invalid line format: {line}")
    return exist_ids


def convert_label(label: int) -> str:
    """
    Convert dataset label format to our format.
    
    Args:
        label: Integer label from dataset (0 or 1)
        
    Returns:
        String label ("Drivelology" or "non-Drivelology")
    """
    if label == 0:
        return "Drivelology"
    elif label == 1:
        return "non-Drivelology"
    else:
        raise ValueError(f"Unknown label: {label}")


def sanitize_text(text: str) -> str:
    """
    Clean text by removing newlines and tabs.
    
    Args:
        text: Input text
        
    Returns:
        Sanitized text
    """
    return str(text).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')


def save_result(save_file: str, text_id: str, text: str, reason: str, answer: str, prediction: str):
    """
    Save classification result to file.
    
    Args:
        save_file: Path to save results
        text_id: Unique identifier for the text
        text: Original text content
        reason: Model's reasoning for the classification
        answer: Ground truth label
        prediction: Model's prediction
    """
    with open(save_file, 'a', encoding='utf-8') as f:
        sanitized_text = sanitize_text(text)
        sanitized_reason = sanitize_text(reason)
        
        line = (
            f"{text_id}\t{sanitized_text}\t"
            f"{sanitized_reason}\t{answer}\t{prediction}"
        )
        
        f.write(f"{line}\n")


def load_dataset_split(dataset_name: str, split: str = "test"):
    """
    Load dataset and return the specified split.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace
        split: Split name to load (default: "test")
        
    Returns:
        Dataset split or None if loading fails
    """
    try:
        dataset = load_dataset(dataset_name, split=split)
        console.log(f"Loaded {len(dataset)} samples from {dataset_name} ({split})")
        return dataset
            
    except Exception as e:
        console.log(f"[ERROR] Error loading dataset: {e}")
        return None