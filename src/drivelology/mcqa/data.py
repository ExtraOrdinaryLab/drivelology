"""
Data loading and processing functions for MCQA tasks.
"""

import os
import random
from typing import List, Tuple, Set, Dict

from rich.console import Console
from datasets import load_dataset

console = Console()


def get_narrative_fields(language: str) -> Tuple[str, List[str]]:
    """
    Get the appropriate narrative field names based on language.
    
    Args:
        language: Language code ('en', 'zh_tw', 'zh')
        
    Returns:
        A tuple containing (positive field name, list of negative field names)
    """
    if language == 'en':
        pos_field = 'pos_en'
        neg_fields = ['neg_en_1', 'neg_en_2', 'neg_en_3', 'neg_en_4']
    elif language == 'zh_tw':
        pos_field = 'pos_tc'
        neg_fields = ['neg_tc_1', 'neg_tc_2', 'neg_tc_3', 'neg_tc_4']
    elif language == 'zh':
        pos_field = 'pos_sc'
        neg_fields = ['neg_sc_1', 'neg_sc_2', 'neg_sc_3', 'neg_sc_4']
    else:
        # Default to English if language not supported
        pos_field = 'pos_en'
        neg_fields = ['neg_en_1', 'neg_en_2', 'neg_en_3', 'neg_en_4']
    
    return pos_field, neg_fields


def load_existing_results(save_file: str) -> Set[str]:
    """
    Load IDs of already processed items.
    
    Args:
        save_file: Path to the results file
        
    Returns:
        A set of already processed item IDs
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
    Clean text by removing newlines and tabs.
    
    Args:
        text: Input text
        
    Returns:
        Sanitized text
    """
    return str(text).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')


def save_result_easy(save_file: str, data_row: Dict, options: List[str], answer: str, prediction: str):
    """
    Save result to file for the easy version (5 options).
    
    Args:
        save_file: Path to save results
        data_row: Dictionary containing the data item
        options: List of narrative options
        answer: Correct answer (A-E)
        prediction: Predicted answer (A-E)
    """
    with open(save_file, 'a', encoding='utf-8') as f:
        sanitized_text = sanitize_text(data_row['text'])
        sanitized_options = [sanitize_text(opt) for opt in options]
        
        line = (
            f"{data_row['id']}\t{sanitized_text}\t"
            f"{sanitized_options[0]}\t{sanitized_options[1]}\t"
            f"{sanitized_options[2]}\t{sanitized_options[3]}\t"
            f"{sanitized_options[4]}\t{answer}\t{prediction}"
        )
        
        f.write(f"{line}\n")


def save_result_hard(save_file: str, data_row: Dict, options: List[str], answer: str, prediction: str):
    """
    Save result to file for the hard version (4 options + "None of the above").
    
    Args:
        save_file: Path to save results
        data_row: Dictionary containing the data item
        options: List of narrative options (only 4 for hard version)
        answer: Correct answer (A-E)
        prediction: Predicted answer (A-E)
    """
    with open(save_file, 'a', encoding='utf-8') as f:
        sanitized_text = sanitize_text(data_row['text'])
        sanitized_options = [sanitize_text(opt) for opt in options]
        
        line = (
            f"{data_row['id']}\t{sanitized_text}\t"
            f"{sanitized_options[0]}\t{sanitized_options[1]}\t"
            f"{sanitized_options[2]}\t{sanitized_options[3]}\t"
            f"{answer}\t{prediction}"
        )
        
        f.write(f"{line}\n")


def load_dataset_split(dataset_name: str, dataset_config: str):
    """
    Load dataset and get the appropriate split.
    
    Args:
        dataset_name: Name of the dataset
        dataset_config: Configuration name of the dataset
        
    Returns:
        Dataset split or None if loading fails
    """
    try:
        dataset = load_dataset(dataset_name, dataset_config)
        console.log(f"Dataset loaded successfully. Available splits: {list(dataset.keys())}")
        
        # Use 'train' split if available, otherwise use the first available split
        if 'train' in dataset:
            data = dataset['train']
        else:
            split_name = list(dataset.keys())[0]
            data = dataset[split_name]
            console.log(f"Using split: {split_name}")
        
        return data
            
    except Exception as e:
        console.log(f"Error loading dataset: {e}")
        return None


def prepare_options_easy(row: Dict, pos_field: str, neg_fields: List[str]) -> Tuple[List[str], str]:
    """
    Prepare randomized options and determine correct answer for easy version.
    
    Args:
        row: Data item dictionary
        pos_field: Field name for positive narrative
        neg_fields: List of field names for negative narratives
        
    Returns:
        Tuple of (list of options, correct answer letter)
    """
    positive_narrative = row[pos_field]
    negative_narratives = [row[field] for field in neg_fields]
    
    # Randomly arrange options
    options = [positive_narrative] + negative_narratives
    random.shuffle(options)
    correct_answer_idx = options.index(positive_narrative)
    correct_answer = chr(ord('A') + correct_answer_idx)
    
    return options, correct_answer


def prepare_options_hard(row: Dict, neg_fields: List[str]) -> Tuple[List[str], str]:
    """
    Prepare randomized options for hard version.
    In the hard version, all options are incorrect and the correct answer is always E.
    
    Args:
        row: Data item dictionary
        neg_fields: List of field names for negative narratives
        
    Returns:
        Tuple of (list of options, correct answer letter)
    """
    negative_narratives = [row[field] for field in neg_fields]
    
    # Randomly shuffle the negative options
    random.shuffle(negative_narratives)
    options = negative_narratives
    
    # The correct answer is always E (None of the above) in the hard version
    correct_answer = 'E'
    
    return options, correct_answer