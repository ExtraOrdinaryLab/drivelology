"""
Utility functions for MCQA tasks.
"""

import os
import argparse
from typing import Tuple, Dict, Any

from rich.console import Console

console = Console()


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Multiple Choice Question Answering (MCQA) for narrative classification")
    parser.add_argument("--prompt_version", type=str, default="v1_en", 
                        help="Prompt template version to use")
    parser.add_argument("--llm_provider", type=str, default="deepseek", 
                        help="LLM provider (e.g., openai, deepseek)")
    parser.add_argument("--llm_model", type=str, default="deepseek-chat", 
                        help="LLM model name (e.g., gpt-4o-mini, deepseek-chat)")
    parser.add_argument("--dataset_name", type=str, default="extraordinarylab/drivel-hub", 
                        help="HuggingFace dataset name")
    parser.add_argument("--dataset_config", type=str, default="v0618", 
                        help="Dataset configuration name")
    parser.add_argument("--think", action="store_true", 
                        help="Enable thinking mode in LLM")
    parser.add_argument("--eval_only", action="store_true", 
                        help="Only evaluate existing results without running predictions")
    return parser.parse_args()


def setup_output_directory(dir_name: str = 'multiple_choices_easy') -> str:
    """
    Create output directory if it doesn't exist.
    
    Args:
        dir_name: Name of the output directory
        
    Returns:
        Path to the output directory
    """
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def get_save_file_path(llm_model: str, prompt_version: str, save_dir: str) -> str:
    """
    Generate path for saving results.
    
    Args:
        llm_model: Name of the LLM model
        prompt_version: Prompt template version
        save_dir: Directory to save results
        
    Returns:
        Full path to the save file
    """
    model_name = llm_model.split("/")[-1]
    save_file = f'{model_name}_{prompt_version}_results.tsv'
    return os.path.join(save_dir, save_file)