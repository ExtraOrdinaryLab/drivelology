"""
Utility functions for Drivelology detection task.
"""

import os
import argparse
from typing import List

from rich.console import Console
from drivelology.detection.config import PROMPT_TEMPLATES

console = Console()


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Drivelology Classification Experiment')
    parser.add_argument('--prompt_version', type=str, default='v1_en', 
                        choices=list(PROMPT_TEMPLATES.keys()),
                        help='Prompt template version to use')
    parser.add_argument('--llm_provider', type=str, default='deepseek', 
                        help='LLM provider')
    parser.add_argument('--llm_model', type=str, default='deepseek-chat', 
                        help='LLM model to use')
    parser.add_argument("--dataset_name", type=str, default="extraordinarylab/drivel-binary", 
                        help="HuggingFace dataset name")
    parser.add_argument('--output_dir', type=str, default='outputs/detection', 
                        help='Output directory for results')
    parser.add_argument("--think", action="store_true", help="Enable thinking mode in LLM")
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (for testing)')
    parser.add_argument('--eval_only', action="store_true", 
                        help="Only evaluate existing results without running predictions")
    
    return parser.parse_args()


def setup_output_directory(dir_name: str) -> str:
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
    # Extract the model name from the full path
    model_name = str(llm_model).split('/')[-1].replace(':', '-')
    save_file = f'{model_name}_{prompt_version}.tsv'
    return os.path.join(save_dir, save_file)