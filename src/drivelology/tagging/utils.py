"""
Utility functions for the multi-label tagging task.

This module contains helper functions for argument parsing and
other utility operations.
"""

import os
import argparse
from rich.console import Console

console = Console()


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Multi-label Tagging for Drivelology")
    parser.add_argument("--prompt_version", type=str, default="v1_en",
                       help="Version of the prompt template")
    parser.add_argument("--llm_provider", type=str, default="openai",
                       help="LLM provider (openai, deepseek, etc.)")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini",
                       help="LLM model for generation")
    parser.add_argument("--dataset_name", type=str, default="extraordinarylab/drivel-hub",
                       help="Name of the dataset on HuggingFace Hub")
    parser.add_argument("--dataset_config", type=str, default="v0618",
                       help="Configuration name of the dataset")
    parser.add_argument("--output_dir", type=str, default="multilabel",
                       help="Directory to save results")
    parser.add_argument("--think", action="store_true",
                       help="Enable thinking mode in LLM")
    parser.add_argument("--eval_only", action="store_true",
                       help="Only evaluate existing results without running generation")
    
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