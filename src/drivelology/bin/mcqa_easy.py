#!/usr/bin/env python3
"""
Multiple Choice Question Answering (MCQA) for narrative classification.

This script runs experiments to evaluate how well language models can
identify the underlying narrative in texts using multiple choice questions.
"""

import os
import simplemind as sm
from dotenv import load_dotenv
from rich.console import Console

# Import modules from the mcqa package
from drivelology.mcqa.config import PROMPT_CONFIGS_EASY
from drivelology.mcqa.models import generate_prediction
from drivelology.mcqa.data import (
    get_narrative_fields, load_existing_results, prepare_options_easy, 
    save_result_easy, load_dataset_split
)
from drivelology.mcqa.evaluation import calculate_metrics
from drivelology.mcqa.utils import parse_arguments, setup_output_directory, get_save_file_path

load_dotenv()
console = Console()


def main():
    """Main execution function for MCQA experiments."""
    args = parse_arguments()

    # Experiment configuration
    prompt_version = args.prompt_version
    llm_provider = args.llm_provider
    llm_model = args.llm_model
    dataset_name = args.dataset_name
    dataset_config = args.dataset_config
    think = args.think
    eval_only = args.eval_only
    
    # Get prompt configuration
    if prompt_version not in PROMPT_CONFIGS_EASY:
        raise ValueError(f"Unknown prompt version: {prompt_version}. Available: {list(PROMPT_CONFIGS_EASY.keys())}")
    
    prompt_config = PROMPT_CONFIGS_EASY[prompt_version]
    console.log(f"Using prompt: {prompt_config.name} ({prompt_config.description})")
    
    # Setup save file
    save_dir = setup_output_directory(dir_name="outputs/mcqa_easy")
    save_file = get_save_file_path(llm_model, prompt_version, save_dir)

    # If eval_only flag is set, just calculate metrics on existing results
    if eval_only and os.path.exists(save_file):
        calculate_metrics(save_file)
        return 
    
    # Load dataset
    console.log(f"Loading dataset: {dataset_name}")
    data = load_dataset_split(dataset_name, dataset_config)
    if data is None:
        return
    
    # Load existing results
    exist_ids = load_existing_results(save_file)
    
    # Initialize LLM
    llm = sm.Session(
        llm_provider=llm_provider, 
        llm_model=llm_model, 
    )
    
    # Get language-specific field names
    pos_field, neg_fields = get_narrative_fields(prompt_config.language)
    
    # Process each data item
    processed_count = 0
    for idx, row in enumerate(data):
        data_id = str(row['id'])
        
        if data_id in exist_ids:
            console.log(f"Skip {data_id} (already processed).")
            continue
        
        # Check if required fields exist
        if not all(field in row for field in [pos_field] + neg_fields):
            console.log(f"Skip {data_id} (missing required fields for language {prompt_config.language}).")
            continue
        
        # Prepare options and determine correct answer
        options, correct_answer = prepare_options_easy(row, pos_field, neg_fields)
        
        # Generate prompt
        prompt = prompt_config.template.format(
            text=row['text'],
            narrative_1=options[0],
            narrative_2=options[1],
            narrative_3=options[2],
            narrative_4=options[3],
            narrative_5=options[4],
        )
        
        # Generate prediction
        prediction = generate_prediction(llm, prompt, think, data_id, save_file)
        
        if prediction is None:
            console.log(f"Skipping {data_id} due to generation failure.")
            continue
        
        console.log(f"ID: {data_id}, Prediction: {prediction}, Answer: {correct_answer}")
        
        # Save result
        save_result_easy(save_file, row, options, correct_answer, prediction)
        
        processed_count += 1
        if processed_count % 10 == 0:
            console.log(f"Processed {processed_count} items...")
    
    # Calculate and display metrics
    console.log("Calculating metrics...")
    calculate_metrics(save_file)
    console.log(f"Results saved to: {save_file}")


if __name__ == '__main__':
    main()