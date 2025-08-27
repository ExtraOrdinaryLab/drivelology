#!/usr/bin/env python3
"""
Drivelology Detection Task.

This script runs experiments to evaluate how well language models can
identify Drivelology texts from non-Drivelology texts.
"""

import os
import simplemind as sm
from dotenv import load_dotenv
from rich.console import Console

# Import modules from the detection package
from drivelology.detection.config import PROMPT_TEMPLATES
from drivelology.detection.models import classify_text
from drivelology.detection.data import (
    generate_id_from_text, load_existing_ids, convert_label,
    save_result, load_dataset_split
)
from drivelology.detection.evaluation import evaluate_results, display_metrics
from drivelology.detection.utils import (
    parse_arguments, setup_output_directory, get_save_file_path
)

load_dotenv()
console = Console()


def main():
    """Main execution function for Drivelology detection experiments."""
    args = parse_arguments()

    # Experiment configuration
    prompt_version = args.prompt_version
    llm_provider = args.llm_provider
    llm_model = args.llm_model
    dataset_name = args.dataset_name
    output_dir = args.output_dir
    think = args.think
    max_samples = args.max_samples
    eval_only = args.eval_only
    
    # Get prompt configuration
    if prompt_version not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt version: {prompt_version}. Available: {list(PROMPT_TEMPLATES.keys())}")
    
    prompt_config = PROMPT_TEMPLATES[prompt_version]
    console.log(f"Using prompt: {prompt_config.name} ({prompt_config.description})")
    
    # Setup save file
    save_dir = setup_output_directory(dir_name=output_dir)
    save_file = get_save_file_path(llm_model, prompt_version, save_dir)

    console.log(f"Results will be saved to: {save_file}")

    # If eval_only flag is set, just calculate metrics on existing results
    if eval_only and os.path.exists(save_file):
        results = evaluate_results(save_file)
        display_metrics(results)
        return
    
    # Load dataset
    console.log("Loading dataset...")
    dataset = load_dataset_split(dataset_name, "test")
    if dataset is None:
        return
    
    # Limit samples for testing if specified
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        console.log(f"Limited to {len(dataset)} samples for testing")
    
    # Load existing processed IDs
    exist_ids = load_existing_ids(save_file)
    console.log(f"Found {len(exist_ids)} already processed samples")
    
    # Initialize LLM
    llm = sm.Session(
        llm_provider=llm_provider, 
        llm_model=llm_model, 
    )
    
    # Process dataset
    processed_count = 0
    skipped_count = 0
    
    for i, sample in enumerate(dataset):
        text = sample['text']
        label = sample['label']
        
        # Generate ID
        text_id = generate_id_from_text(text)
        
        # Skip if already processed
        if text_id in exist_ids:
            skipped_count += 1
            continue
        
        # Convert label format
        answer = convert_label(label)
        
        # Prepare prompt
        prompt = prompt_config.template.format(text=text)
        
        console.log(f"Processing sample {i+1}/{len(dataset)} (ID: {text_id})")
        console.log(f"Text: {text[:100]}...")
        
        # Classify text
        prediction, reason = classify_text(llm, prompt, think, text_id)
        
        if prediction is None:
            console.log(f"[ERROR] Failed to classify text ID {text_id}, skipping...")
            continue
        
        console.log(f"Prediction: {prediction}, Answer: {answer}")
        
        # Save result
        save_result(save_file, text_id, text, reason, answer, prediction)
        processed_count += 1
        
        console.log('=' * 100)

    console.log(f"Processing complete!")
    console.log(f"Processed: {processed_count} samples")
    console.log(f"Skipped: {skipped_count} samples")

    # Evaluate results
    console.log("Evaluating results...")
    results = evaluate_results(save_file)
    display_metrics(results)
    console.log(f"Results saved to: {save_file}")


if __name__ == '__main__':
    main()