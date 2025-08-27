#!/usr/bin/env python3
"""
Multi-label Tagging Script for Drivelology.

This script evaluates a language model's ability to classify Drivelology texts
into multiple rhetorical categories (inversion, wordplay, switchbait, paradox, 
misdirection). The task assesses the model's capacity to identify various forms
of linguistic techniques and rhetorical devices.
"""

import os
import simplemind as sm
from dotenv import load_dotenv
from rich.console import Console
from datasets import load_dataset

# Import modules from the tagging package
from drivelology.tagging.config import PROMPT_CONFIGS
from drivelology.tagging.models import generate_prediction
from drivelology.tagging.data import (
    generate_id_from_text, extract_labels_from_dataset,
    load_existing_results, sanitize_text, save_result
)
from drivelology.tagging.evaluation import calculate_metrics
from drivelology.tagging.utils import parse_arguments, setup_output_directory

load_dotenv()
console = Console()


def main():
    """Main execution function for multi-label tagging experiments."""
    args = parse_arguments()

    # Experiment configuration
    prompt_version = args.prompt_version
    llm_provider = args.llm_provider
    llm_model = args.llm_model
    dataset_name = args.dataset_name
    dataset_config = args.dataset_config
    output_dir = args.output_dir
    think = args.think
    eval_only = args.eval_only
    
    # Get prompt configuration
    if prompt_version not in PROMPT_CONFIGS:
        raise ValueError(f"Unknown prompt version: {prompt_version}. Available: {list(PROMPT_CONFIGS.keys())}")
    
    prompt_config = PROMPT_CONFIGS[prompt_version]
    console.log(f"Using prompt: {prompt_config.name} ({prompt_config.description})")
    
    # Set up results file path
    setup_output_directory(output_dir)
    save_file = f'{llm_model.split("/")[-1]}_{prompt_version}_multilabel.tsv'
    save_file = os.path.join(output_dir, save_file)

    # If eval_only flag is set, just calculate metrics on existing results
    if eval_only and os.path.exists(save_file):
        console.log(f"Evaluation-only mode: Analyzing existing results in {save_file}")
        calculate_metrics(save_file)
        return 
    
    # Load dataset
    console.log(f"Loading dataset: {dataset_name}")
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
            
    except Exception as e:
        console.log(f"Error loading dataset: {e}")
        return
    
    # Load already processed IDs
    exist_ids = load_existing_results(save_file)
    console.log(f"Found {len(exist_ids)} already processed samples")
    
    # Initialize LLM session
    llm = sm.Session(
        llm_provider=llm_provider, 
        llm_model=llm_model, 
    )
    
    # Process each sample
    processed_count = 0
    for idx, row in enumerate(data):
        # Generate or get ID
        if 'id' in row and row['id']:
            data_id = str(row['id'])
            if len(data_id) != 16:
                data_id = generate_id_from_text(row['text'])
        else:
            data_id = generate_id_from_text(row['text'])
        
        # Skip already processed samples
        if data_id in exist_ids:
            continue
        
        text = sanitize_text(row['text'])
        
        # Extract ground truth categories
        try:
            categories = extract_labels_from_dataset(row)
            if not categories:
                console.log(f"Skip {data_id} (no valid categories found).")
                continue
            answer = ', '.join(categories)
        except Exception as e:
            console.log(f"Skip {data_id} (error extracting labels: {e}).")
            continue
        
        # Generate prompt
        prompt = prompt_config.template.format(text=text)
        
        # Generate prediction
        prediction, reason = generate_prediction(llm, prompt, think, data_id, save_file)
        
        if prediction is None:
            console.log(f"Skipping {data_id} due to generation failure.")
            continue
        
        console.log(f"ID: {data_id}")
        console.log(f"Prediction: {prediction}")
        console.log(f"Ground truth: {answer}")
        console.log('=' * 50)
        
        # Save result
        save_result(save_file, {'id': data_id, 'text': text}, reason, answer, prediction)
        
        processed_count += 1
        if processed_count % 10 == 0:
            console.log(f"Processed {processed_count} items...")
    
    # Calculate and display evaluation metrics
    if os.path.exists(save_file):
        console.log("Calculating metrics...")
        calculate_metrics(save_file)
        console.log(f"Results saved to: {save_file}")
    else:
        console.log("No results to analyze")


if __name__ == '__main__':
    main()