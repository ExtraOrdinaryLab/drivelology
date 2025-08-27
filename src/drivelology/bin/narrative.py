#!/usr/bin/env python3
"""
Narrative Generation and Evaluation Script.

This script evaluates a language model's ability to generate implicit narratives
from Drivelology texts. The task assesses the model's capacity to understand
deeper meaning and demonstrate social reasoning skills beyond surface-level comprehension.
"""

import os
from typing import List

import simplemind as sm
from dotenv import load_dotenv
from rich.console import Console


# Import modules from the narrative package
from drivelology.narrative.config import PromptManager, PromptType
from drivelology.narrative.models import generate_narrative, evaluate_with_geval
from drivelology.narrative.data import (
    EvaluationResult, load_dataset_from_hub, get_reference_narrative, 
    get_processed_ids, load_processed_results, append_result, 
    get_result_filepath, extract_language_code
)
from drivelology.narrative.evaluation import update_results_with_bertscore, print_statistics
from drivelology.narrative.utils import parse_arguments, setup_output_directory

load_dotenv()
console = Console()


def run_evaluation_pipeline(
    generation_version: str,
    evaluation_version: str,
    llm_provider: str,
    llm_model: str,
    eval_llm_provider: str,
    eval_llm_model: str,
    dataset_name: str,
    dataset_config: str,
    output_dir: str,
    max_samples: int = None,
    think: bool = False,
    bertscore_batch_size: int = 10
) -> List[EvaluationResult]:
    """
    Run the complete narrative generation and evaluation pipeline.
    
    Args:
        generation_version: Version of generation prompt
        evaluation_version: Version of evaluation prompt
        llm_provider: Provider for generation LLM
        llm_model: Model for generation
        eval_llm_provider: Provider for evaluation LLM
        eval_llm_model: Model for evaluation
        dataset_name: Name of dataset on HuggingFace
        dataset_config: Configuration of dataset
        output_dir: Directory to save results
        max_samples: Maximum number of samples to process
        think: Whether to enable thinking mode
        bertscore_batch_size: Batch size for BERTScore evaluation
        
    Returns:
        List of evaluation results
    """
    # Initialize the prompt manager
    prompt_manager = PromptManager()
    
    # Extract language codes from versions
    language_code = extract_language_code(generation_version)
    eval_language_code = extract_language_code(evaluation_version)
    
    # Ensure generation and evaluation use same language
    if language_code != eval_language_code:
        console.log(f"[WARNING] Generation language ({language_code}) differs from evaluation language ({eval_language_code})")
    
    # Initialize LLM sessions
    generation_llm = sm.Session(
        llm_provider=llm_provider,
        llm_model=llm_model,
    )
    
    evaluation_llm = sm.Session(
        llm_provider=eval_llm_provider,
        llm_model=eval_llm_model,
    )
    
    # Load dataset
    console.log("Loading dataset from Hugging Face Hub...")
    df = load_dataset_from_hub(dataset_name, dataset_config, "test")
    
    if max_samples:
        df = df.head(max_samples)
    
    console.log(f"Loaded {len(df)} samples")
    
    # Create output directory
    setup_output_directory(output_dir)
    
    # Set up result file path and check processed IDs
    result_filepath = get_result_filepath(
        output_dir, llm_model, generation_version, evaluation_version
    )
    processed_ids = get_processed_ids(result_filepath)
    
    console.log(f"Found {len(processed_ids)} already processed samples")
    
    # Whether header needs to be written
    write_header = not os.path.exists(result_filepath)
    
    # Get prompt templates
    generation_prompt = prompt_manager.get_prompt(
        PromptType.NARRATIVE_GENERATION, generation_version
    )
    evaluation_prompt = prompt_manager.get_prompt(
        PromptType.NARRATIVE_EVALUATION, evaluation_version
    )
    
    results = []
    pending_bert_evaluation = []  # Results awaiting BERTScore evaluation
    
    # Generate narratives and collect results
    for index, row in df.iterrows():
        text_id = str(row['id'])
        
        # Skip already processed IDs
        if text_id in processed_ids:
            console.log(f"Skipping already processed ID: {text_id}")
            continue
            
        text = str(row['text']).replace('\n', ' ')
        reference = get_reference_narrative(row, language_code)
        
        if not reference:
            console.log(f"No reference narrative found for ID {text_id}, skipping...")
            continue
        
        console.log(f"Processing ID: {text_id}")
        
        # Generate narrative
        formatted_prompt = generation_prompt.template.format(text=text)
        candidate = generate_narrative(generation_llm, formatted_prompt, think=think)
        
        if not candidate:
            console.log(f"Failed to generate narrative for ID {text_id}, skipping...")
            continue
        
        # G-Eval evaluation
        eval_formatted_prompt = evaluation_prompt.template.format(
            candidate=candidate, reference=reference
        )
        geval_score = evaluate_with_geval(evaluation_llm, candidate, reference, eval_formatted_prompt)
        
        result = EvaluationResult(
            id=text_id,
            text=text,
            reference=reference,
            candidate=candidate,
            geval_score=geval_score
        )
        
        # Add result to collections
        results.append(result)
        pending_bert_evaluation.append(result)
        
        # Immediately write result with G-Eval score
        append_result(result_filepath, result, write_header)
        write_header = False
        
        console.log(f"Generated narrative: {candidate[:100]}...")
        console.log(f"G-Eval score: {geval_score}")
        console.log(f"Result saved to {result_filepath}")
        console.log("=" * 100)
        
        # Batch BERTScore evaluation
        if len(pending_bert_evaluation) >= bertscore_batch_size:
            update_results_with_bertscore(pending_bert_evaluation, language_code, result_filepath)
            pending_bert_evaluation = []
    
    # Process remaining BERTScore evaluations
    if pending_bert_evaluation:
        update_results_with_bertscore(pending_bert_evaluation, language_code, result_filepath)
    
    # If no new results but file exists, load all results from file
    if not results and os.path.exists(result_filepath):
        console.log("All samples have been processed. Loading results from file...")
        results = load_processed_results(result_filepath)
        console.log(f"Loaded {len(results)} processed results from file")
    
    # Print statistics
    if results:
        print_statistics(results)
    else:
        console.log("No results to analyze")
    
    return results


def main():
    """Main execution function for narrative generation experiments."""
    args = parse_arguments()

    # If eval_only flag is set, just load and analyze existing results
    if args.eval_only:
        console.log("Evaluation-only mode: Loading existing results...")
        result_filepath = get_result_filepath(
            args.output_dir, args.llm_model, args.generation_version, args.evaluation_version
        )
        
        if os.path.exists(result_filepath):
            results = load_processed_results(result_filepath)
            print_statistics(results)
        else:
            console.log(f"[ERROR] Results file not found: {result_filepath}")
        return

    # Run full evaluation pipeline
    run_evaluation_pipeline(
        generation_version=args.generation_version,
        evaluation_version=args.evaluation_version,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        eval_llm_provider=args.eval_llm_provider,
        eval_llm_model=args.eval_llm_model,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        think=args.think,
        bertscore_batch_size=args.bertscore_batch_size
    )


if __name__ == '__main__':
    main()