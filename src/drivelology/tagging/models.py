"""
Model-related classes and functions for the multi-label tagging task.

This module contains the Pydantic models for structured responses
and functions for interacting with language models.
"""

from pydantic import BaseModel
from rich.console import Console
from instructor.exceptions import InstructorRetryException, IncompleteOutputException
from typing import List, Tuple

console = Console()

# All supported categories for tagging
ALL_CATEGORIES = ['inversion', 'misdirection', 'paradox', 'switchbait', 'wordplay']


class TaggingResponseModel(BaseModel):
    """
    Pydantic model for tagging response.
    
    Attributes:
        answer: Comma-separated list of categories
        reason: Explanation for the assigned categories
    """
    answer: str
    reason: str


def validate_prediction(prediction: str) -> bool:
    """
    Validate that the prediction contains valid category names.
    
    Args:
        prediction: Comma-separated list of predicted categories
        
    Returns:
        True if all categories are valid, False otherwise
    """
    predictions = [pred.strip().lower() for pred in prediction.split(',')]
    return all(pred in ALL_CATEGORIES for pred in predictions)


def generate_prediction(llm, 
                       prompt: str, 
                       think: bool, 
                       text_id: str, 
                       save_file: str, 
                       max_retries: int = 3) -> Tuple[str, str]:
    """
    Generate tagging predictions using the language model with retry mechanism.
    
    Args:
        llm: Language model session
        prompt: Formatted prompt text
        think: Whether to enable thinking mode
        text_id: Unique identifier for the text
        save_file: Path to save results (for error logging)
        max_retries: Maximum number of retry attempts
        
    Returns:
        Tuple of (prediction, reason) or (None, None) if failed
    """
    for retry_count in range(max_retries):
        try:
            response = llm.generate_data(
                prompt=(prompt if think else r'/set nothink ' + prompt),
                response_model=TaggingResponseModel,
            )
            
            response_json = response.model_dump()
            prediction = response_json['answer']
            reason = response_json['reason']
            
            # Validate prediction categories
            if not validate_prediction(prediction):
                raise ValueError(f"Invalid prediction: {prediction}")
            
            return prediction, reason
            
        except (InstructorRetryException, IncompleteOutputException, ValueError) as e:
            if retry_count == max_retries - 1:
                console.log(f"[ERROR] Failed after {max_retries} attempts for text ID {text_id}. Last error: {str(e)}")
                # Log error to file
                with open(f"{save_file}.errors.log", 'a', encoding='utf-8') as f:
                    f.write(f"{text_id}\t{str(e)}\n")
                return None, None
            
            console.log(f"Attempt {retry_count + 1} failed with error: {str(e)}. Retrying...")
    
    return None, None