"""
Model-related classes and functions for the narrative generation task.

This module contains the Pydantic models for structured responses
and functions for interacting with language models.
"""

import re

from pydantic import BaseModel
from rich.console import Console
from instructor.exceptions import InstructorRetryException, IncompleteOutputException

console = Console()


class NarrativeResponseModel(BaseModel):
    """
    Pydantic model for narrative generation response.
    
    Attributes:
        narrative: The generated narrative text
    """
    narrative: str


class EvaluationResponseModel(BaseModel):
    """
    Pydantic model for narrative evaluation response.
    
    Attributes:
        score: The evaluation score (1-5)
    """
    score: int


def generate_narrative(llm, 
                      prompt: str, 
                      think: bool = False,
                      max_retries: int = 5) -> str:
    """
    Generate a narrative using the language model.
    
    Args:
        llm: Language model session
        prompt: Formatted prompt text
        think: Whether to enable thinking mode
        max_retries: Maximum number of retry attempts
        
    Returns:
        Generated narrative or None if failed
    """
    for retry_count in range(max_retries):
        try:
            response = llm.generate_data(
                prompt=(prompt if think else r'/set nothink ' + prompt),
                response_model=NarrativeResponseModel,
            )
            
            narrative = response.narrative.strip()
            narrative = re.sub(r'\s+', ' ', narrative)
            
            if not narrative:
                raise ValueError("Generated narrative is empty.")
            
            return narrative
            
        except (InstructorRetryException, IncompleteOutputException, ValueError) as e:
            if retry_count == max_retries - 1:
                console.log(f"[ERROR] Failed to generate narrative after {max_retries} attempts: {str(e)}")
                return None
            console.log(f"Attempt {retry_count + 1} failed: {str(e)}")
    
    return None


def evaluate_with_geval(llm,
                       candidate: str, 
                       reference: str,
                       prompt: str,
                       max_retries: int = 5) -> int:
    """
    Evaluate a narrative using G-Eval (LLM as evaluator).
    
    Args:
        llm: Language model session
        candidate: Candidate narrative to evaluate
        reference: Reference narrative to compare against
        prompt: Formatted evaluation prompt
        max_retries: Maximum number of retry attempts
        
    Returns:
        Evaluation score (1-5) or None if failed
    """
    for retry_count in range(max_retries):
        try:
            response = llm.generate_data(
                prompt=prompt,
                response_model=EvaluationResponseModel,
            )
            
            return response.score
            
        except (InstructorRetryException, IncompleteOutputException, ValueError) as e:
            if retry_count == max_retries - 1:
                console.log(f"[ERROR] Failed to evaluate with G-Eval after {max_retries} attempts: {str(e)}")
                return None
            console.log(f"G-Eval attempt {retry_count + 1} failed: {str(e)}")
    
    return None