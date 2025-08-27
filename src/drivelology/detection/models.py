"""
Model-related functions for Drivelology detection task.
"""

from rich.console import Console
from pydantic import BaseModel
from instructor.exceptions import InstructorRetryException, IncompleteOutputException

from drivelology.detection.config import MAX_RETRIES

console = Console()


class DrivelologyResponseModel(BaseModel):
    """
    Pydantic model for LLM response structure.
    
    Attributes:
        answer: Classification result ("Drivelology" or "non-Drivelology")
        reason: Explanation for the classification
    """
    answer: str
    reason: str


def classify_text(llm, prompt: str, think: bool, text_id: str, max_retries: int = MAX_RETRIES) -> tuple:
    """
    Classify a single text with retry mechanism.
    
    Args:
        llm: Language model session
        prompt: Formatted prompt text
        think: Whether to enable thinking mode
        text_id: Unique identifier for the text
        max_retries: Maximum number of retry attempts
        
    Returns:
        Tuple of (prediction, reason) or (None, None) if failed
    """
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = llm.generate_data(
                prompt=(prompt if think else r'/set nothink ' + prompt),
                response_model=DrivelologyResponseModel,
            )

            response_json = response.model_dump()
            reason = response_json['reason']
            prediction = response_json['answer']

            # Validate the prediction
            if prediction not in ['Drivelology', 'non-Drivelology']:
                raise ValueError(f"Invalid prediction: {prediction}")
            
            return prediction, reason
            
        except (InstructorRetryException, IncompleteOutputException, ValueError) as e:
            retry_count += 1
            if retry_count >= max_retries:
                console.log(f"[ERROR] Failed after {max_retries} attempts for text ID {text_id}. Last error: {str(e)}")
                return None, None
            
            console.log(f"Attempt {retry_count} failed with error: {str(e)}.")
    
    return None, None