"""
Pydantic models and LLM interaction functionality.
"""

from pydantic import BaseModel
from rich.console import Console
from instructor.exceptions import InstructorRetryException, IncompleteOutputException

console = Console()


class AnswerResponseModel(BaseModel):
    """
    Model for LLM response validation.
    
    Attributes:
        answer: The selected multiple-choice option (A-E)
    """
    answer: str


def generate_prediction(llm, prompt: str, think: bool, text_id: str, save_file: str, max_retries: int = 3) -> str:
    """
    Generate prediction with retry mechanism.
    
    Args:
        llm: The LLM session object
        prompt: The formatted prompt to send to the LLM
        think: Whether to use thinking mode
        text_id: The ID of the text being processed (for logging)
        save_file: Base path for error logging
        max_retries: Maximum number of retry attempts
        
    Returns:
        The predicted answer (A-E) or None if all attempts fail
    """
    for retry_count in range(max_retries):
        try:
            response = llm.generate_data(
                prompt=(prompt if think else r'/set nothink ' + prompt),
                response_model=AnswerResponseModel,
            )
            
            response_json = response.model_dump()
            prediction = response_json['answer'].strip().upper()
            
            # Validate prediction
            if prediction not in ['A', 'B', 'C', 'D', 'E']:
                raise ValueError(f"Invalid prediction: {prediction}")
            
            return prediction
            
        except (InstructorRetryException, IncompleteOutputException, ValueError) as e:
            if retry_count == max_retries - 1:
                console.log(f"[ERROR] Failed after {max_retries} attempts for text ID {text_id}. Last error: {str(e)}")
                # Log error to file
                with open(f"{save_file}.errors.log", 'a', encoding='utf-8') as f:
                    f.write(f"{text_id}\t{str(e)}\n")
                return None
            
            console.log(f"Attempt {retry_count + 1} failed with error: {str(e)}. Retrying...")
    
    return None
