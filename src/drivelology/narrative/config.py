"""
Configuration settings for the narrative generation task.

This module contains enums, dataclasses, and dictionaries that define
the configuration options for prompt templates and languages.
"""

from enum import Enum
from typing import Dict
from dataclasses import dataclass


class Language(Enum):
    """
    Supported languages for narrative generation and evaluation.
    """
    ENGLISH = "en"
    CHINESE_TRADITIONAL = "zh_tw"
    CHINESE_SIMPLIFIED = "zh"


class PromptType(Enum):
    """
    Types of prompts used in the narrative tasks.
    """
    NARRATIVE_GENERATION = "narrative_generation"
    NARRATIVE_EVALUATION = "narrative_evaluation"


@dataclass
class PromptVersion:
    """
    Configuration for a specific prompt version.
    
    Attributes:
        version: Identifier for the prompt version (e.g., "v1_en")
        language: Language of the prompt
        prompt_type: Type of task the prompt is designed for
        template: The actual prompt template with placeholders
        description: Human-readable description
    """
    version: str
    language: Language
    prompt_type: PromptType
    template: str
    description: str = ""


class PromptManager:
    """
    Manager for accessing and registering prompt templates.
    """
    def __init__(self):
        """Initialize the prompt manager with default templates."""
        self.prompts: Dict[str, PromptVersion] = {}
        self._initialize_prompts()
    
    def _initialize_prompts(self):
        """Initialize the default prompt templates."""
        # English narrative generation prompts
        self.register_prompt(PromptVersion(
            version="v1_en",
            language=Language.ENGLISH,
            prompt_type=PromptType.NARRATIVE_GENERATION,
            template="""
You need to first read and understand the text given. 
Generate a detailed description to illustrate the implicit narrative of the text.

Text: {text}

Output format should be JSON with the following keys:
 - narrative: The narrative of the text in English.
""",
            description="Basic narrative generation prompt"
        ))
        
        self.register_prompt(PromptVersion(
            version="v2_en",
            language=Language.ENGLISH,
            prompt_type=PromptType.NARRATIVE_GENERATION,
            template="""
Read and understand the provided text carefully. 

Task: Generate a detailed description that illustrates the implicit narrative of the text.

Input Text: {text}

Output format should be JSON with the following keys:
 - narrative: The narrative of the text in English.
 """,
            description="Improved narrative generation prompt with clear structure"
        ))
        
        self.register_prompt(PromptVersion(
            version="v3_en",
            language=Language.ENGLISH,
            prompt_type=PromptType.NARRATIVE_GENERATION,
            template="""
Read and understand the provided text carefully.

Task:

Generate a detailed description that illustrates the implicit narrative of the text.
By "implicit narrative," we mean the underlying message, theme, perspective, or emotional undertone that is not directly stated but can be inferred from the text. 
Your description should go beyond surface-level summary and provide insights into the text's underlying themes, perspectives, or intentions.

Input Text:

{text}

Output format should be JSON with the following keys:
 - narrative: The narrative of the text in English.
""",
            description="Advanced narrative generation with detailed instructions"
        ))
        
        # Traditional Chinese narrative generation prompts
        self.register_prompt(PromptVersion(
            version="v1_zh_tw",
            language=Language.CHINESE_TRADITIONAL,
            prompt_type=PromptType.NARRATIVE_GENERATION,
            template="""
請仔細閱讀並理解提供的文本。

任務：生成詳細描述來說明文本的隱含敘事。

輸入文本：{text}

請確保輸出為JSON格式，包含"narrative"鍵值，內容為從輸入文本推導出的隱含敘事詳細描述。
 - narrative: 文本 narrative 必須為中文.
""",
            description="Basic Traditional Chinese narrative generation prompt"
        ))

        self.register_prompt(PromptVersion(
            version="v2_zh_tw",
            language=Language.CHINESE_TRADITIONAL,
            prompt_type=PromptType.NARRATIVE_GENERATION,
            template="""
指令：
你是一位敘事分析師 ，擅長從文本中挖掘隱含的意義與深層敘事。

請仔細閱讀並理解以下文本，然後撰寫一段詳細描述 ，清楚表達該文本所蘊含的implicit narrative，也就是那些沒有直接說出來，但可從語境、語氣與用詞中推斷出的潛在訊息、隱含立場或底層故事。

輸入文本：

{text}

輸出格式：

請以 JSON 格式 回答，包含以下欄位：
 - "narrative": 用中文 寫出對該文本隱性敘事的清晰總結。
""",
            description="Enhanced Traditional Chinese narrative generation prompt"
        ))

        self.register_prompt(PromptVersion(
            version="v3_zh_tw",
            language=Language.CHINESE_TRADITIONAL,
            prompt_type=PromptType.NARRATIVE_GENERATION,
            template="""
你是一位文學分析師 ，專門從事文本背後的潛在敘事與社會文化脈絡研究。

請按照以下步驟分析給定文本：

1. 閱讀並理解文本內容
2. 識別關鍵詞、語氣、語境與可能的情感傾向
3. 推斷作者的立場、受眾、以及可能的隱含訊息
4. 總結出一個清晰的隱性敘事

輸入文本：

{text}

輸出格式：

請以 JSON 格式 回答，包含以下欄位：
 - "narrative": 用中文寫出對該文本隱性敘事的清晰總結。
""",
            description="Advanced Traditional Chinese narrative generation prompt with structured approach"
        ))
        
        # English evaluation prompts
        self.register_prompt(PromptVersion(
            version="v1_en",
            language=Language.ENGLISH,
            prompt_type=PromptType.NARRATIVE_EVALUATION,
            template="""
Task:

Evaluate how accurately the candidate narrative matches the given reference narrative. 
Use a scale from 1 to 5, where 1 indicates the least accuracy and 5 indicates the highest accuracy.

 - Candidate Narrative: {candidate}
 - Reference Narrative: {reference}

Output Format:

Please provide the output in JSON format with the following key:
 - score: The score indicating the level of matching, ranging from 1 to 5.
""",
            description="Basic English narrative evaluation prompt"
        ))
        
        # Traditional Chinese evaluation prompts
        self.register_prompt(PromptVersion(
            version="v1_zh_tw",
            language=Language.CHINESE_TRADITIONAL,
            prompt_type=PromptType.NARRATIVE_EVALUATION,
            template="""
任務：

評估候選敘事與給定參考敘事的匹配準確度。
使用1到5的評分標準，其中1表示最低準確度，5表示最高準確度。

 - 候選敘事：{candidate}
 - 參考敘事：{reference}

輸出格式：

請以JSON格式提供輸出，包含以下鍵值：
 - score: 表示匹配程度的分數，範圍從1到5。
""",
            description="Basic Traditional Chinese narrative evaluation prompt"
        ))
    
    def register_prompt(self, prompt_version: PromptVersion):
        """
        Register a new prompt version.
        
        Args:
            prompt_version: The prompt version to register
        """
        key = f"{prompt_version.prompt_type.value}_{prompt_version.version}"
        self.prompts[key] = prompt_version
    
    def get_prompt(self, prompt_type: PromptType, version: str) -> PromptVersion:
        """
        Get a prompt by its type and version.
        
        Args:
            prompt_type: Type of prompt (generation or evaluation)
            version: Version identifier (e.g., "v1_en")
            
        Returns:
            PromptVersion object
            
        Raises:
            ValueError: If the prompt is not found
        """
        key = f"{prompt_type.value}_{version}"
        if key not in self.prompts:
            raise ValueError(f"Prompt not found: {key}")
        return self.prompts[key]
    
    def list_prompts(self, prompt_type: PromptType = None, language: Language = None) -> Dict[str, PromptVersion]:
        """
        List available prompts, optionally filtered by type and language.
        
        Args:
            prompt_type: Filter by prompt type (optional)
            language: Filter by language (optional)
            
        Returns:
            Dictionary of matching prompts
        """
        filtered_prompts = {}
        for key, prompt in self.prompts.items():
            if prompt_type and prompt.prompt_type != prompt_type:
                continue
            if language and prompt.language != language:
                continue
            filtered_prompts[key] = prompt
        return filtered_prompts


# Constants
MAX_RETRIES = 5