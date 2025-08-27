"""
Configuration settings for the multi-label tagging task.

This module contains dataclasses and dictionaries that define
the configuration options for prompt templates in different languages.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class PromptConfig:
    """
    Configuration for a specific prompt template.
    
    Attributes:
        name: Identifier for the prompt (e.g., "v1_en")
        language: Language of the prompt (e.g., "en", "zh_tw")
        description: Human-readable description
        template: The actual prompt template with placeholders
    """
    name: str
    language: str
    description: str
    template: str


# Define all prompt templates
PROMPT_CONFIGS: Dict[str, PromptConfig] = {
    'v1_en': PromptConfig(
        name='v1_en',
        language='en',
        description='Basic English multilabel classification prompt',
        template="""
Instruction:

Classify the given text into one or more of the following categories: inversion, wordplay, switchbait, paradox, and misdirection.

Definitions:

 - inversion: Constructions that reverse the typical order or meaning of words, phrases, or ideas, producing surprising or ironic effects.
 - wordplay: Creative manipulation of language through puns, double meanings, or phonetic tricks, resulting in humorous or ambiguous interpretations.
 - switchbait: Involves a shift in meaning that hinges on cultural knowledge, language-specific expressions, or code-switching between languages or dialects. Understanding the humour or twist in Switchbait often requires familiarity with particular cultural references, idioms, or linguistic cues that are not immediately obvious to all readers.
 - paradox: Expressions that contain inherent contradictions or logical impossibilities, challenging conventional reasoning. 
 - misdirection: Involves semantic drift or a sudden shift from one topic to another, while still responding within the context of the original topic. This creates an impression of changing the subject, but the reply remains relevant in an unexpected way. 

Input Text:

{text}

Output Format:

Please provide the output in JSON format with the following keys:
 - answer: List the applicable comma-separated categories for the text (e.g., "inversion, misdirection").
 - reason: Provide a clear explanation for why the text is classified into each category.
"""
    ),
    
    'v2_en': PromptConfig(
        name='v2_en',
        language='en',
        description='Improved English multilabel classification prompt',
        template="""
Instruction:

Analyze the input text and classify it into one or more of the following categories: inversion, wordplay, switchbait, paradox, and misdirection. 
Use the definitions below to guide your classification.

Definitions:

 - inversion: Constructions that reverse the typical order or meaning of words, phrases, or ideas, producing surprising or ironic effects.
 - wordplay: Creative manipulation of language through puns, double meanings, or phonetic tricks, resulting in humorous or ambiguous interpretations.
 - switchbait: Involves a shift in meaning that hinges on cultural knowledge, language-specific expressions, or code-switching between languages or dialects. Understanding the humour or twist in Switchbait often requires familiarity with particular cultural references, idioms, or linguistic cues that are not immediately obvious to all readers.
 - paradox: Expressions that contain inherent contradictions or logical impossibilities, challenging conventional reasoning. 
 - misdirection: Involves semantic drift or a sudden shift from one topic to another, while still responding within the context of the original topic. This creates an impression of changing the subject, but the reply remains relevant in an unexpected way. 

Input Text:

{text}

Output Format (JSON):

```json
{{
  "answer": "category1, category2, ...",
  "reason": "Explain why the text fits each chosen category based on the definitions."
}}
```
""" ),
    'v3_en': PromptConfig(
        name='v3_en',
        language='en',
        description='Concise English multilabel classification prompt',
        template="""
Instruction:

Examine the input text and determine which of the following categories it belongs to: inversion, wordplay, switchbait, paradox, and misdirection.
Base your classification strictly on the definitions provided below.

Definitions:

inversion: Constructions that reverse the typical order or meaning of words, phrases, or ideas, producing surprising or ironic effects.
wordplay: Creative manipulation of language through puns, double meanings, or phonetic tricks, resulting in humorous or ambiguous interpretations.
switchbait: Involves a shift in meaning that hinges on cultural knowledge, language-specific expressions, or code-switching between languages or dialects. Understanding the humour or twist in Switchbait often requires familiarity with particular cultural references, idioms, or linguistic cues that are not immediately obvious to all readers.
paradox: Expressions that contain inherent contradictions or logical impossibilities, challenging conventional reasoning.
misdirection: Involves semantic drift or a sudden shift from one topic to another, while still responding within the context of the original topic. This creates an impression of changing the subject, but the reply remains relevant in an unexpected way.
Input Text:

{text}

Please provide the output in JSON format:

```
{{
  "answer": "category1, category2, ...",
  "reason": "Briefly explain how the input text fits each selected category, using the definitions as a basis."
}}
"""
    ),
    
    'v1_zh_tw': PromptConfig(
        name='v1_zh_tw',
        language='zh_tw',
        description='Chinese multilabel classification prompt',
        template="""
指令：

將給定的文本分類為以下一個或多個類別：inversion、wordplay、switchbait、paradox、misdirection。

定義：

 - inversion：顛倒詞語、短語或想法的典型順序或含義的結構，產生令人驚訝或諷刺的效果。
 - wordplay：通過雙關語、雙重含義或語音技巧對語言的創造性操作，產生幽默或模糊的解釋。
 - switchbait：涉及依賴文化知識、特定語言表達或語言/方言之間代碼轉換的意義轉換。理解轉換誘餌中的幽默或轉折通常需要熟悉特定的文化參考、習語或語言線索，這些對所有讀者來說並不立即明顯。
 - paradox：包含內在矛盾或邏輯不可能性的表達，挑戰傳統推理。
 - misdirection：涉及語義漂移或從一個主題突然轉向另一個主題，同時仍在原始主題的語境內回應。這給人一種改變主題的印象，但回應以意想不到的方式保持相關性。

輸入文本：

{text}

輸出格式：

請以 JSON 格式提供輸出，包含以下鍵值：
 - answer: 列出適用於文本的逗號分隔類別（例如，"inversion, wordplay, ..."）。
 - reason: 清楚解釋為什麼文本被分類到每個類別的原因。
"""
    ), 

    'v2_zh_tw': PromptConfig(
        name='v2_zh_tw',
        language='zh_tw',
        description='Chinese multilabel classification prompt',
        template="""
你是一位語言幽默與修辭分析專家。請根據以下五種分類，對給定文本中使用的語言技巧或修辭手法進行精確識別與分類。

你可以選擇一個或多個適用的類別：inversion、wordplay、switchbait、paradox、misdirection。

類別定義：

 - inversion：顛倒詞語、短語或想法的典型順序或含義的結構，產生令人驚訝或諷刺的效果。
 - wordplay：通過雙關語、雙重含義或語音技巧對語言的創造性操作，產生幽默或模糊的解釋。
 - switchbait：涉及依賴文化知識、特定語言表達或語言/方言之間代碼轉換的意義轉換。理解轉換誘餌中的幽默或轉折通常需要熟悉特定的文化參考、習語或語言線索，這些對所有讀者來說並不立即明顯。
 - paradox：包含內在矛盾或邏輯不可能性的表達，挑戰傳統推理。
 - misdirection：涉及語義漂移或從一個主題突然轉向另一個主題，同時仍在原始主題的語境內回應。這給人一種改變主題的印象，但回應以意想不到的方式保持相關性。

輸入文本：

{text}

輸出格式：

請以 JSON 格式 回答，包含以下兩個欄位：
 - "answer": 一個逗號分隔的字符串，列出所有適用的類別名稱（例如："inversion, wordplay"）
 - "reason": 一段清晰的文字，說明為何將該文本歸類到每個指定類別，並引用具體內容作為支持。

示例輸出（僅供格式參考）：

{{
  "answer": "wordplay, misdirection",
  "reason": "文本使用了『花』這個字的雙關語，同時在語境中從自然植物轉移到人際關係，但在整體脈絡中仍保持一致性。"
}}
"""
    ), 

    'v3_zh_tw': PromptConfig(
        name='v3_zh_tw',
        language='zh_tw',
        description='Chinese multilabel classification prompt',
        template="""
指令：
你是一位跨語言幽默與修辭分析專家 。請根據以下五種分類，對給定文本中使用的語言技巧或修辭手法進行精確識別與分類。

你可以選擇一個或多個適用的類別：inversion、wordplay、switchbait、paradox、misdirection。

文本可以是任何語言（如英語、中文、西班牙語等）。你需要基於該語言的文化與語言特點來判斷其使用了哪些修辭手法。

類別定義：

 - inversion：顛倒詞語、短語或想法的典型順序或含義的結構，產生令人驚訝或諷刺的效果。
 - wordplay：通過雙關語、雙重含義或語音技巧對語言的創造性操作，產生幽默或模糊的解釋。
 - switchbait：涉及依賴文化知識、特定語言表達或語言/方言之間代碼轉換的意義轉換。理解轉換誘餌中的幽默或轉折通常需要熟悉特定的文化參考、習語或語言線索，這些對所有讀者來說並不立即明顯。
 - paradox：包含內在矛盾或邏輯不可能性的表達，挑戰傳統推理。
 - misdirection：涉及語義漂移或從一個主題突然轉向另一個主題，同時仍在原始主題的語境內回應。這給人一種改變主題的印象，但回應以意想不到的方式保持相關性。

輸入文本：

{text}

輸出格式：

請以 JSON 格式 回答，包含以下兩個欄位：
 - "answer": 一個逗號分隔的字符串，列出所有適用的類別名稱（例如："inversion, wordplay"）
 - "reason": 一段清晰的文字，說明為何將該文本歸類到每個指定類別，並引用具體內容作為支持。請以中文 撰寫原因，無論文本本身的語言為何。

示例輸出（僅供格式參考）：

{{
  "answer": "wordplay, misdirection",
  "reason": "文本使用了『花』這個字的雙關語，同時在語境中從自然植物轉移到人際關係，但在整體脈絡中仍保持一致性。"
}}
"""
    )
}