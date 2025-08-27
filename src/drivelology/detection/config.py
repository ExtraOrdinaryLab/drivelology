"""
Configuration settings for the Drivelology detection task.
"""

from dataclasses import dataclass

MAX_RETRIES = 5
VALID_ANSWERS = ['Drivelology', 'non-Drivelology']


@dataclass
class PromptConfig:
    """
    Configuration class for managing different prompt templates.
    
    Attributes:
        name: Identifier for the prompt template
        template: The actual prompt template with placeholders
        language: Language code ('en', 'zh_tw', etc.)
        description: Human-readable description of the template
    """
    name: str
    template: str
    language: str
    description: str


# Prompt Templates Dictionary
PROMPT_TEMPLATES = {
    "v1_en": PromptConfig(
        name="v1_en",
        language="en",
        description="Basic English prompt template",
        template="""
Instruction:

Classify whether the given text is a Drivelology sample or not.

Definition:

 - Drivelology: Statements that appear logically coherent but contain deeper, often paradoxical meanings. These challenge conventional interpretation by blending surface-level nonsense with underlying depth, often incorporating elements of humor, irony, or sarcasm, and requiring contextual understanding and emotional insight to unravel their true significance.
 - non-Drivelology: This includes pure nonsense (grammatically correct but semantically meaningless statements, such as "boys will be boys") and normal sentences, including quotes or proverbs, that convey clear or straightforward information without the layered complexity characteristic of Drivelology.

Input Text:

{text}

Output Format:

Please provide the output in JSON format with the following keys:
 - answer: Specify whether the text is "Drivelology" or "non-Drivelology."
 - reason: Provide a clear explanation of why the text is classified as Drivelology or not.
"""
    ),

    "v2_en": PromptConfig(
        name="v2_en",
        language="en",
        description="Enhanced English prompt with examples",
        template="""
Instruction:

Classify whether the given text is a Drivelology sample or not.

Definitions:

 - Drivelology: Statements that appear logically coherent but contain deeper, often paradoxical meanings. These challenge conventional interpretation by blending surface-level nonsense with underlying depth, often incorporating elements of humor, irony, or sarcasm, and requiring contextual understanding and emotional insight to unravel their true significance.
     - Example: "When you love someone, it's hard to hide it. But if you love two people, you'd better keep it hidden."
 - non-Drivelology: This includes pure nonsense (grammatically correct but semantically meaningless statements, such as "Boys will be boys") and normal sentences, including quotes or proverbs, that convey clear or straightforward information without the layered complexity characteristic of Drivelology.
     - Example: "Colourless green ideas sleep furiously."

Input Text:

{text}

Instructions for Reasoning:

Analyze the input text by comparing it to the definitions above. Identify whether it contains logical coherence, paradox, layered meaning, or requires emotional/contextual insight. If uncertain, select the category that best fits and explain your reasoning.

Output Format:

Please provide the output in JSON format with the following keys:

 - answer: Specify "Drivelology" or "non-Drivelology."
 - reason: Clearly explain why the text was classified as such, referencing specific features from the definitions.
"""
    ),

    "v3_en": PromptConfig(
        name="v3_en",
        language="en",
        description="Concise English prompt with JSON output example",
        template="""
Classify the text as "Drivelology" or "non-Drivelology."

Definitions:

 - Drivelology: Logically coherent statements with paradox, layered or hidden meaning, often using humor, irony, or sarcasm. These require emotional or contextual insight to interpret.
     - Example: "When you love someone, it's hard to hide it. But if you love two people, you'd better keep it hidden."
 - non-Drivelology: Pure nonsense (e.g., "Colourless green ideas sleep furiously") or straightforward statements without hidden complexity.

Text:

{text}

Reasoning:

Decide based on the definitions above. If uncertain, choose the closest fit and briefly explain.

Output (JSON only):

```json
{{
  "answer": "Drivelology",
  "reason": "The text contains logical coherence and layered meaning, fitting the Drivelology definition."
}}
```
""" ),
    "v1_zh_tw": PromptConfig(
        name="v1_zh_tw",
        language="zh_tw",
        description="Traditional Chinese prompt template",
        template="""
判斷給定的文本是否為 Drivelology 範例。

定義：

Drivelology：看似邏輯連貫但包含更深層、通常是矛盾意義的陳述。這些通過將表面上的無稽之談與潛在的深度相結合來挑戰傳統解釋，通常融入幽默、諷刺或挖苦的元素，並需要上下文理解和情感洞察來解開其真正意義。
non-Drivelology：包括純粹的無稽之談（語法正確但語義上無意義的陳述，如「男孩就是男孩」）和普通句子，包括傳達明確或簡單信息的引言或諺語，這些信息不具有 Drivelology 特有的層次複雜性。
輸入文本：

{text}

輸出格式：

請使用以下鍵值以 JSON 格式提供輸出：

answer：指定文本是「Drivelology」還是「non-Drivelology」。如若不確定，選擇最符合的類別。

reason：清楚解釋為什麼該文本被分類為 Drivelology 或不是。如果不確定，選擇最符合的類別並解釋原因。 
""" 
    ),

    "v2_zh_tw": PromptConfig(
        name="v2_zh_tw", 
        language="zh_tw", 
        description="Enhanced Traditional Chinese prompt template", 
        template=""" 
指令：

你是一位語言哲學與幽默分析專家。請根據以下定義，判斷給定文本是否屬於 Drivelology 類型。

定義：

Drivelology：看似邏輯連貫但包含更深層、通常是矛盾意義的陳述。這些通過將表面上的無稽之談與潛在的深度相結合來挑戰傳統解釋，通常融入幽默、諷刺或挖苦的元素，並需要上下文理解和情感洞察來解開其真正意義。
non-Drivelology：包括純粹的無稽之談（語法正確但語義上無意義的陳述，如「男孩就是男孩」）和普通句子，包括傳達明確或簡單信息的引言或諺語，這些信息不具有 Drivelology 特有的層次複雜性。
輸入文本：

{text}

請以 JSON 格式 回答，包含以下兩個欄位：

"answer": 判斷結果，僅限於 "Drivelology" 或 "non-Drivelology"。

"reason": 清楚解釋為什麼該文本被歸類為 Drivelology 或 non-Drivelology，引用具體內容作為支持。 
""" 
    ),

    "v3_zh_tw": PromptConfig(
        name="v3_zh_tw", 
        language="zh_tw", 
        description="Cross-language Traditional Chinese prompt template", 
        template=""" 
你是一位跨語言語言風格與幽默分析師 ，擅長識別不同語言中蘊含深層意義的表述方式。

請根據以下定義，判斷給定文本是否屬於 Drivelology 。 即使文本為非中文語言（如英語、法語、日語等），你也應基於該語言的文化背景與語境進行判斷。

定義：

Drivelology：看似邏輯連貫但包含更深層、通常是矛盾意義的陳述。這些通過將表面上的無稽之談與潛在的深度相結合來挑戰傳統解釋，通常融入幽默、諷刺或挖苦的元素，並需要上下文理解和情感洞察來解開其真正意義。
non-Drivelology：包括純粹的無稽之談（語法正確但語義上無意義的陳述，如「男孩就是男孩」）和普通句子，包括傳達明確或簡單信息的引言或諺語，這些信息不具有 Drivelology 特有的層次複雜性。
輸入文本：

{text}

請以 JSON 格式 回答，包含以下兩個欄位：

"answer": 判斷結果，僅限於 "Drivelology" 或 "non-Drivelology"。
"reason": 一段清晰的文字，說明為什麼該文本被歸類為 Drivelology 或 non-Drivelology，引用具體內容作為支持。請以中文 撰寫原因，無論文本本身的語言為何。 
"""
    ) 
}
