"""
Configuration classes and constants for MCQA tasks.
"""

from typing import Dict
from dataclasses import dataclass


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


# Define all prompt templates
PROMPT_CONFIGS_EASY = {
    'v1_en': PromptConfig(
        name='v1_en',
        language='en',
        description='Basic English prompt template',
        template="""
Tell me the best option in the following options which represents the underlying narrative of the text?

Text: {text}

A. {narrative_1}
B. {narrative_2}
C. {narrative_3}
D. {narrative_4}
E. {narrative_5}

Output format should be JSON with the following keys:
 - answer: The option the text belongs to, and it should be uppercase among A, B, C, D, E.
"""
    ),
    
    'v2_en': PromptConfig(
        name='v2_en',
        language='en',
        description='Improved English prompt with clearer instructions',
        template="""
Tell me the best option from the list below that represents the underlying narrative of the text.
By "underlying narrative," we mean the main theme, implicit message, or perspective the text conveys.

Text: {text}

A. {narrative_1}
B. {narrative_2}
C. {narrative_3}
D. {narrative_4}
E. {narrative_5}

If more than one option seems plausible, pick the one that best represents the main narrative.

Output format should be JSON with the following keys:

 - answer: The option the text belongs to, and it should be a single uppercase letter among A, B, C, D, or E.

Output only the JSON object, with no extra commentary.

Example output:

```json
{{
  "answer": "B"
}}
```
"""
    ),
    
    'v3_en': PromptConfig(
        name='v3_en',
        language='en',
        description='Concise English prompt',
        template="""
Tell me which option best represents the underlying narrative (main theme, message, or perspective) of the text.

Text: {text}

A. {narrative_1}
B. {narrative_2}
C. {narrative_3}
D. {narrative_4}
E. {narrative_5}

If more than one fits, pick the best.

Output only JSON:

 - answer: One uppercase letter: A, B, C, D, or E.
 - reason: Brief explanation.

Example:

```json
{{
  "answer": "B"
}}
```
"""
    ),
    
    'v1_zh_tw': PromptConfig(
        name='v1_zh_tw',
        language='zh_tw',
        description='Chinese prompt template',
        template="""
請從下列選項中選出最能代表文本潛在敘述的最佳選項。

所謂「潛在敘述」，指的是文本所傳達的主要主題、隱含信息或觀點。

文本：{text}
A. {narrative_1}
B. {narrative_2}
C. {narrative_3}
D. {narrative_4}
E. {narrative_5}

如果有多個選項看似合理，請選擇最能代表主要敘述的那一個。

輸出格式應為 JSON，包含以下鍵值：
 - answer: 文本所屬的選項，應為 A、B、C、D 或 E 中的一個大寫字母。

僅輸出 JSON 對象，無需額外評論。

輸出範例：
```json
{{
  "answer": "B"
}}
```
"""
    ), 

    'v2_zh_tw': PromptConfig(
        name='v2_zh_tw',
        language='zh_tw',
        description='Chinese prompt template',
        template="""
請告訴我以下選項中，哪一個最能代表文本的底層敘事？

文本：{text}

A. {narrative_1}
B. {narrative_2}
C. {narrative_3}
D. {narrative_4}
E. {narrative_5}

輸出格式應為 JSON，包含以下鍵值：

 - answer: 該文本所屬的選項，應為大寫的 A、B、C、D 或 E。
"""
    ), 

    'v3_zh_tw': PromptConfig(
        name='v3_zh_tw',
        language='zh_tw',
        description='Chinese prompt template',
        template="""
你現在是一位專業的敘事分析師，擅長從文字中挖掘其核心主題與隱含訊息。

請閱讀以下文本，並仔細分析哪一個選項最能體現該文本的底層敘事，也就是它背後真正想傳達的理念或動機。

文本：
{text}

選項如下：

A. {narrative_1}
B. {narrative_2}
C. {narrative_3}
D. {narrative_4}
E. {narrative_5}

請根據你的專業判斷，選擇最符合的一個選項。

請以 JSON 格式輸出你的分析結果，包含以下欄位：

 - answer: 所選選項，僅限大寫字母 A 到 E
"""
    )
}


PROMPT_CONFIGS_HARD = {
    'v1_en': PromptConfig(
        name='v1_en',
        language='en',
        description='Basic English prompt template',
        template="""
Tell me the best option in the following options which represents the underlying narrative of the text?

Text: {text}

A. {narrative_1}
B. {narrative_2}
C. {narrative_3}
D. {narrative_4}
E. None of the above.

Output format should be JSON with the following keys:
 - answer: The option the text belongs to, and it should be uppercase among A, B, C, D, E.
"""
    ),
    
    'v2_en': PromptConfig(
        name='v2_en',
        language='en',
        description='Improved English prompt with clearer instructions',
        template="""
Tell me the best option from the list below that represents the underlying narrative of the text.
By "underlying narrative," we mean the main theme, implicit message, or perspective the text conveys.

Text: {text}

A. {narrative_1}
B. {narrative_2}
C. {narrative_3}
D. {narrative_4}
E. None of the above.

If none of the options fully fit, select "E. None of the above." If more than one option seems plausible, pick the one that best represents the main narrative.

Output format should be JSON with the following keys:

 - answer: The option the text belongs to, and it should be a single uppercase letter among A, B, C, D, or E.

Output only the JSON object, with no extra commentary.

Example output:

```json
{{
  "answer": "B"
}}
```
"""
    ),
    
    'v3_en': PromptConfig(
        name='v3_en',
        language='en',
        description='Concise English prompt',
        template="""
Tell me which option best represents the underlying narrative (main theme, message, or perspective) of the text.

Text: {text}

A. {narrative_1}
B. {narrative_2}
C. {narrative_3}
D. {narrative_4}
E. None of the above.

If none fit, choose E. If more than one fits, pick the best.

Output only JSON:
 - answer: One uppercase letter: A, B, C, D, or E.

Example:

```json
{{
  "answer": "B"
}}
```
"""
    ),
    
    'v1_zh_tw': PromptConfig(
        name='v1_zh_tw',
        language='zh_tw',
        description='Chinese prompt template',
        template="""
請從以下選項中選出最能代表文本潛在敘事的最佳選項。

「潛在敘事」指的是文本傳達的主要主題、隱含訊息或觀點。

文本：{text}
A. {narrative_1}
B. {narrative_2}
C. {narrative_3}
D. {narrative_4}
E. 以上皆非。

如果沒有任何選項完全符合，請選擇「E. 以上皆非」。如果有多個選項看似合理，請選擇最能代表主要敘事的那一個。

輸出格式應為 JSON，包含以下鍵值：
 - answer：文本所屬的選項，應為 A、B、C、D 或 E 中的一個大寫字母。

只需輸出 JSON 對象，無需額外評論。

範例輸出：
```json
{{
  "answer": "B"
}}
```
"""
    ), 

    'v2_zh_tw': PromptConfig(
        name='v2_zh_tw',
        language='zh_tw',
        description='Chinese prompt template',
        template="""
請告訴我以下選項中，哪一個最能代表文本的底層敘事？

文本：{text}

A. {narrative_1}
B. {narrative_2}
C. {narrative_3}
D. {narrative_4}
E. 以上皆非

輸出格式應為 JSON，包含以下鍵值：

 - answer: 該文本所屬的選項，應為大寫的 A、B、C、D 或 E。
"""
    ), 

    'v3_zh_tw': PromptConfig(
        name='v3_zh_tw',
        language='zh_tw',
        description='Chinese prompt template',
        template="""
你現在是一位專業的敘事分析師，擅長從文字中挖掘其核心主題與隱含訊息。

請閱讀以下文本，並仔細分析哪一個選項最能體現該文本的底層敘事，也就是它背後真正想傳達的理念或動機。

文本：
{text}

選項如下：

A. {narrative_1}
B. {narrative_2}
C. {narrative_3}
D. {narrative_4}
E. 以上皆非

請根據你的專業判斷，選擇最符合的一個選項。
如果你認為所有選項都不足以代表該文本的底層意義，請選擇 E。

請以 JSON 格式輸出你的分析結果，包含以下欄位：

 - answer: 所選選項，僅限大寫字母 A 到 E
"""
    )
}