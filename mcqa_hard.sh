#!/bin/bash

PROMPT_VERSION="v1_en"
LLM_PROVIDER="openai" # openai, deepseek, anthropic
LLM_MODEL="gpt-4o-mini" # gpt-4o-mini, deepseek-chat, claude-3-5-haiku-20241022

python -m drivelology.bin.mcqa_hard \
    --prompt_version $PROMPT_VERSION \
    --llm_provider $LLM_PROVIDER \
    --llm_model $LLM_MODEL \
    --dataset_name extraordinarylab/drivel-hub