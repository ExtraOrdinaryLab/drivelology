#!/bin/bash

GENERATION_VERSION="v1_en"
EVALUATION_VERSION="v1_en"
LLM_PROVIDER="openai" # openai, deepseek, anthropic
LLM_MODEL="gpt-4o-mini" # gpt-4o-mini, deepseek-chat, claude-3-5-haiku-20241022

python -m drivelology.bin.narrative \
    --generation_version $GENERATION_VERSION \
    --evaluation_version $EVALUATION_VERSION \
    --llm_provider $LLM_PROVIDER \
    --llm_model $LLM_MODEL \
    --output_dir "outputs/narrative" \
    --dataset_name "extraordinarylab/drivel-hub" \
    --dataset_config "v0618"