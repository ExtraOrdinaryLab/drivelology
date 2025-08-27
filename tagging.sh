#!/bin/bash

PROMPT_VERSION="v1_en"
LLM_PROVIDER="openai"
LLM_MODEL="gpt-4o-mini"

python -m drivelology.bin.tagging \
    --prompt_version $PROMPT_VERSION \
    --llm_provider $LLM_PROVIDER \
    --llm_model $LLM_MODEL \
    --output_dir "outputs/tagging" \
    --dataset_name "extraordinarylab/drivel-hub" \
    --dataset_config "v0618"