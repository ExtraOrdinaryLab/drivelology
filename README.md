# Drivelology: Challenging LLMs with Interpreting Nonsense with Depth

This repository contains the code and resources for the paper "Drivel-ology: Challenging LLMs with Interpreting Nonsense with Depth," accepted at EMNLP 2025.

> **Update:** Drivelology is now officially supported by `evalscope`! This is now the recommended way to run evaluations. Please refer to [here](https://github.com/modelscope/evalscope/pull/927). The original execution scripts are kept for legacy purposes.

## Setup

```bash
# Create and activate conda environment
conda create --name drivelology python=3.10
conda activate drivelology

# Install dependencies
bash setup.sh

# For Jupyter Notebook users
conda install ipykernel ipywidgets -y
python -m ipykernel install --user --name drivelology --display-name "drivelology"
```

## Dataset

The Drivelology dataset is available on the Hugging Face Hub: [extraordinarylab/drivel-hub](https://huggingface.co/datasets/extraordinarylab/drivel-hub)
