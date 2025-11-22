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

## Tasks

### 1. Multiple-Choice Question Answering (MCQA)

This multiple-choice question answering (MCQA) task asks the model to pick the correct narrative for a Drivelology sample from several options.

#### Easy MCQA

The Easy version offers one correct answer and four distractors.

```bash
bash mcqa_easy.sh
```

#### Hard MCQA

The Hard version adds a "none of the above" option, requiring deeper reasoning, as this option should only be chosen if none of the provided narratives adequately capture the underlying meaning of the Drivelology sample. 

```bash
bash mcqa_hard.sh
```

### 2. Detection

This binary classification task asks LLMs to identify whether a text is Drivelology or not. 

```bash
bash detection.sh
```

### 3. Narrative Writing

This task assesses the model's ability to generate a coherent and meaningful implicit narrative that underlies a given Drivelology sample. This task challenges the model to move beyond surface-level comprehension and demonstrate social and non-linear logical reasoning skills. 

```bash
bash narrative.sh
```

### 4. Multi-label Tagging

The model is asked to assign one or more categories (i.e., Misdirection, Paradox, Switchbait, Inversion, Wordplay), to each Drivelology sample. Since samples often fit multiple categories, this is a multi-label task. Annotators select the most fitting tags, capturing the layered nature of Drivelology.

```bash
bash tagging.sh
```
