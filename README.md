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

## Citation

```bibtex
@misc{wang2025drivelologychallengingllmsinterpreting,
      title={Drivel-ology: Challenging LLMs with Interpreting Nonsense with Depth}, 
      author={Yang Wang and Chenghao Xiao and Chia-Yi Hsiao and Zi Yan Chang and Chi-Li Chen and Tyler Loakman and Chenghua Lin},
      year={2025},
      eprint={2509.03867},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.03867}, 
}

@inproceedings{wang-etal-2025-drivel,
    title = "Drivel-ology: Challenging {LLM}s with Interpreting Nonsense with Depth",
    author = "Wang, Yang  and
      Xiao, Chenghao  and
      Hsiao, Chia-Yi  and
      Chang, Zi Yan  and
      Chen, Chi-Li  and
      Loakman, Tyler  and
      Lin, Chenghua",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = "nov",
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1177/",
    doi = "10.18653/v1/2025.emnlp-main.1177",
    pages = "23085--23107",
    ISBN = "979-8-89176-332-6"
}
```
