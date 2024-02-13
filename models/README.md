Scripts to train [DNABERT-3UTR](dnabert-3utr/), [DNABERT2-3UTR](dnabert2-3utr/), [NT-3UTR](ntrans-3utr/), and [state space](state_space/) models.

## Installation of the model training pipeline

1. Create new [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) environment:

```
conda create -n lm-3utr-models python=3.10
conda activate lm-3utr-models
```
2. Install [Pytorch v.2.0.1](https://pytorch.org/)

3. Install the other requirements using pip:

```
pip install -r requirements.txt
```

4. To train DNABERT-2 models also install 
```
pip install triton==2.0.0.dev20221202 --force --no-dependencies
```

Training of DNABERT-2 is currently only possible on NVIDIA A100 due to the employed flash attention implementation.