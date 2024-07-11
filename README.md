# GMOCAT-CODE
Official Code for paper "GMOCAT: A Graph-Enhanced Multi-Objective Method for Computerized Adaptive Testing" (KDD 2023)
## requirements:
Python, PyTorch, dgl

## how to run codes
We have provided the preprocessed dataset `assist2009` so that we can directly run experiments on `IRT` with `assist2009`.
To run GMOCAT, please run `train_gcat.sh`.


## how to run codes from scratch
1. put raw data in `raw_data/`.
2. run `preprocessing.py` and `construct_graphs.py`.
3. run `pretrain.sh`.
4. run the selection algorithms `train_gcat.sh`.

CAT baselines can be found in https://github.com/bigdata-ustc/CAT, https://github.com/bigdata-ustc/NCAT and https://github.com/arghosh/BOBCAT.
