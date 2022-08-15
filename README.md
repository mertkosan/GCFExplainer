# Global Counterfactual Explainer for Graph Neural Networks

- All codes and datasets are tested with Python 3.8.0, PyTorch 1.7.1, PyTorchGeometric 1.7.0, NumPY 1.21.4, NetworkX 2.5, tqdm 4.53.0. 
- We run our experiments on a machine with 2 NVIDIA GeForce RTX 2080 GPU (8GB of RAM) and 32 Intel  Xeon CPUs (2.10GHz and 128GB of RAM).

## Important files

- gnn.py: training gnn models, we already shared trained ones at data/{dataset_name}/gnn
- vrrw.py: To generate counterfactuals
- summary.py: To generate summary from counterfactuals

### Sample command to run dataset for counterfactual and summary generation

`python vrrw.py --dataset <dataset_name>`

`python summary.py --dataset <dataset_name>`

