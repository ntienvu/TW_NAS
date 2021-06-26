# Code for "Optimal Transport Kernels for Sequential and Parallel Neural Architecture Search" at ICML2021

## Requirements
- tensorflow == 1.14.0
- pytorch == 1.2.0, torchvision == 0.4.0
- pot == 0.7 https://pythonot.github.io/# 
- matplotlib, jupyter
- nasbench101 (follow the installation instructions [here](https://github.com/google-research/nasbench))
- nasbench201 (follow the installation instructions [here](https://github.com/D-X-Y/NAS-Bench-201))

## Dataset
To run on NASBench101, download `nasbench_only108.tfrecord` and place it in the top level folder of this repo.
To run on NASBench201, download `NAS-Bench-201-v1_0-e61699.pth` and place it in the top level folder of this repo.

## Sequential NAS on the NASBench search space
```
python run_experiments/run_experiments_sequential.py
```
This will run the sequential NAS setting including the BO algorithm against several other sequential NAS algorithms on the NASBench101 search space.

## Batch NAS on the NASBench search space
```
python run_experiments/run_experiments_batch.py
```
This will run the batch NAS setting including the k-DPP quality algorithm against several other batch baseline algorithms on the NASBench201 search space.


To customize your experiment, open `params.py`. Here, you can change the hyperparameters and the algorithms to run.


We adapt the source code from BANANAS to enable the fair comparison with BANANAS and other baselines https://github.com/naszilla/bananas
