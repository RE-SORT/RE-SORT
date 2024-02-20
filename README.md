# RE-SORT

This is an official implementation of ***RE-SORT*** for CTR prediction task, as described in our paper:

[RE-SORT: Removing Spurious Correlation in Multilevel Interaction for CTR Prediction](https://arxiv.org/pdf/2309.14891.pdf). arXiv preprint:2309.14891, 2024.

![Overview Framework](./re-sort.png)

## Introduction

RE-SORT: A CTR prediction framework that removes spurious correlations in multilevel feature interactions, which leverages critical causal relationships between items and users in diverse nonlinear feature spaces to enhance the CTR prediction.

## Dependencies

RE-SORT has the following dependencies:

+ python 3.6+
+ pytorch 1.10+ 

## Quick Start

python run_expid.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}


## 🔥 Citation
If you find our RE-SORT helpful for your research, please consider citing the following paper:
```
@article{song2024resort,
  Title={RE-SORT: Removing Spurious Correlation in Multilevel Interaction for CTR Prediction},
  journal={arXiv preprint:2309.14891},
  year={2024}
}
```
