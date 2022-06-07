# Continuous Convolutional Neural Networks: Towards a General Purpose CNN for Long Range Dependencies in N-D
___
This repository contains the source code for the paper:

[Towards a General Purpose CNN for Long Range Dependencies in N-D]() 

**Abstract**

The use of Convolutional Neural Networks (CNNs) is widespread in Deep Learning due to a range of desirable model properties which result in an efficient and effective machine learning framework. However, performant CNN\break architectures must be tailored to specific tasks in order to incorporate considerations such as the input length, resolution, and dimentionality. In this work, we overcome the need for problem-specific CNN architectures with our \textit{Continuous Convolutional Neural Network} (CCNN): a single CNN architecture equipped with continuous convolutional kernels that can be used for tasks on data of arbitrary resolution, dimensionality and length without structural changes. Continuous convolutional kernels model long range dependencies at every layer, and remove the need for downsampling layers and task-dependent depths needed in current CNN architectures. We show the generality of our approach by applying the same CCNN to a wide set of tasks on sequential ($1D$) and visual data ($2D$). Our CCNN performs competitively and often outperforms the current state-of-the-art across all tasks considered.

## Installation

### conda
We provide an environment file; ``environment.yml`` containing the required dependencies. Clone the repo and run the following command in the root of this directory:
```
conda env create -f environment.yml
```

## Repository Structure
This repository is organized as follows:
- ``ckconv`` contains the main PyTorch library of our model.
- ``models`` contains the model architectures used in our experiments.
- ``datamodules`` contains the pytorch lightning datamodules used in our experiments.
- ``cfg`` contains the configuration file in which to specify default arguments to be passed to the script.


## Reproducing experiments
Please see the [experiments readme](/experiments/README.md) for details on reproducing the paper's experiments.
