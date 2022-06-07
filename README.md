## Continuous Convolutional Neural Networks: Towards a General Purpose CNN for Long Range Dependencies in $\mathrm{N}$D
This repository contains the source code for the paper:

[Towards a General Purpose CNN for Long Range Dependencies in N-D]() 

**Abstract**

The use of Convolutional Neural Networks (CNNs) is widespread in Deep Learning due to a range of desirable model properties which 
result in an efficient and effective machine learning framework. However, performant CNN architectures must be tailored to specific
tasks in order to incorporate considerations such as the input length, resolution, and dimentionality. In this work, we overcome 
the need for problem-specific CNN architectures with our *Continuous Convolutional Neural Network* (CCNN): a single CNN architecture 
equipped with continuous convolutional kernels that can be used for tasks on data of arbitrary resolution, dimensionality and length
without structural changes. Continuous convolutional kernels model long range dependencies at every layer, and remove the need for
downsampling layers and task-dependent depths needed in current CNN architectures. We show the generality of our approach by applying
the same CCNN to a wide set of tasks on sequential (1$\mathrm{D}$) and visual data (2$\mathrm{D}$). Our CCNN performs competitively
and often outperforms the current state-of-the-art across all tasks considered.

### Installation

#### conda
We provide an environment file; ``environment.yml`` containing the required dependencies. Clone the repo and run the following command in the root of this directory:
```
conda env create -f environment.yml
```
If you would like to install pytorch without cuda, instead run:
```
conda env create -f environment-nocuda.yml
```

### Repository Structure
This repository is organized as follows:
- ``ckconv`` contains the main PyTorch library of our model.
- ``models`` contains the model architectures used in our experiments.
- ``datamodules`` contains the pytorch lightning datamodules used in our experiments.
- ``cfg`` contains the configuration file in which to specify default arguments to be passed to the script.

### Using the code

All experiments are run with `main.py`. Flags are handled by [Hydra](https://hydra.cc/docs/intro).
See `cfg/config.yaml` for all available flags. Flags can be passed as `xxx.yyy=value`.

#### Useful flags

- `net.*` describes settings for the models (model definition `models/resnet.py`).
- `kernel.*` describes settings for the MAGNet kernel generator networks.
- `mask.*` describes settings for the FlexConv Gaussian mask.
- `conv.*` describes settings for the convolution operation. It can be used to switch between FlexConv, CKConv, regular Conv, and their separable variants.
- `dataset.*` specifies the dataset to be used, as well as variants, e.g., permuted, sequential.
- `train.*` specifies the settings used for the Trainer of the models.
- `train.do=False`: Only test the model. Useful in combination with pre-training.
- `optimizer.*` specifies and configures the optimizer to be used.
- `debug=True`: By default, all experiment scripts connect to Weights & Biases to log the experimental results. Use this flag to run without connecting to Weights & Biases.
- `pretrained.*`: Use these to load checkpoints before training.

### Reproducing experiments
Please see the [experiments README](/experiments/README.md) for details on reproducing the paper's experiments.

### Cite
If you found this work useful in your research, please consider citing:

```
@misc{romero2022towards,
      title={Towards a General Purpose CNN for Long Range Dependencies in $\mathrm{N}$D}, 
      author={David W. Romero and David M. Knigge and Albert Gu and Erik J. Bekkers and Efstratios Gavves and Jakub M. Tomczak and Mark Hoogendoorn},
      year={2022},
}
```

### Acknowledgements

This work is supported by the [Qualcomm Innovation Fellowship (2021)](https://www.qualcomm.com/research/research/university-relations/innovation-fellowship/2021-europe) 
granted to David W. Romero. David W. Romero sincerely thanks Qualcomm for his support. David W. Romero is financed as part of the
Efficient Deep Learning (EDL) programme (grant number P16-25), partly funded by the Dutch Research Council (NWO). David Knigge is 
partially funded by Elekta Oncology Systems AB and a RVO public-private partnership grant (PPS2102).

This work was carried out on the Dutch national infrastructure with the support of SURF Cooperative.