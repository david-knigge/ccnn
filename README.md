## Modelling Long Range Dependencies in N-D: From Task-Specific to a General Purpose CNN

Code repository of the paper [Modelling Long Range Dependencies in N-D: From Task-Specific to a General Purpose CNN](https://arxiv.org/abs/2301.10540).

**Abstract**

Performant Convolutional Neural Network (CNN) architectures must be tailored to specific tasks in order to consider the length, resolution, and dimensionality of the input data. In this work, we tackle the need for problem-specific CNN architectures.\break We present the \textit{Continuous Convolutional Neural Network} (CCNN): a single CNN able to process data of arbitrary resolution, dimensionality and length without any structural changes.  Its key component are its \textit{continuous convolutional kernels} which model long-range dependencies at every layer, and thus remove the need of current CNN architectures for task-dependent downsampling and depths. We showcase the generality of our method by using the \emph{same architecture} for tasks on sequential (1D), visual (2D) and point-cloud (3D) data. Our CCNN matches and often outperforms the current state-of-the-art across all tasks considered.
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
@article{knigge2023modelling,
  title={Modelling Long Range Dependencies in N-D: From Task-Specific to a General Purpose CNN},
  author={Knigge, David M and Romero, David W and Gu, Albert and Bekkers, Erik J and Gavves, Efstratios and Tomczak, Jakub M and Hoogendoorn, Mark and Sonke, Jan-Jakob},
  journal={International Conference on Learning Representations},
  year={2023}
}
```

### Acknowledgements

This work is supported by the [Qualcomm Innovation Fellowship (2021)](https://www.qualcomm.com/research/research/university-relations/innovation-fellowship/2021-europe) 
granted to David W. Romero. David W. Romero sincerely thanks Qualcomm for his support. David W. Romero is financed as part of the
Efficient Deep Learning (EDL) programme (grant number P16-25), partly funded by the Dutch Research Council (NWO). David Knigge is 
partially funded by Elekta Oncology Systems AB and a RVO public-private partnership grant (PPS2102).

This work was carried out on the Dutch national infrastructure with the support of SURF Cooperative.
