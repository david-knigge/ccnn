# Experiments

We provide the commands used to run the experiments published in the paper. 

**Checkpoints** For selected models, we provide model checkpoints which may be used to directly reproduce results. These models should be deposited in a folder ``/artifacts`` at the root of the project directory. To then load a checkpointed model use script parameters ``cfg.pretrained.load=True`` and ``cfg.pretrained.filename=...`` where you insert the filename of the checkpointed model. 

**Randomness** Please note that due to randomness in certain PyTorch operations on CUDA, it may not be possible to reproduce certain results with high precision. Please see PyTorch's manual on deterministic behavior for more details, as well as run_experiments.py::set_manual_seed() for specifications on how we seed our experiments.

**Running an experiment** We use [hydra](https://hydra.cc/docs/intro/) for managing the configuration of our experiments. Experiments are ran from the ``main.py`` script in the root of this repo, and hyperparameters from the [config file](/cfg/config.yaml) that you would like overwritten are specified using dot notation. For example, the following would overwrite the number of hidden channels; ``python main.py net.hidden_size=64 ...``. Below, you can find an overview of the commands and configurations used in each of the reported experiments. To reproduce an experiment, copy the corresponding command from below and execute it in the root of this repo. 

## Table 1: Pixel-level 1D image classification

**sMNIST CCNN-4,140**:
```
conv.bias=True conv.causal=True conv.type=SeparableFlexConv conv.use_fft=True dataset.data_type=sequence dataset.name=MNIST dataset.params.permuted=False kernel.bias=True kernel.chang_initialize=True kernel.no_hidden=32 kernel.no_layers=3 kernel.omega_0=2976.4910261630357 kernel.type=MAGNet mask.dynamic_cropping=True mask.init_value=0.075 mask.threshold=0.1 mask.type=gaussian net.block.type=S4 net.dropout=0.1 net.dropout_in=0 net.dropout_type=Dropout2d net.no_blocks=4 net.no_hidden=140 net.nonlinearity=GELU net.norm=BatchNorm net.type=ResNet optimizer.lr=0.01 optimizer.mask_lr_ratio=1 optimizer.name=AdamW optimizer.weight_decay=1e-06 scheduler.name=cosine scheduler.warmup_epochs=10 test.batch_size_multiplier=1 test.before_train=False train.batch_size=100 train.epochs=210 train.max_epochs_no_improvement=200 train.mixed_precision=False wandb.entity=ck-experimental wandb.project=main
```

**sMNIST CCNN-6,380**:
```
python main.py conv.bias=True conv.causal=True conv.type=SeparableFlexConv conv.use_fft=True dataset.data_type=sequence dataset.name=MNIST dataset.params.permuted=False kernel.bias=True kernel.chang_initialize=True kernel.no_hidden=64 kernel.no_layers=3 kernel.omega_0=2976.4910261630357 kernel.type=MAGNet mask.dynamic_cropping=True mask.init_value=0.075 mask.threshold=0.1 mask.type=gaussian net.block.type=S4 net.dropout=0.1 net.dropout_in=0 net.dropout_type=Dropout2d net.no_blocks=6 net.no_hidden=380 net.nonlinearity=GELU net.norm=BatchNorm net.type=ResNet optimizer.lr=0.01 optimizer.mask_lr_ratio=1 optimizer.name=AdamW optimizer.weight_decay=1e-06 scheduler.name=cosine scheduler.warmup_epochs=10 test.batch_size_multiplier=1 test.before_train=False train.batch_size=100 train.epochs=210 train.max_epochs_no_improvement=200 train.mixed_precision=False wandb.entity=ck-experimental wandb.project=main
```

**pMNIST CCNN-4,140**:
```
python main.py conv.bias=True conv.causal=True conv.type=SeparableFlexConv conv.use_fft=True dataset.data_type=sequence dataset.name=MNIST dataset.params.permuted=True kernel.bias=True kernel.chang_initialize=True kernel.no_hidden=32 kernel.no_layers=3 kernel.omega_0=2985.6332826938724 kernel.type=MAGNet mask.dynamic_cropping=True mask.init_value=0.075 mask.threshold=0.1 mask.type=gaussian net.block.type=S4 net.dropout=0 net.dropout_in=0.2 net.dropout_type=Dropout2d net.no_blocks=4 net.no_hidden=140 net.nonlinearity=GELU net.norm=BatchNorm net.type=ResNet optimizer.lr=0.02 optimizer.mask_lr_ratio=1 optimizer.name=AdamW optimizer.weight_decay=0 scheduler.name=cosine scheduler.warmup_epochs=10 test.batch_size_multiplier=1 test.before_train=False train.batch_size=100 train.epochs=210 train.max_epochs_no_improvement=200 train.mixed_precision=False wandb.entity=ck-experimental wandb.project=main
```

**pMNIST CCNN-6,380**:
```
python main.py conv.bias=True conv.causal=True conv.type=SeparableFlexConv conv.use_fft=True dataset.data_type=sequence dataset.name=MNIST dataset.params.permuted=True kernel.bias=True kernel.chang_initialize=True kernel.no_hidden=64 kernel.no_layers=3 kernel.omega_0=2985.633 kernel.type=MAGNet mask.dynamic_cropping=True mask.init_value=0.075 mask.threshold=0.1 mask.type=gaussian net.block.type=S4 net.dropout=0 net.dropout_in=0.2 net.dropout_type=Dropout2d net.no_blocks=6 net.no_hidden=380 net.nonlinearity=GELU net.norm=BatchNorm net.type=ResNet optimizer.lr=0.02 optimizer.mask_lr_ratio=1 optimizer.name=AdamW optimizer.weight_decay=0 scheduler.name=cosine scheduler.warmup_epochs=10 test.batch_size_multiplier=1 test.before_train=False train.batch_size=100 train.epochs=210 train.max_epochs_no_improvement=200 train.mixed_precision=False wandb.entity=ck-experimental wandb.project=main
```

**sCIFAR10 CCNN-4,140**:
```
python main.py net.type=ResNet net.no_hidden=140 net.no_blocks=4 net.norm=BatchNorm net.nonlinearity=GELU net.dropout_in=0.0 net.dropout=0.0 net.block.type=S4 net.block.prenorm=True net.dropout_type=Dropout2d kernel.type=MAGNet kernel.no_hidden=32 kernel.no_layers=3 kernel.omega_0=2386.49 kernel.chang_initialize=True kernel.size=same conv.type=SeparableFlexConv conv.causal=True conv.use_fft=True dataset.name=CIFAR10 dataset.data_type=sequence dataset.params.grayscale=False dataset.augment=True mask.type=gaussian mask.threshold=0.1 mask.init_value=0.075 mask.dynamic_cropping=True mask.learn_mean=False train.epochs=210 train.batch_size=50 train.max_epochs_no_improvement=200 optimizer.lr=2e-2 optimizer.name=AdamW optimizer.mask_lr_ratio=1.0 optimizer.weight_decay=0.0 scheduler.name=cosine scheduler.warmup_epochs=10 debug=False
```

**sCIFAR10 CCNN-6,380**:
```
python main.py conv.bias=True conv.causal=True conv.type=SeparableFlexConv conv.use_fft=True dataset.data_type=sequence dataset.name=CIFAR10 dataset.params.grayscale=False kernel.bias=True kernel.chang_initialize=True kernel.no_hidden=64 kernel.no_layers=3 kernel.omega_0=4005.146 kernel.type=MAGNet mask.dynamic_cropping=True mask.init_value=0.075 mask.threshold=0.1 mask.type=gaussian net.block.type=S4 net.dropout=0.25 net.dropout_type=Dropout2d net.no_blocks=6 net.no_hidden=380 net.nonlinearity=GELU net.norm=BatchNorm net.type=ResNet optimizer.lr=0.01 optimizer.mask_lr_ratio=1 optimizer.name=AdamW optimizer.weight_decay=0 scheduler.name=cosine scheduler.warmup_epochs=10 test.batch_size_multiplier=1 test.before_train=False train.batch_size=50 train.epochs=210 train.max_epochs_no_improvement=200 train.mixed_precision=False wandb.entity=ck-experimental wandb.project=main
```


## Table 2: Speech classification
**MFCC CCNN-4,140**:
```
```

**MFCC CCNN-6,380**:
```
```

**Raw CCNN-4,140**:
```

```

**Raw CCNN-6,380**:
```
python main.py conv.bias=True conv.causal=True conv.type=SeparableFlexConv conv.use_fft=True dataset.data_type=default dataset.name=SpeechCommands dataset.params.mfcc=False hooks_enabled=False kernel.bias=True kernel.chang_initialize=True kernel.no_hidden=64 kernel.no_layers=3 kernel.omega_0=1295.61 kernel.type=MAGNet mask.dynamic_cropping=True mask.init_value=0.075 mask.learn_mean=False mask.threshold=0.1 mask.type=gaussian net.block.type=S4 net.dropout=0.2 net.dropout_type=Dropout2d net.no_blocks=6 net.no_hidden=380 net.nonlinearity=GELU net.norm=BatchNorm net.type=ResNet optimizer.lr=0.02 optimizer.mask_lr_ratio=1 optimizer.name=AdamW optimizer.weight_decay=1e-06 scheduler.name=cosine scheduler.warmup_epochs=10 test.batch_size_multiplier=1 test.before_train=False train.batch_size=10 train.distributed=True train.epochs=160 train.max_epochs_no_improvement=150 train.mixed_precision=False wandb.entity=ck-experimental wandb.project=main
```

**0.5x CCNN-4,140**:
```
```

**0.5x CCNN-6,380**:
```
```

## Table 3: 2D image classfication
**CIFAR10 CCNN-4,140**:
```
python main.py train.batch_size=64 train.epochs=350 dataset.augment=True dataset.name=CIFAR10 dataset.data_type=default device=cuda net.dropout=0.1 net.dropout_in=0 net.type=ResNet net.norm=BatchNorm net.no_blocks=4 net.no_hidden=140 net.block.type=S4 net.block.prenorm=True net.nonlinearity=GELU kernel.type=MAGNet kernel.omega_0=4231.568793085858 kernel.no_hidden=32 kernel.no_layers=3 kernel.size=33 kernel.chang_initialize=True mask.dynamic_cropping=true mask.init_value=0.75 mask.learn_mean=false mask.temperature=0 mask.threshold=0.1 mask.type=gaussian conv.type=SeparableFlexConv conv.use_fft=False scheduler.name=cosine scheduler.warmup_epochs=5 optimizer.name=AdamW optimizer.mask_lr_ratio=0.1 optimizer.weight_decay=0.0001 optimizer.lr=0.02 seed=0 debug=False
```

**CIFAR10 CCNN-6,380**:
```
python main.py conv.bias=True conv.causal=False conv.type=SeparableFlexConv conv.use_fft=False dataset.augment=True dataset.data_type=default dataset.name=CIFAR10 dataset.params.grayscale=False kernel.bias=True kernel.chang_initialize=True kernel.no_hidden=64 kernel.no_layers=3 kernel.omega_0=976.781 kernel.size=33 kernel.type=MAGNet mask.dynamic_cropping=True mask.init_value=0.075 mask.threshold=0.1 mask.type=gaussian net.block.type=S4 net.dropout=0.15 net.dropout_type=Dropout2d net.no_blocks=6 net.no_hidden=380 net.nonlinearity=GELU net.norm=BatchNorm net.type=ResNet optimizer.lr=0.01 optimizer.mask_lr_ratio=0.1 optimizer.name=AdamW optimizer.weight_decay=0 scheduler.name=cosine scheduler.warmup_epochs=10 test.batch_size_multiplier=1 test.before_train=False train.batch_size=50 train.epochs=210 train.max_epochs_no_improvement=200 train.mixed_precision=False wandb.entity=ck-experimental wandb.project=main
```

**CIFAR100 CCNN-4,140**:
```
python main.py conv.bias=True conv.causal=False conv.type=SeparableFlexConv conv.use_fft=False dataset.data_type=default dataset.name=CIFAR100 kernel.bias=True kernel.chang_initialize=True kernel.no_hidden=32 kernel.no_layers=3 kernel.omega_0=3521.5468260391735 kernel.size=33 kernel.type=MAGNet mask.dynamic_cropping=True mask.init_value=0.075 mask.threshold=0.1 mask.type=gaussian net.block.type=S4 net.dropout=0.1 net.dropout_type=Dropout2d net.no_blocks=4 net.no_hidden=140 net.nonlinearity=GELU net.norm=BatchNorm net.type=ResNet optimizer.lr=0.02 optimizer.mask_lr_ratio=1 optimizer.name=AdamW optimizer.weight_decay=0.0001 scheduler.name=cosine scheduler.warmup_epochs=10 test.batch_size_multiplier=1 test.before_train=False train.batch_size=50 train.epochs=210 train.max_epochs_no_improvement=200 train.mixed_precision=False wandb.entity=ck-experimental wandb.project=main
```

**CIFAR100 CCNN-6,380**:
```
python main.py conv.bias=True conv.causal=False conv.type=SeparableFlexConv conv.use_fft=False dataset.data_type=default dataset.name=CIFAR100 dataset.params.grayscale=False kernel.bias=True kernel.chang_initialize=True kernel.no_hidden=64 kernel.no_layers=3 kernel.size=31 kernel.omega_0=679.142 kernel.type=MAGNet mask.dynamic_cropping=True mask.init_value=0.075 mask.threshold=0.1 mask.type=gaussian net.block.type=S4 net.dropout=0.2 net.dropout_type=Dropout2d net.no_blocks=6 net.no_hidden=380 net.nonlinearity=GELU net.norm=BatchNorm net.type=ResNet optimizer.lr=0.02 optimizer.mask_lr_ratio=1 optimizer.name=AdamW optimizer.weight_decay=0 scheduler.name=cosine scheduler.warmup_epochs=10 test.batch_size_multiplier=1 test.before_train=False train.batch_size=50 train.epochs=210 train.max_epochs_no_improvement=200 train.mixed_precision=False wandb.entity=ck-experimental wandb.project=main
```

**STL10 CCNN-4,140**:
```
python main.py conv.bias=True conv.causal=False conv.type=SeparableFlexConv conv.use_fft=False dataset.augment=True dataset.data_type=default dataset.name=STL10 kernel.bias=True kernel.chang_initialize=True kernel.no_hidden=32 kernel.no_layers=3 kernel.omega_0=954.281 kernel.size=33 kernel.type=MAGNet mask.dynamic_cropping=True mask.init_value=0.075 mask.threshold=0.1 mask.type=gaussian net.block.type=S4 net.dropout=0.1 net.dropout_type=Dropout net.no_blocks=4 net.no_hidden=140 net.nonlinearity=GELU net.norm=BatchNorm net.type=ResNet optimizer.lr=0.02 optimizer.mask_lr_ratio=1 optimizer.name=AdamW optimizer.weight_decay=0 scheduler.name=cosine scheduler.warmup_epochs=10 test.batch_size_multiplier=1 test.before_train=False train.accumulate_grad_steps=4 train.batch_size=64 train.epochs=210 train.max_epochs_no_improvement=200 train.mixed_precision=False wandb.entity=ck-experimental wandb.project=main
```

**STL10 CCNN-6,380**:
```
python main.py conv.bias=True conv.causal=False conv.type=SeparableFlexConv conv.use_fft=False dataset.augment=True dataset.data_type=default dataset.name=STL10 kernel.bias=True kernel.chang_initialize=True kernel.no_hidden=64 kernel.no_layers=3 kernel.omega_0=954.281 kernel.size=31 kernel.type=MAGNet mask.dynamic_cropping=True mask.init_value=0.075 mask.threshold=0.1 mask.type=gaussian net.block.type=S4 net.dropout=0.1 net.dropout_type=Dropout net.no_blocks=6 net.no_hidden=380 net.nonlinearity=GELU net.norm=BatchNorm net.type=ResNet optimizer.lr=0.01 optimizer.mask_lr_ratio=1 optimizer.name=AdamW optimizer.weight_decay=0 scheduler.name=cosine scheduler.warmup_epochs=10 test.batch_size_multiplier=1 test.before_train=False train.accumulate_grad_steps=2 train.batch_size=48 train.epochs=210 train.max_epochs_no_improvement=200 train.mixed_precision=False wandb.entity=ck-experimental wandb.project=main
```

**2DImage CCNN-4,140**:
```
python main.py conv.bias=True conv.causal=False conv.type=SeparableFlexConv conv.use_fft=False dataset.data_type=default dataset.name=CIFAR10 dataset.params.grayscale=True kernel.bias=True kernel.chang_initialize=True kernel.no_hidden=32 kernel.no_layers=3 kernel.omega_0=2085.433586112234 kernel.size=33 kernel.type=MAGNet mask.dynamic_cropping=True mask.init_value=0.075 mask.threshold=0.1 mask.type=gaussian net.block.type=S4 net.dropout=0.2 net.dropout_type=Dropout2d net.no_blocks=4 net.no_hidden=140 net.nonlinearity=GELU net.norm=BatchNorm net.type=ResNet optimizer.lr=0.02 optimizer.mask_lr_ratio=1 optimizer.name=AdamW optimizer.weight_decay=1e-06 scheduler.name=cosine scheduler.warmup_epochs=10 test.batch_size_multiplier=1 test.before_train=False train.batch_size=50 train.epochs=210 train.max_epochs_no_improvement=200 train.mixed_precision=False wandb.entity=ck-experimental wandb.project=main
```

**2DImage CCNN-6,380**:
```
python main.py conv.bias=True conv.causal=False conv.type=SeparableFlexConv conv.use_fft=False dataset.data_type=default dataset.name=CIFAR10 dataset.params.grayscale=True kernel.bias=True kernel.chang_initialize=True kernel.no_hidden=64 kernel.no_layers=3 kernel.omega_0=2306.081 kernel.size=33 kernel.type=MAGNet mask.dynamic_cropping=True mask.init_value=0.075 mask.threshold=0.1 mask.type=gaussian net.block.type=S4 net.dropout=0.2 net.dropout_type=Dropout2d net.no_blocks=6 net.no_hidden=380 net.nonlinearity=GELU net.norm=BatchNorm net.type=ResNet optimizer.lr=0.02 optimizer.mask_lr_ratio=1 optimizer.name=AdamW optimizer.weight_decay=0 scheduler.name=cosine scheduler.warmup_epochs=10 test.batch_size_multiplier=1 test.before_train=False train.batch_size=50 train.epochs=210 train.max_epochs_no_improvement=200 train.mixed_precision=False wandb.entity=ck-experimental wandb.project=main
```

**2DPathfinder CCNN-4,140**:
```
python main.py conv.bias=True conv.causal=False conv.type=SeparableFlexConv conv.use_fft=False dataset.data_type=default dataset.name=PathFinder dataset.params.resolution=32 kernel.bias=True kernel.chang_initialize=True kernel.no_hidden=32 kernel.no_layers=3 kernel.omega_0=1239.138351528754 kernel.size=33 kernel.type=MAGNet mask.dynamic_cropping=True mask.init_value=0.225 mask.threshold=0.1 mask.type=gaussian net.block.type=S4 net.dropout=0.1 net.dropout_type=Dropout2d net.no_blocks=4 net.no_hidden=140 net.nonlinearity=GELU net.norm=BatchNorm net.type=ResNet optimizer.lr=0.01 optimizer.mask_lr_ratio=1 optimizer.name=AdamW optimizer.weight_decay=0 scheduler.name=cosine scheduler.warmup_epochs=10 test.batch_size_multiplier=1 test.before_train=False train.batch_size=100 train.epochs=210 train.max_epochs_no_improvement=200 train.mixed_precision=False wandb.entity=ck-experimental wandb.project=main
```

**2DPathfinder CCNN-6,380**:
```
python main.py conv.bias=True conv.causal=False conv.type=SeparableFlexConv conv.use_fft=False dataset.data_type=default dataset.name=PathFinder dataset.params.resolution=32 kernel.bias=True kernel.chang_initialize=True kernel.no_hidden=64 kernel.no_layers=3 kernel.omega_0=3908.323 kernel.size=33 kernel.type=MAGNet mask.dynamic_cropping=True mask.init_value=0.225 mask.threshold=0.1 mask.type=gaussian net.block.type=S4 net.dropout=0.2 net.dropout_type=Dropout2d net.no_blocks=6 net.no_hidden=380 net.nonlinearity=GELU net.norm=BatchNorm net.type=ResNet optimizer.lr=0.01 optimizer.mask_lr_ratio=1 optimizer.name=AdamW optimizer.weight_decay=0 scheduler.name=cosine scheduler.warmup_epochs=10 test.batch_size_multiplier=1 test.before_train=False train.batch_size=50 train.distributed=True train.epochs=210 train.max_epochs_no_improvement=200 train.mixed_precision=False wandb.entity=ck-experimental wandb.project=main
```

## Table 4: Long Range Arena

**ListOps CCNN-4,140**:
```
python main.py conv.bias=True conv.causal=True conv.type=SeparableFlexConv conv.use_fft=True dataset.data_type=default dataset.name=ListOps hooks_enabled=False kernel.bias=True kernel.chang_initialize=True kernel.no_hidden=32 kernel.no_layers=3 kernel.omega_0=784.655 kernel.type=MAGNet mask.dynamic_cropping=True mask.init_value=0.03 mask.learn_mean=False mask.threshold=0.1 mask.type=gaussian net.block.type=S4 net.dropout=0.1 net.dropout_type=Dropout net.no_blocks=4 net.no_hidden=140 net.nonlinearity=GELU net.norm=BatchNorm net.type=ResNet optimizer.lr=0.001 optimizer.mask_lr_ratio=0.1 optimizer.name=AdamW optimizer.weight_decay=1e-06 scheduler.name=cosine scheduler.warmup_epochs=10 test.batch_size_multiplier=1 test.before_train=False train.batch_size=50 train.epochs=60 train.max_epochs_no_improvement=150 train.mixed_precision=False wandb.entity=ck-experimental wandb.project=main
```

**ListOps CCNN-6,380**:
```
python main.py conv.bias=True conv.causal=True conv.type=SeparableFlexConv conv.use_fft=True dataset.data_type=default dataset.name=ListOps hooks_enabled=False kernel.bias=True kernel.chang_initialize=True kernel.no_hidden=32 kernel.no_layers=3 kernel.omega_0=784.655 kernel.type=MAGNet mask.dynamic_cropping=True mask.init_value=0.03 mask.learn_mean=False mask.threshold=0.1 mask.type=gaussian net.block.type=S4 net.dropout=0.25 net.dropout_type=Dropout net.no_blocks=6 net.no_hidden=380 net.nonlinearity=GELU net.norm=BatchNorm net.type=ResNet optimizer.lr=0.001 optimizer.mask_lr_ratio=0.1 optimizer.name=AdamW optimizer.weight_decay=0 scheduler.name=cosine scheduler.warmup_epochs=10 test.batch_size_multiplier=1 test.before_train=False train.batch_size=50 train.epochs=60 train.max_epochs_no_improvement=150 train.mixed_precision=False wandb.entity=ck-experimental wandb.project=main
```

**Text CCNN-4,140**:
```

```

**Text CCNN-6,380**:
```
```

**Image CCNN-4,140**:
```
python main.py conv.bias=True conv.causal=True conv.type=SeparableFlexConv conv.use_fft=True dataset.data_type=sequence dataset.name=CIFAR10 dataset.params.grayscale=True kernel.bias=True kernel.chang_initialize=True kernel.no_hidden=32 kernel.no_layers=3 kernel.omega_0=4005.146 kernel.type=MAGNet mask.dynamic_cropping=True mask.init_value=0.075 mask.threshold=0.1 mask.type=gaussian net.block.type=S4 net.dropout=0.1 net.dropout_type=Dropout2d net.no_blocks=4 net.no_hidden=140 net.nonlinearity=GELU net.norm=BatchNorm net.type=ResNet optimizer.lr=0.01 optimizer.mask_lr_ratio=1 optimizer.name=AdamW optimizer.weight_decay=0 scheduler.name=cosine scheduler.warmup_epochs=10 test.batch_size_multiplier=1 test.before_train=False train.batch_size=50 train.epochs=210 train.max_epochs_no_improvement=200 train.mixed_precision=False wandb.entity=ck-experimental wandb.project=main
```

**Image CCNN-6,380**:
```
python main.py conv.bias=True conv.causal=True conv.type=SeparableFlexConv conv.use_fft=True dataset.data_type=sequence dataset.name=CIFAR10 dataset.params.grayscale=True kernel.bias=True kernel.chang_initialize=True kernel.no_hidden=64 kernel.no_layers=3 kernel.omega_0=4005.146 kernel.type=MAGNet mask.dynamic_cropping=True mask.init_value=0.075 mask.threshold=0.1 mask.type=gaussian net.block.type=S4 net.dropout=0.1 net.dropout_type=Dropout2d net.no_blocks=6 net.no_hidden=380 net.nonlinearity=GELU net.norm=BatchNorm net.type=ResNet optimizer.lr=0.01 optimizer.mask_lr_ratio=1 optimizer.name=AdamW optimizer.weight_decay=0 scheduler.name=cosine scheduler.warmup_epochs=10 test.batch_size_multiplier=1 test.before_train=False train.batch_size=50 train.epochs=210 train.max_epochs_no_improvement=200 train.mixed_precision=False wandb.entity=ck-experimental wandb.project=main
```

**Pathfinder CCNN-4,140**:
```
python main.py conv.bias=True conv.causal=True conv.type=SeparableFlexConv conv.use_fft=True dataset.data_type=sequence dataset.name=PathFinder dataset.params.resolution=32 kernel.bias=True kernel.chang_initialize=True kernel.no_hidden=32 kernel.no_layers=3 kernel.omega_0=2272.557 kernel.type=MAGNet mask.dynamic_cropping=True mask.init_value=0.4 mask.threshold=0.1 mask.type=gaussian net.block.type=S4 net.dropout=0.2 net.dropout_type=Dropout2d net.no_blocks=4 net.no_hidden=140 net.nonlinearity=GELU net.norm=BatchNorm net.type=ResNet optimizer.lr=0.01 optimizer.mask_lr_ratio=1 optimizer.name=AdamW optimizer.weight_decay=0 scheduler.name=cosine scheduler.warmup_epochs=10 test.batch_size_multiplier=1 test.before_train=False train.batch_size=100 train.epochs=210 train.max_epochs_no_improvement=200 train.mixed_precision=False wandb.entity=ck-experimental wandb.project=main
```

**Pathfinder CCNN-6,380**:
```
python main.py conv.bias=True conv.causal=True conv.type=SeparableFlexConv conv.use_fft=True dataset.data_type=sequence dataset.name=PathFinder dataset.params.resolution=32 kernel.bias=True kernel.chang_initialize=True kernel.no_hidden=64 kernel.no_layers=3 kernel.omega_0=2272.557 kernel.type=MAGNet mask.dynamic_cropping=True mask.init_value=0.4 mask.threshold=0.1 mask.type=gaussian net.block.type=S4 net.dropout=0.1 net.dropout_type=Dropout2d net.no_blocks=6 net.no_hidden=380 net.nonlinearity=GELU net.norm=BatchNorm net.type=ResNet optimizer.lr=0.01 optimizer.mask_lr_ratio=1 optimizer.name=AdamW optimizer.weight_decay=1e-06 scheduler.name=cosine scheduler.warmup_epochs=10 test.batch_size_multiplier=1 test.before_train=False train.batch_size=100 train.epochs=210 train.max_epochs_no_improvement=200 train.mixed_precision=False wandb.entity=ck-experimental wandb.project=main
```
