# ShadowMode
A PyTorch implementation of ShadowMode based on the paper [Predict unknown with ShadowMode]().

<div align="center">
  <img src="results/architecture.png"/>
</div>

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

## Dataset
`Uisee` dataset is used in this repo.

## Usage
### Train Model
```
python train.py --num_epochs 50 --load_ids
optional arguments:
--data_path                   path to dataset [default value is '/home/data/uisee/shadow_mode']
--with_random                 with branch random weight or not [default value is False]
--load_ids                    load already generated ids or not [default value is False]
--batch_size                  train batch size [default value is 32]
--num_epochs                  train epochs number [default value is 40]
--ensemble_size               ensemble model size [default value is 12]
--meta_class_size             meta class size [default value is 6]
--gpu_ids                     selected gpu [default value is '0,1']
```
