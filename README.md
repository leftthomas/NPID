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
### Train Features Extractor
```
python train.py --epochs 50 --dictionary_size 4096
optional arguments:
--data_path                   Path to dataset [default value is '/home/data/imagenet/ILSVRC2012']
--batch_size                  Number of images in each mini-batch [default value is 256]
--epochs                      Number of sweeps over the dataset to train [default value is 200]
--features_dim                Dim of features for each image [default value is 128]
--dictionary_size             Size of dictionary [default value is 65536]
```

### Train Model
```
python test.py --epochs 100 --features_dim 256
optional arguments:
--data_path                   Path to dataset [default value is '/home/data/imagenet/ILSVRC2012']
--batch_size                  Number of images in each mini-batch [default value is 256]
--epochs                      Number of sweeps over the dataset to train [default value is 100]
--features_dim                Dim of features for each image [default value is 128]
--model                       Features extractor file [default value is 'epochs/features_extractor.pth']
```
