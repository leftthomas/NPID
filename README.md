# ShadowMode
A PyTorch implementation of ShadowMode based on the paper [ShadowMode: Predict Unknown]().

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

## Dataset
`ILSVRC2012` dataset is used in this repo, the dataset should be downloaded from [ImageNet](http://image-net.org/challenges/LSVRC/2012/)
and extracted as following:
```
{data_root}/
  train/
    n01693334/
      # JPEG files 
    n02018207/
    ......
    n04286575/
    n04596742/
  val/
    n01693334/
      # JPEG files 
    n02018207/
    ......
    n04286575/
    n04596742/
```

## Usage
### Train Features Extractor
```
python train.py --epochs 50 --load_ids
optional arguments:
--data_path                   Path to dataset [default value is '/home/data/imagenet/ILSVRC2012']
--model_type                  Backbone type [default value is 'resnet18'] (choices=['resnet18', 'resnet34', 'resnet50', 'resnext50_32x4d'])
--share_type                  Shared module type [default value is 'layer1'] (choices=['none', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4'])
--batch_size                  Number of images in each mini-batch [default value is 256]
--epochs                      Number of sweeps over the dataset to train [default value is 100]
--ensemble_size               Ensemble model size [default value is 48]
--meta_class_size             Meta class size [default value is 12]
--gpu_ids                     Selected gpu [default value is '0,1,2,3,4,5,6,7']
--load_ids                    Load already generated ids or not [default value is False]
```

### Train Model
```
python test.py --epochs 50 --dictionary_size 4096
optional arguments:
--data_path                   Path to dataset [default value is '/home/data/imagenet/ILSVRC2012']
--batch_size                  Number of images in each mini-batch [default value is 256]
--epochs                      Number of sweeps over the dataset to train [default value is 100]
--data_path                   Features extractor file [default value is 'epochs/imagenet_resnet18_layer1_48_12_features_extractor.pth']
--gpu_ids                     Selected gpu [default value is '0,1,2,3,4,5,6,7']
```

## Results

<table>
	<tbody>
		<!-- START TABLE -->
		<!-- TABLE HEADER -->
		<th>Name</th>
		<th>train time (s/iter)</th>
		<th>train mem (GB)</th>
		<th>Top1 Acc %</th>
		<th>Top5 Acc %</th>
		<th>download link</th>
		<!-- TABLE BODY -->
		<!-- ROW: r18 -->
		<tr>
			<td align="center">ResNet18</td>
			<td align="center">80.49</td>
			<td align="center">53.92</td>
			<td align="center">42.71</td>
			<td align="center">68.69</td>
			<td align="center"><a href="https://pan.baidu.com/s/1jP7zWezVPBZWx_9LjJCgWg">model</a>&nbsp;|&nbsp;xxi8</td>
		</tr>
		<!-- ROW: r50 -->
		<tr>
			<td align="center">ResNet50</td>
			<td align="center">81.16</td>
			<td align="center">54.54</td>
			<td align="center">43.61</td>
			<td align="center">69.50</td>
			<td align="center"><a href="https://pan.baidu.com/s/1BeGS7gckGAczd1euB55EFA">model</a>&nbsp;|&nbsp;1jhd</td>
		</tr>
	</tbody>
</table>

