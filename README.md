# NPID
A PyTorch implementation of NPID based on the paper [Unsupervised Feature Learning via Non-Parametric Instance Discrimination](https://arxiv.org/abs/1805.01978).

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

## Dataset
`CIFAR10` dataset is used in this repo, the dataset will be downloaded by `PyTorch` automatically.

## Usage
```
python train.py --epochs 50 --feature_dim 256
optional arguments:
--feature_dim                 Feature dim for each image [default value is 128]
--m                           Negative sample number [default value is 4096]
--temperature                 Temperature used in softmax [default value is 0.1]
--momentum                    Momentum used for the update of memory bank [default value is 0.5]
--k                           Top k most similar images used to predict the label [default value is 200]
--batch_size                  Number of images in each mini-batch [default value is 128]
--epochs                      Number of sweeps over the dataset to train [default value is 200]
--gpu_ids                     Selected gpu [default value is '0,1,2,3,4,5,6,7']
```

## Results

<table>
	<tbody>
		<!-- START TABLE -->
		<!-- TABLE HEADER -->
		<th>Backbone</th>
		<th>feature dim</th>
		<th>batch size</th>
		<th>epoch num</th>
		<th>temperature</th>
		<th>momentum</th>
		<th>k</th>
		<th>Top1 Acc %</th>
		<th>Top5 Acc %</th>
		<th>download link</th>
		<!-- TABLE BODY -->
		<!-- ROW: r18 -->
		<tr>
			<td align="center">ResNet18</td>
			<td align="center">128</td>
			<td align="center">128</td>
			<td align="center">200</td>
			<td align="center">0.1</td>
			<td align="center">0.5</td>
			<td align="center">200</td>
			<td align="center">42.71</td>
			<td align="center">68.69</td>
			<td align="center"><a href="https://pan.baidu.com/s/1jP7zWezVPBZWx_9LjJCgWg">model</a>&nbsp;|&nbsp;xxi8</td>
		</tr>
	</tbody>
</table>

