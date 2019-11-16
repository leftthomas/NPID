# REIR
A PyTorch implementation of REIR based on the paper [Randomized Ensembles for Image Retrieval]().

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

## Datasets
[CARS196](http://ai.stanford.edu/~jkrause/cars/car_dataset.html), [CUB200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), 
[Standard Online Products](http://cvgl.stanford.edu/projects/lifted_struct/) and 
[In-shop Clothes](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html) are used in this repo.

You should download these datasets by yourself, and extract them into `data` directory, make sure the dir names are 
`car`, `cub`, `sop` and `isc`. Then run `data_utils.py` to preprocess them.

## Usage
### Train Model
```
python train.py --data_name cub --crop_type cropped --model_type resnet34 --num_epochs 50
optional arguments:
--data_name                   dataset name [default value is 'car'](choices=['car', 'cub', 'sop', 'isc'])
--crop_type                   crop data or not, it only works for car or cub dataset [default value is 'uncropped'](choices=['uncropped', 'cropped'])
--label_type                  assign label with random method or fixed method [default value is 'fixed'](choices=['fixed', 'random'])
--recalls                     selected recall [default value is '1,2,4,8']
--model_type                  backbone type [default value is 'resnet18'](choices=['resnet18', 'resnet34', 'resnet50', 'resnext50_32x4d'])
--share_type                  shared module type [default value is 'layer1'](choices=['none', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4'])
--with_random                 with branch random weight or not [default value is False]
--load_ids                    load already generated ids or not [default value is False]
--batch_size                  train batch size [default value is 8]
--num_epochs                  train epochs number [default value is 20]
--ensemble_size               ensemble model size [default value is 48]
--meta_class_size             meta class size [default value is 12]
--gpu_ids                     selected gpu [default value is '0,1']
```

### Inference Demo
```
python inference.py --retrieval_num 10 --data_type train
optional arguments:
--query_img_name              query image name [default value is 'data/car/uncropped/008055.jpg']
--data_base                   queried database [default value is 'car_uncropped_fixed_random_resnet18_48_12_data_base.pth']
--data_type                   retrieval database type [default value is 'test'](choices=['train', 'test'])
--retrieval_num               retrieval number [default value is 8]
```

### Ablation Study
```
python ablation.py --save_results
optional arguments:
--better_data_base            better database [default value is 'car_uncropped_fixed_unrandom_layer1_resnet18_48_12_data_base.pth']
--worse_data_base             worse database [default value is 'car_uncropped_random_unrandom_layer1_resnet18_48_12_data_base.pth']
--data_type                   retrieval database type [default value is 'test'](choices=['train', 'test'])
--retrieval_num               retrieval number [default value is 8]
--save_results                with save results or not [default value is False]
```

## Benchmarks
Adam optimizer is used with learning rate scheduling. The models are trained with batch size `8` on two 
NVIDIA Tesla V100 (32G) GPUs.

The images are preprocessed with resize (256, 256), random horizontal flip and normalize. 

For `CARS196` and `CUB200` datasets, ensemble size `48`, meta class size `12` and `20` epochs are used. 

For `SOP` and `In-shop` datasets, ensemble size `24`, meta class size `192` and `30` epochs are used.

Here is the model parameter details:
<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>resnet18</th>
      <th>resnet34</th>
      <th>resnet50</th>
      <th>resnext50_32x4d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">CARS196</td>
      <td align="center">529,365,376</td>
      <td align="center">1,011,079,808</td>
      <td align="center">1,118,974,592</td>
      <td align="center">1,094,093,696</td>
    </tr>
    <tr>
      <td align="center">CUB200</td>
      <td align="center">529,365,376</td>
      <td align="center">1,011,079,808</td>
      <td align="center">1,118,974,592</td>
      <td align="center">1,094,093,696</td>
    </tr>
    <tr>
      <td align="center">SOP</td>
      <td align="center">541,381,888</td>
      <td align="center">1,023,096,320</td>
      <td align="center">1,166,970,368</td>
      <td align="center">1,142,089,472</td>
    </tr>
    <tr>
      <td align="center">In-shop</td>
      <td align="center">533,797,696</td>
      <td align="center">1,015,512,128</td>
      <td align="center">1,136,677,952</td>
      <td align="center">1,111,797,056</td>
    </tr>
  </tbody>
</table>

Here is the results of uncropped `CARS196` dataset:

<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>resnet18</th>
      <th>resnet34</th>
      <th>resnet50</th>
      <th>resnext50_32x4d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">R@1</td>
      <td align="center">94.47%</td>
      <td align="center">94.23%</td>
      <td align="center">94.32%</td>
      <td align="center">95.01%</td>
    </tr>
    <tr>
      <td align="center">R@2</td>
      <td align="center">97.21%</td>
      <td align="center">96.75%</td>
      <td align="center">96.94%</td>
      <td align="center">97.26%</td>
    </tr>
    <tr>
      <td align="center">R@4</td>
      <td align="center">98.60%</td>
      <td align="center">98.20%</td>
      <td align="center">98.30%</td>
      <td align="center">98.20%</td>
    </tr>
    <tr>
      <td align="center">R@8</td>
      <td align="center">99.37%</td>
      <td align="center">98.94%</td>
      <td align="center">98.97%</td>
      <td align="center">98.94%</td>
    </tr>
  </tbody>
</table>

Here is the results of cropped `CARS196` dataset:

<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>resnet18</th>
      <th>resnet34</th>
      <th>resnet50</th>
      <th>resnext50_32x4d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">R@1</td>
      <td align="center">96.78%</td>
      <td align="center">96.59%</td>
      <td align="center">96.56%</td>
      <td align="center">96.81%</td>
    </tr>
    <tr>
      <td align="center">R@2</td>
      <td align="center">98.38%</td>
      <td align="center">98.16%</td>
      <td align="center">98.09%</td>
      <td align="center">98.18%</td>
    </tr>
    </tr>
    <tr>
      <td align="center">R@4</td>
      <td align="center">99.08%</td>
      <td align="center">98.89%</td>
      <td align="center">98.73%</td>
      <td align="center">98.88%</td>
    </tr>
    <tr>
      <td align="center">R@8</td>
      <td align="center">99.47%</td>
      <td align="center">99.27%</td>
      <td align="center">99.26%</td>
      <td align="center">99.41%</td>
  </tbody>
</table>

Here is the results of uncropped `CUB200` dataset:

<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>resnet18</th>
      <th>resnet34</th>
      <th>resnet50</th>
      <th>resnext50_32x4d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">R@1</td>
      <td align="center">81.99%</td>
      <td align="center">73.68%</td>
      <td align="center">75.25%</td>
      <td align="center">78.39%</td>
    </tr>
    <tr>
      <td align="center">R@2</td>
      <td align="center">88.49%</td>
      <td align="center">81.55%</td>
      <td align="center">83.52%</td>
      <td align="center">85.35%</td>
    </tr>
    <tr>
      <td align="center">R@4</td>
      <td align="center">92.52%</td>
      <td align="center">87.67%</td>
      <td align="center">89.26%</td>
      <td align="center">90.11%</td>
    </tr>
    <tr>
      <td align="center">R@8</td>
      <td align="center">95.59%</td>
      <td align="center">92.02%</td>
      <td align="center">93.21%</td>
      <td align="center">93.75%</td>
    </tr>
  </tbody>
</table>

Here is the results of cropped `CUB200` dataset:

<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>resnet18</th>
      <th>resnet34</th>
      <th>resnet50</th>
      <th>resnext50_32x4d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">R@1</td>
      <td align="center">88.99%</td>
      <td align="center">86.24%</td>
      <td align="center">85.48%</td>
      <td align="center">86.36%</td>
    </tr>
    <tr>
      <td align="center">R@2</td>
      <td align="center">92.94%</td>
      <td align="center">91.51%</td>
      <td align="center">91.34%</td>
      <td align="center">91.32%</td>
    </tr>
    <tr>
      <td align="center">R@4</td>
      <td align="center">95.53%</td>
      <td align="center">94.43%</td>
      <td align="center">94.70%</td>
      <td align="center">94.75%</td>
    </tr>
    <tr>
      <td align="center">R@8</td>
      <td align="center">97.65%</td>
      <td align="center">96.44%</td>
      <td align="center">96.78%</td>
      <td align="center">96.47%</td>
    </tr>
  </tbody>
</table>

Here is the results of `SOP` dataset:

<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>resnet18</th>
      <th>resnet34</th>
      <th>resnet50</th>
      <th>resnext50_32x4d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">R@1</td>
      <td align="center">90.69%</td>
      <td align="center">90.86%</td>
      <td align="center">83.27%</td>
      <td align="center">85.82%</td>
    </tr>
    <tr>
      <td align="center">R@10</td>
      <td align="center">96.07%</td>
      <td align="center">95.91%</td>
      <td align="center">89.00%</td>
      <td align="center">91.31%</td>
    </tr>
    <tr>
      <td align="center">R@100</td>
      <td align="center">98.11%</td>
      <td align="center">98.01%</td>
      <td align="center">92.70%</td>
      <td align="center">94.59%</td>
    </tr>
    <tr>
      <td align="center">R@1000</td>
      <td align="center">99.29%</td>
      <td align="center">99.22%</td>
      <td align="center">96.27%</td>
      <td align="center">97.21%</td>
    </tr>
  </tbody>
</table>

Here is the results of `In-shop` dataset:

<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>resnet18</th>
      <th>resnet34</th>
      <th>resnet50</th>
      <th>resnext50_32x4d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">R@1</td>
      <td align="center">77.11%</td>
      <td align="center">83.38%</td>
      <td align="center">77.20%</td>
      <td align="center">88.02%</td>
    </tr>
    <tr>
      <td align="center">R@10</td>
      <td align="center">92.91%</td>
      <td align="center">95.33%</td>
      <td align="center">92.81%</td>
      <td align="center">96.83%</td>
    </tr>
    <tr>
      <td align="center">R@20</td>
      <td align="center">94.86%</td>
      <td align="center">96.82%</td>
      <td align="center">94.90%</td>
      <td align="center">97.74%</td>
    </tr>
    <tr>
      <td align="center">R@30</td>
      <td align="center">95.80%</td>
      <td align="center">97.38%</td>
      <td align="center">95.87%</td>
      <td align="center">98.22%</td>
    </tr>
    <tr>
      <td align="center">R@40</td>
      <td align="center">96.40%</td>
      <td align="center">97.77%</td>
      <td align="center">96.48%</td>
      <td align="center">98.57%</td>
    </tr>
    <tr>
      <td align="center">R@50</td>
      <td align="center">96.80%</td>
      <td align="center">98.08%</td>
      <td align="center">97.00%</td>
      <td align="center">98.76%</td>
    </tr>
  </tbody>
</table>

