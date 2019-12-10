import argparse

import torch
import tqdm
from torchvision import datasets

import utils
from model import Net

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test MoCo')
    parser.add_argument('--model', '-m', default='epochs/model.pth', help='Model file')
    args = parser.parse_args()
    model_path = args.model
    transform = utils.test_transform

    mnist = datasets.MNIST('./', train=False, download=True, transform=transform)

    model = Net()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model = model.to('cuda')

    data = []
    targets = []
    for m in tqdm.tqdm(mnist):
        target = m[1]
        targets.append(target)
        x = m[0]
        x = x.view(1, *x.shape)
        feat = model(x.to('cuda'))
        data.append(feat.cpu().data.numpy()[0])


