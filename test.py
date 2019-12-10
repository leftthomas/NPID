import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
from torchvision import datasets

import utils
from model import Net


def show(mnist, targets, ret):
    target_ids = range(len(set(targets)))

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'violet', 'orange', 'purple']

    plt.figure(figsize=(12, 10))

    ax = plt.subplot(aspect='equal')
    for label in set(targets):
        idx = np.where(np.array(targets) == label)[0]
        plt.scatter(ret[idx, 0], ret[idx, 1], c=colors[label], label=label)

    for i in range(0, len(targets), 250):
        img = (mnist[i][0] * 0.3081 + 0.1307).numpy()[0]
        img = OffsetImage(img, cmap=plt.cm.gray_r, zoom=0.5)
        ax.add_artist(AnnotationBbox(img, ret[i]))

    plt.legend()
    plt.savefig('results/tsne.png')


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

    ret = TSNE(n_components=2, random_state=0).fit_transform(data)

    show(mnist, targets, ret)