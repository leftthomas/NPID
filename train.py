import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

import utils
from model import Net


def initialize_queue(model_k, train_loader):
    queue = torch.zeros((0, features_dim), dtype=torch.float).to('cuda')

    for batch_idx, (data, target) in enumerate(train_loader):
        x_k = data[1].to('cuda')
        k = model_k(x_k).detach()
        queue = utils.queue_data(queue, k)
        queue = utils.dequeue_data(queue, k=dictionary_size)
        break
    return queue


def train(model_q, model_k, train_loader, queue, optimizer, epoch, temp=0.07):
    model_q.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        x_q = data[0]
        x_k = data[1]

        x_q, x_k = x_q.to('cuda'), x_k.to('cuda')
        q = model_q(x_q)
        k = model_k(x_k)
        k = k.detach()

        N = data[0].shape[0]
        K = queue.shape[0]
        l_pos = torch.bmm(q.view(N, 1, -1), k.view(N, -1, 1))
        l_neg = torch.mm(q.view(N, -1), queue.T.view(-1, K))

        logits = torch.cat([l_pos.view(N, 1), l_neg], dim=1)

        labels = torch.zeros(N, dtype=torch.long)
        labels = labels.to('cuda')

        cross_entropy_loss = nn.CrossEntropyLoss()
        loss = cross_entropy_loss(logits / temp, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        utils.momentum_update(model_q, model_k)

        queue = utils.queue_data(queue, k)
        queue = utils.dequeue_data(queue)

    total_loss /= len(train_loader.dataset)

    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, total_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MoCo')
    parser.add_argument('--data_path', default='/home/data/imagenet/ILSVRC2012', type=str, help='path to dataset')
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', '-e', type=int, default=50, help='Number of sweeps over the dataset to train')
    parser.add_argument('--features_dim', '-f', type=int, default=128, help='Dim of features for each image')
    parser.add_argument('--dictionary_size', '-d', type=int, default=32, help='Size of dictionary')
    args = parser.parse_args()

    batch_size, epochs, features_dim, data_path = args.batch_size, args.epochs, args.features_dim, args.data_path
    dictionary_size = args.dictionary_size
    train_data = datasets.ImageFolder(root='{}/{}'.format(data_path, 'train'), transform=utils.train_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model_q, model_k = Net(features_dim).to('cuda'), Net(features_dim).to('cuda')
    optimizer = optim.Adam(model_q.parameters(), lr=0.001, weight_decay=0.0001)

    queue = initialize_queue(model_k, train_loader)

    for epoch in range(1, epochs + 1):
        train(model_q, model_k, train_loader, queue, optimizer, epoch)
        torch.save(model_q.state_dict(), os.path.join('epochs/model.pth'))
