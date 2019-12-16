import argparse
import os
import warnings

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

import utils
from model import Model

warnings.filterwarnings("ignore")


def train(model, train_loader, meta_ids, optimizer, epoch):
    model.train()
    total_loss, total_acc, n_data, train_bar = 0.0, 0.0, 0, tqdm(train_loader)
    for data, target in train_bar:
        optimizer.zero_grad()
        output = model(data.to(gpu_ids[0]))
        labels = meta_ids[target].to(gpu_ids[0])
        loss = cross_entropy_loss(output.permute(0, 2, 1).contiguous(), labels)
        loss.backward()
        optimizer.step()

        n_data += len(data)
        total_loss += loss.item() * len(data)
        total_acc += torch.sum((torch.argmax(output, dim=-1) == labels).float()).item() / ensemble_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f} Acc:{:.2f}%'
                                  .format(epoch, epochs, total_loss / n_data, total_acc / n_data * 100))

    return total_loss / n_data, total_acc / n_data * 100


def test(model, test_loader, meta_ids, epoch):
    model.eval()
    total_loss, total_top1, total_top5, n_data, memory_bank = 0.0, 0.0, 0.0, 0, []
    train_bar = tqdm(train_loader, desc='Feature extracting')
    test_bar = tqdm(test_loader)
    with torch.no_grad():
        for data, target in train_bar:
            memory_bank.append(model(data.to('cuda')))
        memory_bank = torch.cat(memory_bank).t().contiguous()
        memory_bank_labels = torch.tensor(train_loader.dataset.targets).to('cuda')
        for data, target in test_bar:
            y = model(data.to('cuda'))
            n_data += len(data)
            sim_index = torch.mm(y, memory_bank).argsort(dim=-1, descending=True)[:, :min(memory_bank.size(-1), 200)]
            sim_labels = torch.index_select(memory_bank_labels, dim=-1, index=sim_index.reshape(-1)).view(len(data), -1)
            pred_labels = []
            for sim_label in sim_labels:
                pred_labels.append(torch.histc(sim_label.float(), bins=len(train_loader.dataset.classes)))
            pred_labels = torch.stack(pred_labels).argsort(dim=-1, descending=True)
            total_top1 += torch.sum(
                (pred_labels[:, :1] == target.to('cuda').unsqueeze(dim=-1)).any(dim=-1).float()).cpu().item()
            total_top5 += torch.sum(
                (pred_labels[:, :5] == target.to('cuda').unsqueeze(dim=-1)).any(dim=-1).float()).cpu().item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / n_data * 100, total_top5 / n_data * 100))

    return total_loss / n_data, total_top1 / n_data * 100, total_top5 / n_data * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Shadow Mode')
    parser.add_argument('--data_path', default='/home/data/imagenet/ILSVRC2012', type=str, help='Path to dataset')
    parser.add_argument('--model_type', default='resnet18', type=str,
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnext50_32x4d'], help='Backbone type')
    parser.add_argument('--share_type', default='layer1', type=str,
                        choices=['none', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4'], help='Shared module type')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=100, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--ensemble_size', default=48, type=int, help='Ensemble model size')
    parser.add_argument('--meta_class_size', default=12, type=int, help='Meta class size')
    parser.add_argument('--gpu_ids', default='0,1,2,3,4,5,6,7', type=str, help='Selected gpu')
    parser.add_argument('--load_ids', action='store_true', help='Load already generated ids or not')

    # args parse and data prepare
    args = parser.parse_args()
    data_path, model_type, share_type, batch_size = args.data_path, args.model_type, args.share_type, args.batch_size
    epochs, ensemble_size, meta_class_size = args.epochs, args.ensemble_size, args.meta_class_size
    gpu_ids, load_ids = [int(gpu) for gpu in args.gpu_ids.split(',')], args.load_ids
    train_data = datasets.ImageFolder(root='{}/train'.format(data_path), transform=utils.train_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    test_data = datasets.ImageFolder(root='{}/val'.format(data_path), transform=utils.test_transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8)

    # model setup and meta id config
    model = nn.DataParallel(Model(meta_class_size, ensemble_size, share_type, model_type).to(gpu_ids[0]),
                            device_ids=gpu_ids)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print("# trainable parameters:", sum(param.numel() if param.requires_grad else 0 for param in model.parameters()))
    lr_scheduler = MultiStepLR(optimizer, milestones=[int(epochs * 0.6), int(epochs * 0.8)], gamma=0.1)
    cross_entropy_loss = nn.CrossEntropyLoss()
    results = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = '{}_{}_{}_{}'.format(model_type, share_type, ensemble_size, meta_class_size)
    ids_name = 'results/imagenet_{}_ids.pth'.format(save_name_pre)
    if load_ids:
        if os.path.exists(ids_name):
            meta_ids = torch.load(ids_name)
        else:
            raise FileNotFoundError('{} is not exist'.format(ids_name))
    else:
        meta_ids = torch.tensor(utils.assign_meta_id(meta_class_size, len(train_data.classes), ensemble_size))
        torch.save(meta_ids, ids_name)

    # training loop
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, train_loader, meta_ids, optimizer, epoch)
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        test_loss, test_acc_1, test_acc_5 = test(model, test_loader, meta_ids, epoch)
        results['test_loss'].append(test_loss)
        results['test_acc_1'].append(test_acc_1)
        results['test_acc_5'].append(test_acc_5)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/imagenet_{}_features_extractor_results.csv'
                          .format(save_name_pre), index_label='epoch')
        lr_scheduler.step(epoch)
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.module.state_dict(), 'epochs/imagenet_{}_features_extractor.pth'.format(save_name_pre))
