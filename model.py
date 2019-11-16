import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnext50_32x4d


class Model(nn.Module):
    def __init__(self, meta_class_size, ensemble_size, share_type, model_type, with_random, device_ids):
        super(Model, self).__init__()

        # backbone
        backbones = {'resnet18': (resnet18, 1), 'resnet34': (resnet34, 1), 'resnet50': (resnet50, 4),
                     'resnext50_32x4d': (resnext50_32x4d, 4)}
        backbone, expansion = backbones[model_type]
        module_names = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']

        # configs
        self.ensemble_size, self.with_random, self.device_ids = ensemble_size, with_random, device_ids

        # common features
        self.common_extractor, basic_model = [], backbone(pretrained=True)
        common_module_names = [] if share_type == 'none' else module_names[:module_names.index(share_type) + 1]
        for name, module in basic_model.named_children():
            if name in common_module_names:
                self.common_extractor.append(module)
        self.common_extractor = nn.Sequential(*self.common_extractor).cuda(device_ids[0])
        print("# trainable common feature parameters:",
              sum(param.numel() if param.requires_grad else 0 for param in self.common_extractor.parameters()))

        # individual features
        self.head, self.layer1, self.layer2, self.layer3, self.layer4 = [], [], [], [], []
        individual_module_names = module_names if share_type == 'none' else \
            module_names[module_names.index(share_type) + 1:]
        for i in range(ensemble_size):
            basic_model, heads = backbone(pretrained=True), []
            for name, module in basic_model.named_children():
                if name in individual_module_names and name in ['conv1', 'bn1', 'relu', 'maxpool']:
                    heads.append(module)
                if name in individual_module_names and name == 'layer1':
                    self.layer1.append(module.cuda(device_ids[0]))
                if name in individual_module_names and name == 'layer2':
                    self.layer2.append(module.cuda(device_ids[0]))
                if name in individual_module_names and name == 'layer3':
                    self.layer3.append(module.cuda(device_ids[0 if i < ensemble_size / 6 else 1]))
                if name in individual_module_names and name == 'layer4':
                    self.layer4.append(module.cuda(device_ids[0 if i < ensemble_size / 6 else 1]))
            self.head.append(nn.Sequential(*heads).cuda(device_ids[0]))
        self.head = nn.ModuleList(self.head)
        self.layer1 = nn.ModuleList(self.layer1)
        self.layer2 = nn.ModuleList(self.layer2)
        self.layer3 = nn.ModuleList(self.layer3)
        self.layer4 = nn.ModuleList(self.layer4)
        print("# trainable individual feature parameters:",
              (sum(param.numel() if param.requires_grad else 0 for param in self.head.parameters()) +
               sum(param.numel() if param.requires_grad else 0 for param in self.layer1.parameters()) + sum(
                          param.numel() if param.requires_grad else 0 for param in self.layer2.parameters()) + sum(
                          param.numel() if param.requires_grad else 0 for param in self.layer3.parameters()) + sum(
                          param.numel() if param.requires_grad else 0 for param in
                          self.layer4.parameters())) // ensemble_size)

        # individual classifiers
        self.classifiers = nn.ModuleList([nn.Linear(512 * expansion, meta_class_size).cuda(device_ids[1])
                                          for _ in range(ensemble_size)])
        print("# trainable individual classifier parameters:",
              sum(param.numel() if param.requires_grad else 0 for param in
                  self.classifiers.parameters()) // ensemble_size)

    def forward(self, x):
        batch_size = x.size(0)
        common_feature = self.common_extractor(x)
        if self.with_random:
            branch_weight = torch.rand(self.ensemble_size, device=x.device)
            branch_weight = F.softmax(branch_weight, dim=-1)
        else:
            branch_weight = torch.ones(self.ensemble_size, device=x.device)
        out = []
        for i in range(self.ensemble_size):
            individual_feature = branch_weight[i] * common_feature
            if len(self.head) != 0:
                individual_feature = self.head[i](individual_feature.cuda(self.device_ids[0]))
            if len(self.layer1) != 0:
                individual_feature = self.layer1[i](individual_feature.cuda(self.device_ids[0]))
            if len(self.layer2) != 0:
                individual_feature = self.layer2[i](individual_feature.cuda(self.device_ids[0]))
            if len(self.layer3) != 0:
                individual_feature = self.layer3[i](
                    individual_feature.cuda(self.device_ids[0 if i < self.ensemble_size / 6 else 1]))
            if len(self.layer4) != 0:
                individual_feature = self.layer4[i](
                    individual_feature.cuda(self.device_ids[0 if i < self.ensemble_size / 6 else 1]))
            global_feature = F.adaptive_avg_pool2d(individual_feature, output_size=(1, 1)).view(batch_size, -1)
            classes = self.classifiers[i](global_feature.cuda(self.device_ids[1]))
            out.append(classes)
        out = torch.stack(out, dim=1)
        return out
