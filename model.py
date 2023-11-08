from argparse import Namespace
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import ResNet, BasicBlock


class VerboseResNet(ResNet):
    def verbose_forward(self, x: Tensor):
        with torch.no_grad():
            reprs = []
            # See note [TorchScript super()]
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            reprs.append(x)
            x = self.layer1(x)
            reprs.append(x)
            x = self.layer2(x)
            reprs.append(x)
            x = self.layer3(x)
            reprs.append(x)
            x = self.layer4(x)
            reprs.append(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x, reprs


class CLR(nn.Module):
    def __init__(self, backbone, args: Namespace):
        super().__init__()
        self.args = args
        self.backbone = backbone
        self.backbone.fc = nn.Identity()

        # TODO : for ResNet18, start from 512
        self.backbone_dim = 512
        self.online_head = nn.Linear(self.backbone_dim, args.label_num)

        if self.args.mode == "simclr":
            sizes = [self.backbone_dim, 256, 128]
            layers = []
            for i in range(len(sizes) - 2):
                layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
                layers.append(nn.BatchNorm1d(sizes[i+1]))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[-1]))
            self.projector = nn.Sequential(*layers)
        elif self.args.mode == "single":
            self.projector = nn.Linear(self.backbone_dim, 128, bias=False)

        self.relu = nn.ReLU(inplace=True)

    def verbose_forward(self, y1):
        r1, reprs = self.backbone.verbose_forward(y1)
        z1 = self.projector(r1)
        reprs.append(z1)
        return reprs

    def forward(self, y1, y2, labels):
        r1 = self.backbone(y1)
        r2 = self.backbone(y2)

        if self.args.mode == "baseline":
            z1 = r1
            z2 = r2
            loss = infoNCE(z1, z2) / 2 + infoNCE(z2, z1) / 2
        elif self.args.mode == "directclr":
            z1 = r1[:, :self.args.dim]
            z2 = r2[:, :self.args.dim]
            loss = infoNCE(z1, z2) / 2 + infoNCE(z2, z1) / 2
        elif self.args.mode == "group":
            idx = np.arange(2048)
            np.random.shuffle(idx)
            loss = 0
            for i in range(8):
                start = i * 256
                end = start + 256
                z1 = r1[:, idx[start:end]]
                z2 = r2[:, idx[start:end]]
                loss = loss + infoNCE(z1, z2) / 2 + infoNCE(z2, z1) / 2

        elif self.args.mode == "simclr" or self.args.mode == "single":
            z1 = self.projector(r1)
            z2 = self.projector(r2)
            loss = infoNCE(z1, z2) / 2 + infoNCE(z2, z1) / 2

        logits = self.online_head(r1.detach())
        cls_loss = torch.nn.functional.cross_entropy(logits, labels)
        acc = torch.sum(torch.eq(torch.argmax(logits, dim=1), labels)) / logits.size(0)

        loss = loss + cls_loss

        return loss, acc


def infoNCE(nn, p, temperature=0.1):
    nn = torch.nn.functional.normalize(nn, dim=1)
    p = torch.nn.functional.normalize(p, dim=1)
    # nn = gather_from_all(nn)
    # p = gather_from_all(p)
    logits = nn @ p.T
    logits /= temperature
    n = p.shape[0]
    labels = torch.arange(0, n, dtype=torch.long).cuda()
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss


def get_layers(model_name):
    if model_name == 'resnet18':
        return [2, 2, 2, 2]


def get_model(backbone_name, model_args, **kwargs):
    backbone = VerboseResNet(BasicBlock, get_layers(backbone_name), **kwargs)
    return CLR(backbone, model_args)
