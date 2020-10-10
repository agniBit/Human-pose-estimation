import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cfg.config as config
from torch.nn import CrossEntropyLoss
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from torchsummary import summary
from torch.autograd import Variable
import torchvision.models as models
import resnet50
import sys

# resnet50 = models.resnet50(pretrained=True)
# modules = list(resnet50.children())[:-2]
# resnet50 = nn.Sequential(*modules)
# for p in resnet50.parameters():
#     p.requires_grad = False
# outputs = []
#
#
# def hook(module, input, output):
#     outputs.append(output)
#
#
# backbone = resnet50.resnet50(pretrained=True)
# # backbone.conv1.register_froward_hook
# for p in backbone.parameters():
#     p.requires_grad = False
#
# # backbone.conv1.register_froward_hook(hook)
# # backbone.layer1.register_forward_hook(hook)
# # backbone.layer2.register_forward_hook(hook)
# # backbone.layer3.register_forward_hook(hook)
# # backbone.layer4.register_forward_hook(hook)
#
# out = backbone(torch.rand(1, 3, 224, 224))
# print(summary(backbone, (3, 224, 224)))
# print(out[0].shape, out[1].shape, out[2].shape, out[3].shape, out[4].shape)
# sum = 0
# for o1 in out:
#     sum += o1.shape[0]*o1.shape[1]*o1.shape[2]*o1.shape[3]
# print(sum)
# assert False

class Reduce(nn.Module):
    def __init__(self, lvl):
        super(Reduce, self).__init__()
        self.reduce_plans = nn.Conv2d(cfg[lvl].out, cfg[lvl].reduce_to, (3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        return self.reduce_plans(x)


class layer_ds(nn.Module):
    def __init__(self, lvl):
        super(layer_ds, self).__init__()
        self.lvl = lvl
        self.conv5 = nn.Conv2d(cfg[lvl].inplans, cfg[lvl].conv5plans, (5, 5), stride=(1, 1), padding=(2, 2))
        self.conv3 = nn.Conv2d(cfg[lvl].inplans, cfg[lvl].conv3plans, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv1 = nn.Conv2d(cfg[lvl].inplans, cfg[lvl].conv1plans, (1, 1), stride=(1, 1))
        self.relu = nn.LeakyReLU()
        self.batchNorm = nn.BatchNorm2d(cfg[lvl].out)
        self.pool = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.reduce = Reduce(lvl)

    def forward(self, x):
        conv5 = self.conv5(x)
        conv3 = self.conv3(x)
        conv1 = self.conv1(x)
        cat_plans = torch.cat([conv5, conv3, conv1], 1)
        out_plans = self.reduce.forward(self.batchNorm(self.pool(self.relu(cat_plans))))
        return out_plans


class upsample(nn.Module):
    def __init__(self, lvl):
        super(upsample, self).__init__()
        self.conv5t = nn.ConvTranspose2d(cfg[lvl].inplans, cfg[lvl].conv5plans, (5, 5),
                                         stride=(2, 2), padding=(2, 2), output_padding=1)
        self.conv3t = nn.ConvTranspose2d(cfg[lvl].inplans, cfg[lvl].conv3plans, (3, 3),
                                         stride=(2, 2), padding=(1, 1), output_padding=1)
        self.conv1t = nn.ConvTranspose2d(cfg[lvl].inplans, cfg[lvl].conv1plans, (1, 1),
                                         stride=(2, 2), output_padding=1)
        self.relu = nn.LeakyReLU()
        self.batchNorm = nn.BatchNorm2d(cfg[lvl].out)
        self.reduce = Reduce(lvl)

    def forward(self, x):
        conv5t = self.conv5t(x)
        conv3t = self.conv3t(x)
        conv1t = self.conv1t(x)
        out_ = self.batchNorm(self.relu(torch.cat([conv5t, conv3t, conv1t], 1)))
        out_plans = self.reduce.forward(out_)
        return out_plans


class LevelNetModel(nn.Module):
    def __init__(self, load_model_weights):
        super(LevelNetModel, self).__init__()


class Model(nn.Module):
    def __init__(self, load_head_weights=True, backbone_name='resnet50', pretrained=True, isFeatures=False):
        super(Model, self).__init__()
        self.levelnetmodel = LevelNetModel(load_head_weights)
        self.isFeatures = isFeatures
        self.features = dict()
        self.pretrained = pretrained
        self.backbone = None
        self.backbone_name = backbone_name
        self.outputs = []

    def load_backbone(self):
        if self.backbone_name == 'resnet50':
            if self.pretrained:
                self.backbone = models.resnet50(pretrained=True)
                modules = list(self.backbone.children())[:-2]
                self.backbone = nn.Sequential(*modules)
                for p in self.backbone.parameters():
                    p.requires_grad = False
            else:
                self.backbone = models.resnet50()
                modules = list(self.backbone.children())[:-2]
                self.backbone = nn.Sequential(*modules)
            self.backbone.layer1.register_forward_hook(self.hook)
            self.backbone.layer2.register_forward_hook(self.hook)
            self.backbone.layer3.register_forward_hook(self.hook)
            self.backbone.layer4.register_forward_hook(self.hook)
        else:
            print("not a valid backbone")

    def hook(self, module, input, output):
        self.outputs.append(output)

    def forward(self, x):
        if self.isFeatures:
            self.load_backbone()
