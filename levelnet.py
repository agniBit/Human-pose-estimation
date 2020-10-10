import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import cfg.config as config
import cv2
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from torchsummary import summary

cfg = config.get_cfg_defaults()


class Reduce(nn.Module):
    def __init__(self, lvl):
        super(Reduce, self).__init__()
        self.reduce_plans = nn.Conv2d(cfg[lvl].out, cfg[lvl].reduce_to, (3, 3), stride=(1, 1), padding=(1,1))
        self.reduce_plans_2 = nn.Conv2d(cfg[lvl].reduce_to, cfg[lvl].reduce_to, (3, 3), stride=(1, 1), padding=(1,1))
    def forward(self, x):
        return self.reduce_plans_2(self.reduce_plans(x))

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
        self.conv5t = nn.ConvTranspose2d(cfg[lvl].inplans, cfg[lvl].conv5plans, (5, 5), stride=(2, 2), padding=(2,2), output_padding=1)
        self.conv3t = nn.ConvTranspose2d(cfg[lvl].inplans, cfg[lvl].conv3plans, (3, 3), stride=(2, 2), padding=(1,1), output_padding=1)
        self.conv1t = nn.ConvTranspose2d(cfg[lvl].inplans, cfg[lvl].conv1plans, (1, 1), stride=(2, 2) ,output_padding=1)
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


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(3, cfg["x"].outplans, (3, 3), stride=(1, 1), padding=(1, 1))
        self.relu = nn.LeakyReLU()
        self.batchNorm = nn.BatchNorm2d(cfg["x"].outplans)
        self.layer_x_2d = layer_ds('x_2d')
        self.layer_x_4d = layer_ds('x_4d')
        self.layer_x_8d = layer_ds('x_8d')
        self.layer_x_16d = layer_ds('x_16d')
        self.layer_x_32d = layer_ds('x_32d')
        self.layer_x_16u = upsample('x_16u')
        self.layer_x_8u = upsample('x_8u')
        self.layer_x_4u = upsample('x_4u')
        self.layer_x_2u = upsample('x_2u')
        self.layer_x_u = nn.ConvTranspose2d(cfg.x_u.inplans , cfg.x_u.reduce_to, (3, 3), stride=(2, 2), padding=(1, 1), output_padding= (1, 1))
        self.relu_x_u = nn.LeakyReLU()
        self.batchNorm_x_u = nn.BatchNorm2d(6*cfg.out_features)
        self.conv_x = nn.Conv2d(6*cfg.out_features , 3*cfg.out_features,(3,3),stride=(1, 1),padding=(1, 1))
        self.relu_x = nn.ReLU()
        self.batch_norm_x = nn.BatchNorm2d(3*cfg.out_features)
        self.conv_x_out = nn.Conv2d(3*cfg.out_features , cfg.out_features,(3,3),stride=(1, 1),padding=(1, 1))
        self.out = nn.ReLU()

    def forward(self, x):
        x_d = self.batchNorm(self.relu(self.conv(x)))
        x_2d = self.layer_x_2d.forward(x_d)
        x_4d = self.layer_x_4d.forward(x_2d)
        x_8d = self.layer_x_8d.forward(x_4d)
        x_16d = self.layer_x_16d.forward(x_8d)
        x_32d = self.layer_x_32d.forward(x_16d)
        x_16u = torch.cat([self.layer_x_16u.forward(x_32d), x_16d], 1)
        x_8u = torch.cat([self.layer_x_8u.forward(x_16u), x_8d], 1)
        x_4u = torch.cat([self.layer_x_4u.forward(x_8u), x_4d], 1)
        x_2u = torch.cat([self.layer_x_2u.forward(x_4u), x_2d], 1)
        x_u =  self.batchNorm_x_u(self.relu_x_u(torch.cat([self.layer_x_u.forward(x_2u), x_d], 1)))
        x = self.batch_norm_x(self.relu_x(self.conv_x(x_u)))
        out = self.out(self.conv_x_out(x))
        return out
