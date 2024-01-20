import torch
import torch.nn as nn
from math import sqrt

from basic_block import CAB

class FC4(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FC4, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=in_ch, out_channels=96, kernel_size=11, stride=4, padding=0, bias=False)
        self.layer2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2, bias=False)
        self.layer3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=5, stride=1, padding=2, bias=False)
        self.layer4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=5, stride=1, padding=2, bias=False)
        self.layer5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=5, stride=1, padding=2, bias=False)
        self.layer6 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer7 = nn.Conv2d(in_channels=128, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.Maxpooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.layer1(x))
        out = self.Maxpooling(out)
        out = self.relu(self.layer2(out))
        out = self.Maxpooling(out)
        out = self.relu(self.layer3(out))
        out = self.relu(self.layer4(out))
        out = self.relu(self.layer5(out))
        out = self.Maxpooling(out)
        out = self.relu(self.layer6(out))
        out = abs(self.layer7(out))

        return out


class FC4_CAB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FC4_CAB, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=in_ch, out_channels=96, kernel_size=11, stride=4, padding=0, bias=False)
        self.layer2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2, bias=False)
        self.layer3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=5, stride=1, padding=2, bias=False)
        self.layer4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=5, stride=1, padding=2, bias=False)
        self.layer5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=5, stride=1, padding=2, bias=False)
        self.CAB_1 = CAB(ch=384)

        self.layer6 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer7 = nn.Conv2d(in_channels=128, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.Maxpooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.relu(self.layer1(x))
        out = self.Maxpooling(out)
        out = self.relu(self.layer2(out))
        out = self.Maxpooling(out)
        out = self.relu(self.layer3(out))
        out = self.CAB_1(out)
        out = self.relu(self.layer5(out))
        out = self.Maxpooling(out)
        out = self.relu(self.layer6(out))
        out = abs(self.layer7(out))

        return out



class Full_model(nn.Module):
    def __init__(self, in_ch):
        super(Full_model, self).__init__()
        self.Local_illu = FC4(in_ch=3, out_ch=3)
        self.Confidence = FC4_CAB(in_ch=3, out_ch=1)

    def forward(self,x):
        RGB = x[:,:3,:,:]
        Local = self.Local_illu(RGB)
        normal = (torch.norm(Local[:, :, :, :], p=2, dim=1, keepdim=True) + 1e-04)
        Local = Local[:, :, :, :] / normal

        NIR = x[:,3:,:,:]
        conf_map = self.Confidence(NIR)
        conf_map_re = torch.reshape(conf_map,[conf_map.size(0),conf_map.size(2)*conf_map.size(3)])
        sum = torch.sum(conf_map_re,dim=1)
        sum = sum.reshape([sum.size(0),1,1,1])
        conf_map = conf_map / (sum.repeat([1,1,conf_map.size(2),conf_map.size(3)])+0.00001)

        conf_map = conf_map.repeat([1,3,1,1])

        local_il = Local * conf_map
        local_il = local_il.reshape(local_il.size(0), local_il.size(1), local_il.size(2) * local_il.size(3))
        weighted_sum = torch.sum(local_il, dim=2)
        pred = weighted_sum / (torch.norm(weighted_sum, dim=1, p=2, keepdim=True)+0.00001)

        return pred, Local, conf_map