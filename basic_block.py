import torch
import torch.nn as nn


class Channel_Attention(nn.Module):
    def __init__(self, ch):
        super(Channel_Attention, self).__init__()
        self.CA_down = nn.Conv2d(in_channels=ch, out_channels=1, kernel_size=1, padding=0, stride=1)
        self.CA_up = nn.Conv2d(in_channels=1, out_channels=ch, kernel_size=1, padding=0, stride=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a,b,c,d = x.size()
        x_reshape = torch.reshape(x,[a,b,c*d])
        GP = torch.mean(x_reshape,dim=2).unsqueeze(dim=2).unsqueeze(dim=2)
        GP = GP.repeat([1,1,c,d])

        scale = self.relu(self.CA_down(GP))
        scale = self.sigmoid(self.CA_up(scale))
        return x * scale



class CAB(nn.Module):  # Channel Attention Block
    def __init__(self, ch):
        super(CAB, self).__init__()

        self.CA = Channel_Attention(ch=ch)
        self.conv1 = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=ch*2, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        out1 = self.relu(self.conv2(out1))
        Channel = self.CA(out1)

        return Channel  # x + res