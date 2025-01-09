from math import log
import torch
import torch.nn as nn
from torchvision.transforms import RandomHorizontalFlip,RandomVerticalFlip
import orn
import math
from ..builder import NECKS
import torch.nn.functional as F

class RotationInvariantPooling(nn.Module):
    """Rotating invariant pooling module."""

    def __init__(self,nOrientation=8):
        super(RotationInvariantPooling, self).__init__()
        # self.nInputPlane = nInputPlane # 256
        self.nOrientation = nOrientation # 8

    def forward(self, x):
        """Forward function."""
        N, c, h, w = x.size()  
        x = x.view(N, -1, self.nOrientation, h, w)   
        x, _ = x.max(dim=2, keepdim=False) 
        return x

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
     
    def forward(self, q, k,):
        b, _, dim = q.shape
        scale = dim ** -0.5
        attn = (q @ k.transpose(-1, -2).contiguous()) * scale #b,hw,hw
        return attn.softmax(dim=-1)

class ConcatAttention(nn.Module):
    def __init__(self, channel):
        super(ConcatAttention, self).__init__()
      

        self.attn = Attention()
      
        self.conv = nn.Sequential(
            orn.ORConv2d(channel*2,channel*2,arf_config=[1,8],kernel_size=3,padding=1),
            orn.ORConv2d(channel*2,channel*2,arf_config=[8,8],kernel_size=1),
            RotationInvariantPooling(),
            nn.BatchNorm2d(channel*2),
            nn.ReLU(),
            RotationInvariantPooling(nOrientation=2)
        )
    
    def forward(self, x, y):
        b, c, h, w = x.shape
        cat = self.conv(torch.cat((x, y), dim=1)).reshape(b, c, -1).transpose(-1, -2).contiguous()

        x = x.reshape(b, c, -1).transpose(-1, -2).contiguous() #b,hw,c
        y = y.reshape(b, c, -1).transpose(-1, -2).contiguous()
       
        attn1 = self.attn(x, cat)
        attn2 = self.attn(y, cat)
        
        return tuple([(attn1 @ cat).transpose(-1, -2).reshape(b, c, h, w).contiguous(),
                      (attn2 @ cat).transpose(-1, -2).reshape(b, c, h, w).contiguous(),
                      attn1, attn2])
       

class ECABlock(nn.Module):
    def __init__(self, channel, gamma=2, beta=1):
        super(ECABlock, self).__init__()
        t = int(abs((log(channel, 2) + beta) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, k, padding=int(k / 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2).contiguous())
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1).contiguous())
        return x * y.expand_as(x)


@NECKS.register_module()
class MSPM(nn.Module):
    def __init__(self, reverse=True, channel=(768, 384, 192, 96)):
        super(MSPM, self).__init__()
        self.reverse = reverse
        self.attn1 = ConcatAttention(channel[0])
        self.attn2 = ConcatAttention(channel[1])
    
        
        if self.reverse:
            self.reverse_attn1 = ConcatAttention(channel[0])
            self.reverse_attn2 = ConcatAttention(channel[1])

        self.flip = RandomHorizontalFlip(p=1)
        self.flip1 = RandomVerticalFlip(p=1)
        self.eca1 = ECABlock(channel[0])
        self.eca2 = ECABlock(channel[1])
        self.reverse_eca1 = ECABlock(channel[0])
        self.reverse_eca2 = ECABlock(channel[1])

    def forward(self, inputs, reverses):
        front1 = inputs[-1]
        back1 = self.flip(self.flip1(reverses[-1]))
        feats1 = self.attn1(front1, back1)
        front_attn1 = self.eca1(feats1[0])
        front2 = inputs[-2]
        back2 = self.flip(self.flip1(reverses[-2]))
        feats2 = self.attn2(front2, back2)
        front_attn2 = self.eca2(feats2[0])

        results = []
        for i in range(len(inputs) - 2):
            results.append(inputs[i])
        results.append(front_attn2)
        results.append(front_attn1)
        if self.reverse:
            back_attn1 = self.reverse_eca1(feats1[1])
            back_attn2 = self.reverse_eca2(feats2[1])
            for i in range(len(inputs) - 2):
                results.append(self.flip(self.flip1(reverses[i])))
            results.append(back_attn2)
            results.append(back_attn1)
        
        return tuple(results)