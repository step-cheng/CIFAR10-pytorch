"""
ResNet50 uses residual networks to preserve gradients during backpropagation
I adapt the ResNet50 model for CIFAR10 by making a few adjustments to the first several layers and the linear stack.
Changed first conv layer from stride=2 to stride=1 since image input very little already.
"""

import torch
from torch import nn

# WHAT DOES DOWNSAMPLE DO?
class Bottleneck(nn.Module):
    def __init__(self,cin, cout, cmid,s=1,ds=None):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(cin, cmid,1,1,bias=False),
            nn.BatchNorm2d(cmid),
            nn.Conv2d(cmid,cmid,3,s,padding=1,bias=False),
            nn.BatchNorm2d(cmid),
            nn.Conv2d(cmid,cout,1,1,bias=False),
            nn.BatchNorm2d(cout),
        )
        self.relu = nn.ReLU(inplace=True)
        self.ds = ds
    def forward(self,x):
        identity = x
        out = self.stack(x)
        if self.ds:
            identity = self.ds(x)
        out += identity
        output = self.relu(out)
        return output


class myResNet50(nn.Module):
    # bottles: key: layer, value: cin cout cmid, num bottles
    bottles = {1 : (64,64,256,3), 2 : (256,128,512,4), 3 : (512,256,1024,6),
               4 : (1024,512,2048,3)}
    def __init__(self):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3,64,7,1,3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,padding=1)
        )
        self.conv_stack = self.make_layers(self.bottles)
        self.linear = nn.Sequential(
            nn.Linear(2048,10, bias=True),
            nn.Softmax(dim=1)
        )
        
    def forward(self,x):
        out = self.initial(x)
        output = self.conv_stack(out)
        output = output.view(output.shape[0], -1)
        logits = self.linear(output)
        return logits

    def make_layers(self,bottles):
        layers = []
        for key in bottles:
            cin,cmid,cout,num = bottles[key]
            lay = []
            if key == 1:
                ds = nn.Sequential(
                    nn.Conv2d(cin,cout,1,1,bias=False),
                    nn.BatchNorm2d(cout)
                )
                lay.append(Bottleneck(cin,cout,cmid,ds=ds))
            else:
                ds = nn.Sequential(
                    nn.Conv2d(cin,cout,1,2,bias=False),
                    nn.BatchNorm2d(cout)
                )
                lay.append(Bottleneck(cin,cout,cmid,s=2,ds=ds))
            for i in range(num-1):
                lay.append(Bottleneck(cout,cout,cmid))
            layers.append(nn.Sequential(*lay))
        layers.append(nn.AdaptiveAvgPool2d((1,1)))
        return nn.Sequential(*layers)


def test():
    input = torch.randn((2,3,32,32))
    resnet = myResNet50()
    output = resnet(input)
    print(output)

# test()