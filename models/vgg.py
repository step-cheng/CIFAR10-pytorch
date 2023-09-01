"""
VGG19 model adapted for the CIFAR10 dataset. Since VGG19 was made for the ImageNet dataset, 
and the inputs were 224x224 pixels, the linear layers needed to be modified. Adaptive pooling
was replaced with average pooling since the HxW after the convolution stack becomes 1x1. 
Batch Normalization was added too.
"""

import torch
from torch import nn

layers = [3,64,64,'P',128,128,'P',256,256,256,256,'P',512,512,512,512,'P',512,512,512,512,'P']

class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = self.make_layers(layers)
        # Adapt vgg19's linear stack since it was originally for imagenet
        self.linear_stack = nn.Sequential(
            nn.Linear(512,64,bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(64,64,bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(64,10,bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        out = self.layers(x)
        out = out.view(out.shape[0],-1)
        logits = self.linear_stack(out)
        return logits
        
    def make_layers(self,layers):
        res = []
        input = layers[0]
        for output in layers[1:]:
            if output == 'P':
                res.append(nn.MaxPool2d(kernel_size=2,stride=2))
            else:
                res.append(nn.Conv2d(input,output, kernel_size=(3,3), stride=1, padding=1))
                res.append(nn.ReLU(inplace=True))
                res.append(nn.BatchNorm2d(output))
                input = output
        # since vgg does maxpool 5 times, each image has HxW 1x1, thus cannot do adaptive avg pooling, instead, just keep that value
        res.append(nn.AvgPool2d(kernel_size=1,stride=1))
        return nn.Sequential(*res)

def vgg19test():
    input = torch.randn((2,3,32,32))
    model = VGG19()
    logits = model(input)
    return logits

# logits = vgg19test()