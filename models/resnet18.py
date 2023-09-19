import torch
import torch.nn as nn
import torchvision.models as models

myResNet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# freeze all except the last resnet block, layer 4
i = 0
for param in myResNet18.parameters():
	if (i>=45): break
	param.requires_grad = False
	i+=1

myResNet18.fc = nn.Linear(in_features=512,out_features=10, bias=True)

""""
conv1 1
bn1 2
relu 0
maxpool 0
layer1 12
layer2 15
layer3 15
layer4 15
fc 2
"""