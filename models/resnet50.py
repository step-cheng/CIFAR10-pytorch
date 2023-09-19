import torch
import torch.nn as nn
import torchvision.models as models

myResNet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# freeze everything except layer 4
i = 0
for param in myResNet50.parameters():
	if i >= 129: break
	param.requires_grad = False
	i += 1

myResNet50.fc = nn.Linear(2048, 10, bias=True)

"""
conv1 1
bn1 2
relu 0
maxpool 0
layer1 30
layer2 39
layer3 57
layer4 30
avgpool 0
fc 2
"""