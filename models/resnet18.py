import torch
import torch.nn as nn
import torchvision.models as models

print(models.resnet18())
myResNet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for param in myResNet18.parameters():
	param.requires_grad = False

myResNet18.fc = nn.Linear(in_features=512,out_features=10, bias=True)

print(myResNet18)