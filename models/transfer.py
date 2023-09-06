import torch
import torch.nn as nn
from torchvision import models


myVgg19 = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT)

for param in myVgg19.parameters():
	param.requires_grad = False

new_classifier = nn.Sequential(
	nn.Linear(in_features=512, out_features=64, bias=True),
	nn.ReLU(inplace=True),
	nn.Linear(64,10,bias=True)
)
myVgg19.avgpool = nn.AvgPool2d(kernel_size=1,stride=1)
myVgg19.classifier = new_classifier
