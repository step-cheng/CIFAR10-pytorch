import torch
import torch.nn as nn
from torchvision import models
import inspect

# create an instance of Vgg model
myVgg19 = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT)
myVgg19.avgpool = nn.Identity()
myVgg19.features = myVgg19.features[:52]
for param in myVgg19.features[:40].parameters():
	param.requires_grad = False

new_classifier = nn.Sequential(
	nn.Linear(in_features=512*2*2, out_features=128, bias=True),
	nn.ReLU(inplace=True),
	nn.Linear(128,10,bias=True)
)

myVgg19.classifier = new_classifier

