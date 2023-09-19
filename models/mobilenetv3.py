import torch
import torch.nn as nn
import torchvision.models as models

myMobileNet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

for param in myMobileNet.features[:122].parameters():
	param.requires_grad = False

myMobileNet.classifier[3] = nn.Linear(1024,10,bias=True)

"""
0	3
1	10
2	9
3	9
4	13
5	13
6	13
7	13
8	13
9	13
10	13
11	13
12	3
"""