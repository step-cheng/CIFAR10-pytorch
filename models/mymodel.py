"""
My own model, which I made to help me learn how to use the PyTorch framework. After making this model,
which generally achieves a 65% training accuracy, I became much more comfortable working with PyTorch,
and started implementing famous architectures.
"""

import torch
from torch import nn

# nn is composed of a bunch of subclasses, myNN inherits from nn.Module
# within a subclass of torch.nn.Module, it's assumed we want to track gradients on the layer's weights
class myNN(nn.Module):

  def __init__(self):
    super().__init__()
    self.flat = nn.Flatten()

    # Sequential class stores modules that will be passed sequentially through constructor
    # input: 64x3x32x32;   output: 64x16x8x8
    self.conv_pool_stack = nn.Sequential(
      nn.Conv2d(3,16,kernel_size=3,padding=1, bias=False),
      nn.ReLU(),
      nn.BatchNorm2d(16),
      nn.Conv2d(16,16,kernel_size=3,padding=1, bias=False),
      nn.ReLU(),
      nn.MaxPool2d(2,stride=2),
      nn.BatchNorm2d(16),
      nn.Conv2d(16,32,kernel_size=3,padding=1,bias=False),
      nn.ReLU(),
      nn.BatchNorm2d(32),
      nn.Conv2d(32,32,kernel_size=3,padding=1,bias=False),
      nn.ReLU(),
      nn.BatchNorm2d(32),
      nn.MaxPool2d(2,stride=2),
      nn.BatchNorm2d(32)
    )

    # input: 64x32x8x8;    output: 16x10
    self.linear_relu_stack = nn.Sequential(
      self.flat,           # Flatten class flattens starting at dimension default 1 and ending at dimension default -1 --> 16x2048
      nn.Linear(2048, 512, bias=True),
      nn.ReLU(),
      nn.Linear(512, 64, bias=True),
      nn.ReLU(),
      nn.Linear(64, 10, bias=True),
      nn.Softmax(dim=1)
    )

  def forward(self,x):
    after_conv = self.conv_pool_stack(x)
    logits = self.linear_relu_stack(after_conv)
    return logits