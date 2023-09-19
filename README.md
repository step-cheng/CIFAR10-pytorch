# CIFAR10-pytorch

## Goal
Familiarize myself with PyTorch, Create and train my own CNN on the CIFAR10 image classification dataset and achieve an acceptable test accuracy, and Implement transfer learning and fine-tuning of various ImageNet architectures on the dataset to familiarize myself with the architectures. Achieve around 80+% testing accuracy on each model.

## Description
As my second project in deep learning, while following Prof. Justin Johnson's Deep Learning, I wanted to learn how to use PyTorch and reinforce my knowledge of convolutional neural networks and the various ImageNet architectures over the years. I used the CIFAR-10 dataset.

My model is a 9 layer model, 6 convolutional layers and 3 fully connected layers. For its convolutional layers, I only used a series of 3x3 2d convolutions with padding of size 1, which borrows from the idea of increasing the receptive field to 5x5 by pairing two 3x3 convolutions. I normalized the data across each channel for the entire train set and test set prior to running the model for data preprocessing, and I used batch normalization after each convolution. For learning, I used a cosine learning rate schedule with Adam optimization. To address overfitting, I used a weight decay of 0.01 and horizontal flip data augmentation. 

I fine-tuned VGG19, ResNet-18, ResNet-50, and MobileNetv3 small. Since the CIFAR10 dataset is pretty different from ImageNet, I not only changed the classifier layer to match the number of classes in CIFAR10, but I also froze less layers. For example, in ResNet-18, I did not freeze the last residual network block. In Vgg19, since max pooling was done 5 times, each channel layer by the end of the convolution stack was 1x1 since CIFAR-10 has 32x32 images. Replacing the last max pool layer with the identity function increased the accuracy from around 50% to around 75%. Further fine-tuning of the pre-trained models would further increase the accuracy.

The hyper parameters for the models needed to be different. For the pre-trained models, reducing the learning rate from 0.001 to 0.0001 seemed to benefit the accuracy. I believe this is because only a few layers are being trained and the rest are frozen, so a smaller training rate is necessary since fewer layers are actually learning on the dataset, compared to training an entire model. I also did not need as many epochs since less weights were being trained.

Current Test Accuracies:

My Model: 87.26%

VGG19: 79.6%

ResNet18: 81.07%

ResNet50: 80.55%

MobileNetv3: 39.85% (work in progress)


## What I Learned
I learned how to use PyTorch, including how to load data, create .nn Module classes, use optimizers and learning rate schedules, backpropagate, etc. I also reinforced my knowledge of convolutional neural networks, ordering batch normalization and nonlinear activation functions, and how the ImageNet architectures work. 
