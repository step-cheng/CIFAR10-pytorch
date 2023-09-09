import torch
from torchvision import datasets, transforms

def accuracy(pred,y):
  """Returns accuracy of predictions returned by the model by comparing with labels"""
  guesses = torch.argmax(pred,dim=1)
  matches = torch.sum(guesses==y).item()
  return matches/torch.numel(y), matches

# Train set: tensor([0.4914, 0.4822, 0.4465]) tensor([0.2470, 0.2435, 0.2616])
# Train set: tensor([0.4942, 0.4851, 0.4504]) tensor([0.2467, 0.2429, 0.2616])
def find_norm_args(dataset):
  """Used to get the per channel mean and std of a dataset.
  In the code, used to find the args for transforms.Normalize when retrieving dataset"""
  loader = torch.utils.data.DataLoader(dataset,batch_size=len(dataset))
  for batch, (X,y) in enumerate(loader):
    batch_avgs = torch.mean(X,dim=[0,2,3])
    batch_stds = torch.std(X,dim=[0,2,3])
    print(batch_avgs, batch_stds)
    break
  
# DEPRECATED
# Normalizes batch within train data for preprocessing, no need because of dataset normalize transform
def preprocess(x):
  # find std and mean per channel
  x_stds = torch.std(x, dim=(0,2,3))
  x_means = torch.sum(x, dim=(0,2,3)) / (x.shape[0]*x.shape[2]*x.shape[3])

  print(x_means, x_stds)

  for channel in range(3):
    x[:,channel,:,:] -= x_means[channel]
    x[:,channel,:,:] /= x_stds[channel]
  return x
