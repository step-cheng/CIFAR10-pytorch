import torch
from torch import nn

# # Returns list of layers for a model
# def get_children(model: torch.nn.Module):
#     # get children form model!
#     children = list(model.children())
#     flatt_children = []
#     if children == []:
#         # if model has no children; model is last child! :O
#         return model
#     else:
#        # look for children from children... to the last child!
#        for child in children:
#             try:
#                 flatt_children.extend(get_children(child))
#             except TypeError:
#                 flatt_children.append(get_children(child))
#     return flatt_children

class Resnet50(nn.Module):
    def __init__():
        super().__init__()