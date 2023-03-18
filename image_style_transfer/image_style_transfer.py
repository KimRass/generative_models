# References
    # https://github.com/Aleadinglight/Pytorch-VGG-19/blob/master/VGG_19.ipynb

import torch
import torch.nn as nn
import torch.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.models import vgg19_bn