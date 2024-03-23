import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import h5py
from torch import Tensor
from typing import Type
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
from torchview import draw_graph
from torchviz import make_dot
import warnings
warnings.filterwarnings("ignore")
from torchmetrics import Accuracy
import copy
import torch.optim as optim
import tqdm
from tqdm import tqdm
from torchmetrics import Accuracy
from torch.utils.data import TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from PIL import Image
from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.vision_transformer import PatchEmbed, Block


from dataclasses import dataclass
@dataclass
class ModelArgs:
    dim: int = 24          # Dimension of the model embeddings
    hidden_dim: int = 512   # Dimension of the hidden layers
    n_heads: int = 8        # Number of attention heads
    n_layers: int = 10       # Number of layers in the transformer
    patch_size: int = 5     # Size of the patches (typically square)
    n_channels: int = 3     # Number of input channels (e.g., 3 for RGB images)
    n_patches: int = 75     # Number of patches in the input
    n_classes: int = 2     # Number of target classes
    dropout: float = 0.2    # Dropout rate for regularization
