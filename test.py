from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import models, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
import copy
import glob
import hyperparams as hp
import datasets
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import itertools
from skimage import io
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from tqdm import tqdm

# load model
model_ft = models.wide_resnet50_2(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, hp.num_classes)
input_size = 224

# load checkpoint
print("Loading checkpoint...")
checkpoint = torch.load(hp.ckpt)
model_ft.load_state_dict(checkpoint['model'])
print("Checkpoint loaded")

# set model to eval mode
model_ft.eval()

# use the same image transformations as in training
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# run inference on test images
with open("results.txt", "w") as res:
    res.write("id,class\n")

    for ln in tqdm(open(hp.submission_dataset)):
        im_path = os.path.join(hp.dataroot, f"{ln.split(',')[0]}.png")
        image = io.imread(im_path)

        image = transform(image)

        image = image.unsqueeze(0)

        output = model_ft(image)
        _, pred_class = torch.max(output, 1)

        res.write(f"{ln.split(',')[0]},{pred_class.item()}\n")
