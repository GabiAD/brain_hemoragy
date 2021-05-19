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
import io
import Augmentor
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

# set random seed to correctly compare the results of experiments
random.seed(42)

# logs and checkpoints
model_save_name = hp.name
checkpoint_path = hp.ckpt if hp.ckpt not in ["", None] else None

LOGS_DIR = os.path.join("./logs", model_save_name)
CKPT_DIR = os.path.join("./checkpoints", model_save_name)

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

writer = SummaryWriter(LOGS_DIR)

# load hyperparameters from file
model_name = hp.model_name
num_classes = hp.num_classes
batch_size = hp.batch_size
num_epochs = hp.num_epochs
feature_extract = hp.feature_extract

# set device to gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# reference: https://www.tensorflow.org/tensorboard/image_summaries
def plot_confusion_matrix(cm, class_names):
  """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
  figure = plt.figure(figsize=(8, 8))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  # Normalize the confusion matrix.
  cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return figure


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, fold_number=None):
    fold = f"_fold_{fold_number}" if fold_number is not None else ""
    print(f"Fold:{fold}")
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    iter = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")

        # Each epoch has a training and validation phase
        for phase in dataloaders.keys():
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            correct_labels = []
            predict_labels = []

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    # compute outputs and loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # if phase is training, update the weights
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        iter += 1
                        if iter % hp.log_iterations == 0:
                            writer.add_scalars(f'{fold}/Loss', {'Train': loss.item()}, iter)
                            writer.flush()
                    elif phase == 'val':
                        predict_labels.extend(torch.max(outputs, -1)[1].cpu().detach().tolist())
                        correct_labels.extend(labels.tolist())

            # if phase is validating, log losses, accuracy, f1 score and plot confusion matrix
            if phase == 'val':
                writer.add_scalars(f'{fold}/Loss', {'Val': loss.item()}, iter)
                cm = confusion_matrix(correct_labels, predict_labels)

                figure = plot_confusion_matrix(cm, class_names=["negative", "positive"])

                figure.canvas.draw()
                cm_image = torch.from_numpy(np.array(figure.canvas.renderer.buffer_rgba())).float().permute(2,0,1)/255
                cm_image = cm_image[:-1]
                print(cm_image.shape)

                writer.add_image(f"{fold}/Confusion_matrix", cm_image, iter)

                f1 = f1_score(correct_labels, predict_labels)
                writer.add_scalar(f'{fold}/F1', f1, iter)

                acc = accuracy_score(correct_labels, predict_labels)
                writer.add_scalar(f'{fold}/Accuracy', acc, iter)

                writer.flush()

        # if train on all images, save checkpoints
        if fold_number is None:
            torch.save({
                'model': model.state_dict(),
                }, os.path.join(CKPT_DIR, f"checkpoint{fold}_{epoch}.pth"))
        print()


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# load models with pretrained weights
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "wide_resnet50_2":
        """ "wide_resnet50_2"
        """
        model_ft = models.wide_resnet50_2(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


# apply transformations on every batch
def collate_fn(batch):
    # make pytorch tensors from images and normalize them
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    originals_imgs = [b[0] for b in batch]
    labels = [b[1] for b in batch]

    # negative_imgs = [originals_imgs[i] for i in range(len(originals_imgs)) if labels[i] == 0]
    # positive_imgs = [originals_imgs[i] for i in range(len(originals_imgs)) if labels[i] == 1]

    # apply different augmentations to train examples, e.g. flip and rotation
    #####
    p = Augmentor.DataPipeline([originals_imgs])
    # p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
    p.flip_left_right(probability=0.5)

    images_aug = p.sample(1)
    augmented_imgs = images_aug[0]
    #####
    # #####
    # seq = iaa.Sequential([
    #     iaa.Fliplr(0.5),
    #     iaa.MultiplySaturation(mul=(.5, 1.4))
    #     # iaa.PerspectiveTransform(scale=(0, 0.06))
    # ])
    #
    # augmented_imgs = seq(images=augmented_imgs)  # done by the library
    #####

    originals_imgs = [transform(im) for im in originals_imgs]
    # originals_imgs = [im.permute(2,0,1) for im in originals_imgs]
    augmented_imgs = [transform(im) for im in augmented_imgs]

    # all_imgs = augmented_imgs
    # all_imgs.extend([transform(im) for im in negative_imgs])

    imgs = torch.stack(augmented_imgs)
    # labels = [1] * len(positive_imgs) + [0] * len(negative_imgs)

    return imgs, torch.tensor(labels)


# use the same transformations on validation images, except augmentations
def collate_fn_val(batch):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    originals_imgs = [b[0] for b in batch]
    labels = [b[1] for b in batch]

    originals_imgs = [transform(im) for im in originals_imgs]
    imgs = torch.stack(originals_imgs)

    return imgs, torch.tensor(labels)


if __name__ == "__main__":
    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    label_img_pairs = [ln.strip() for ln in open(hp.dataset).readlines()]
    random.shuffle(label_img_pairs)

    # use cross validation method for hyperparameters tuning
    # either use 3-fold cross validation for testing hyperparameters or use one the entire dataset to train model with a previous testet set of hyperparameters
    for i in range(hp.train_val_splits):
        if hp.train_val_splits > 1:
            train_samples = label_img_pairs[:int(i/hp.train_val_splits*len(label_img_pairs))] + label_img_pairs[int((i+1)/hp.train_val_splits*len(label_img_pairs)):]
            val_samples = label_img_pairs[int(i/hp.train_val_splits*len(label_img_pairs)):int((i+1)/hp.train_val_splits*len(label_img_pairs))]
        else:
            train_samples = label_img_pairs
            val_samples = []

        dataset_train = datasets.BrainsDataset(root=hp.dataroot, image_label_pairs=train_samples)
        dataset_val = datasets.BrainsDataset(root=hp.dataroot, image_label_pairs=val_samples)

        dataloaders_dict = {}
        dataloaders_dict["train"] = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        if hp.train_val_splits > 1:
            dataloaders_dict["val"] = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_val)

        # Send the model to GPU
        model_ft = model_ft.to(device)

        params_to_update = model_ft.parameters()
        print("Params to learn:")
        if feature_extract:
            params_to_update = []
            for name,param in model_ft.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)
        else:
            for name,param in model_ft.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)

        # select the optimizer
        # optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        optimizer_ft = optim.Adam(params_to_update, lr=hp.lr, betas=(hp.beta1, hp.beta2))

        # select the criterion
        criterion = nn.CrossEntropyLoss()

        if hp.train_val_splits > 1:
            # Train and evaluate
            train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, fold_number=i)
        else:
            train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)
