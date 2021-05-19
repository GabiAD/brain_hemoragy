name = "wide_resnet_finetune_full_model_batch_64_epochs_60_Adam_augment_flip_all"
ckpt = "./checkpoints/vgg_finetune_full_model_batch_32_epochs_40_SGD_augment_flip_all_train_all_images/checkpoint_13.pth"

dataroot = "./data/imgs"  # Root directory
dataset="./data/train_labels.txt"
submission_dataset="./data/sample_submission.txt"

log_iterations = 1000
save_iterations = 4000

workers = 4  # Number of workers for dataloader
batch_size = 4  # Batch size during training
image_size = 224  # 64  # Spatial size of training images. All images will be resized to this size using a transformer

lr = 1e-3  # Learning rate for optimizers
beta1 = 0  # Beta1 hyperparam for Adam optimizers
beta2 = 0.9  # Beta2 hyperparam for Adam optimizers

ngpu = 1  # Number of GPUs available. Use 0 for CPU mode.

model_name = "wide_resnet50_2"  # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
num_classes = 2  # Number of classes in the dataset
batch_size = 64  # Batch size for training (change depending on how much memory you have)
num_epochs = 60  # Number of epochs to train for
train_val_splits = 3

feature_extract = False  # use freezed pretrained weights
