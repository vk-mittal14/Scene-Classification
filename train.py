# import the lib. we wil need
import torch
import torchvision
from torch import nn
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm
from model import Encoder, Decoder
import random
import warnings
warnings.filterwarnings("ignore")

SEED = 2019

cudnn.benchmark = False
cudnn.deterministic = True
random.seed(SEED)
np.random.seed(SEED+1)
torch.manual_seed(SEED+2)
torch.cuda.manual_seed_all(SEED+3)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(.2),
        transforms.RandomVerticalFlip(.3),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


data_dir = "./data"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True) for x in ["train", "val"]}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes



encoder = Encoder(8).to(device)
model_ft = Decoder(2048, 8, 32, 64, 128, 31, 30).to(device)
criterion = nn.CrossEntropyLoss().to(device)
plist = [
        {'params': encoder.parameters(), 'lr': 1e-5, "weight_decay": 1e-4},
        {'params': model_ft.parameters(), 'lr': 1e-3, "weight_decay": 1e-4}
        ]

optimizer_ft = optim.Adam(plist)

def train_model(encoder, model, criterion, optimizer, num_epochs=50):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    val_acc_history = []
    val_loss_history = []
    train_acc_history = []
    train_loss_history = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                encoder.train()  # Set model to training mode
                model.train()
            else:
                encoder.eval()   # Set model to evaluate mode
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], total=len(dataloaders[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    inputs = encoder(inputs)
                    outputs = model(inputs).to(device)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            # if phase == 'train':
                # scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_encoder_wts = copy.deepcopy(encoder.state_dict())

            if phase == "val":
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)

            if phase == "train":
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)
                
        torch.save(
            {"model": model.state_dict(),
            "best_model": best_model_wts,
            "encoder": encoder.state_dict(),
            "best_encoder": best_encoder_wts,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "history": {"loss": [train_loss_history, val_loss_history], 
                            "accuracy": [train_acc_history, val_acc_history]}},
            "arcnet_training.pth"
        )
        print()


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    encoder.load_state_dict(best_encoder_wts)
    return model, encoder, {"loss": [train_loss_history, val_loss_history], 
                            "accuracy": [train_acc_history, val_acc_history]}



model_ft, encoder_ft, models_history = train_model(encoder, model_ft, criterion, optimizer_ft, num_epochs=50)