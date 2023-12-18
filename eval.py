#%% Import Packages
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

# Customized function
from model import UNet
from dataset import ImageDataset, ToTensor, RandomFlip, Normalization
from utils import save, load


# Evaluation Parameters
BATCH_SIZE = 4

data_dir = './dataset'
ckpt_dir = './checkpoint'
log_dir = './log'
result_dir = './results'

if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Transform
transform = transforms.Compose([
    Normalization(mean=0.5, std=0.5), ToTensor()
])

dataset_test = ImageDataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
loader_test = DataLoader(dataset_test , batch_size=BATCH_SIZE, shuffle=False)

# Network
net = UNet().to(device)
# Loss function
fn_loss = nn.BCEWithLogitsLoss().to(device)
# Optimizer
optim = torch.optim.Adam(net.parameters(), lr=lr)
# etc
num_data_test = len(dataset_test)
num_batch_test = int(np.ceil(num_data_test / BATCH_SIZE))
# etc function
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x*std) + mean
fn_class = lambda x: 1.0 * (x>0.5)



# Network Training
ST_EPOCH = 0
net, optim, ST_EPOCH = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

with torch.no_grad():
    net.eval()
    loss_arr = []

    for batch_idx, data in enumerate(loader_test, 1):
        # forward pass
        label = data['label'].to(device)
        input = data['input'].to(device)
        output = net(input)
        # Loss 
        loss = fn_loss(output, label) 
        loss_arr += [loss.item()]
        # To numpy
        label = fn_tonumpy(label)
        input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
        output = fn_tonumpy(fn_class(output))

        print(f"TEST: BATCH [{batch_idx} / {num_batch_test}] | LOSS [{ np.mean(loss_arr) }]")

        for j in range(label.shape[0]):
            id = num_batch_test * (batch_idx - 1) + j
            # save (png format)
            plt.imsave(os.path.join(result_dir, "png", f"label_{id}.png", label[j].squeeze(), cmap='gray'))
            plt.imsave(os.path.join(result_dir, "png", f"input_{id}.png", input[j].squeeze(), cmap='gray'))
            plt.imsave(os.path.join(result_dir, "png", f"output_{id}.png", output[j].squeeze(), cmap='gray'))
            # save (numpy format)
            np.save(os.path.join(result_dir, "numpy", f"label_{id}.npy", label[j].squeeze()))
            np.save(os.path.join(result_dir, "numpy", f"input_{id}.npy", input[j].squeeze()))
            np.save(os.path.join(result_dir, "numpy", f"output_{id}.npy", output[j].squeeze()))
            
print(f"AVERAGE TEST: BATCH [{batch_idx} / {num_batch_test}] | LOSS [{ np.mean(loss_arr) }]")
