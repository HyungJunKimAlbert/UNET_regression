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
# parser
import argparse
# Customized function
from model import UNet
from dataset import ImageDataset, ToTensor, RandomFlip, Normalization
from utils import save, load


# Create Parser 
parser = argparse.ArgumentParser(description="Train the UNet", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default="./datasets", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./results", type=str, dest="result_dir")
# FLAG
parser.add_argument("--mode", default="train", type=str, dest="mode")
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")

# Parser object
args = parser.parse_args()

# Training Parameters
lr = args.lr
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir

mode = args.mode
train_continue = args.train_continue

# define  device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print(f"Learning Rate: { lr }")
print(f"Batch size: { BATCH_SIZE }")
print(f"NUM_EPOCHS: { NUM_EPOCHS }")

print(f"data_dir: { data_dir }")
print(f"ckpt_dir: { ckpt_dir }")
print(f"log_dir: { log_dir }")
print(f"result_dir: { result_dir }")

print(f"Mode: { mode }")
print(f"Train_continue: { train_continue }")


# create result directory
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

if mode == 'train':
    # Transform
    transform = transforms.Compose([
        Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()
    ])
    dataset_train = ImageDataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
    loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)

    dataset_val = ImageDataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
    loader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)

    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)
    num_batch_train = int(np.ceil(num_data_train / BATCH_SIZE))
    num_batch_val = int(np.ceil(num_data_val / BATCH_SIZE))
else:
    transform = transforms.Compose([
        Normalization(mean=0.5, std=0.5), ToTensor()
    ])

    dataset_test = ImageDataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
    loader_test = DataLoader(dataset_test , batch_size=BATCH_SIZE, shuffle=False)
    
    num_data_test = len(dataset_test)
    num_batch_test = int(np.ceil(num_data_test / BATCH_SIZE))

# Network
net = UNet().to(device)
# Loss function
fn_loss = nn.BCEWithLogitsLoss().to(device)
# Optimizer
optim = torch.optim.Adam(net.parameters(), lr=lr)
# etc function
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x*std) + mean
fn_class = lambda x: 1.0 * (x>0.5)
# Tensorboard
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))


if mode == 'train': # TRAIN MODE
    ST_EPOCH = 0

    if train_continue == "on": # load weights
        net, optim, ST_EPOCH = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    # Network Training
    for epoch in range(ST_EPOCH+1, NUM_EPOCHS+1):
        net.train()
        loss_arr = []

        for batch_idx, data in enumerate(loader_train, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)
            output = net(input)
            # backward pass
            optim.zero_grad()
            loss = fn_loss(output, label)
            loss.backward()
            optim.step()

            # Loss
            loss_arr += [loss.item()]
            print(f'TRAIN: EPOCH [{epoch} / {NUM_EPOCHS}] | BATCH [{batch_idx} / {num_batch_train}] | LOSS [{loss}]')

            # To numpy
            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))
            # Tensorboard
            writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch_idx, dataformats='NHWC')
            writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch_idx, dataformats='NHWC')
            writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch_idx, dataformats='NHWC')

        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

        with torch.no_grad():
            net.eval()
            loss_arr = []

            for batch_idx, data in enumerate(loader_val, 1):
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

                print(f"VALID: EPOCH [{epoch} / {NUM_EPOCHS}] | BATCH [{batch_idx} / {num_batch_val}] | LOSS [{loss}]")

                # Tensorboard
                writer_val.add_image('label', label, num_batch_val * (epoch-1) + batch_idx, dataformats='NHWC')
                writer_val.add_image('input', input, num_batch_val * (epoch-1) + batch_idx, dataformats='NHWC')
                writer_val.add_image('output', output, num_batch_val * (epoch-1) + batch_idx, dataformats='NHWC')
        writer_val.add_scalar('loss', np.mean(loss_arr), epoch)
        
        if epoch % 50 == 0:
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)
    # Tensorboard close
    writer_train.close()
    writer_val.close()

else:   # TEST MODE
    net, optim, ST_EPOCH = load(ckpt_dir=ckpt_dir, net=net, optim=optim)    # load weights
    
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
                plt.imsave(os.path.join(result_dir, "png", f"label_{id}.png"), label[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, "png", f"input_{id}.png"), input[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, "png", f"output_{id}.png"), output[j].squeeze(), cmap='gray')
                # save (numpy format)
                np.save(os.path.join(result_dir, "numpy", f"label_{id}.npy"), label[j].squeeze())
                np.save(os.path.join(result_dir, "numpy", f"input_{id}.npy"), input[j].squeeze())
                np.save(os.path.join(result_dir, "numpy", f"output_{id}.npy"), output[j].squeeze())
                
    print(f"AVERAGE TEST: BATCH [{batch_idx} / {num_batch_test}] | LOSS [{ np.mean(loss_arr) }]")





# #%% test
# transform = transforms.Compose([
#     Normalization(mean=0.5, std=0.5),
#     RandomFlip(), 
#     ToTensor()
# ])

# datatset_train = ImageDataset(data_dir='./dataset/train', transform=transform)

# #%% check sample 
# data = datatset_train.__getitem__(0)
# input = data['input']
# label = data['label']


# print('SHAPE')
# print(f"Input: {input.shape}, Label: {label.shape}")

# print('tpye')
# print(f"Input: {input.type()}, Label: {label.type()}")


# plt.subplot(121)
# plt.imshow(input.squeeze())

# plt.subplot(122)
# plt.imshow(label.squeeze())