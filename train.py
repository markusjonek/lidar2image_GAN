import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import cv2

import time
import os

from model import *

import wandb


torch.manual_seed(10)
np.random.seed(10)


def load_data():
    x_file = 'X_data.npz'
    y_file = 'Y_data.npz'

    x_data = np.load(x_file)
    y_data = np.load(y_file)

    num_files = len(x_data.files)

    data = []

    intensity_threshold = 0.001

    total_points = 0

    random_indices = np.random.permutation(num_files)

    for i in tqdm(random_indices):
        unc = x_data[str(i)]
        c = y_data[str(i)]

        unc[:, :3] = unc[:, :3] / 75.0

        c = c[:, 3:]
        c = c / 255.0

        # remove points with low intensity
        inten = unc[:, 3]
        refl = inten * np.linalg.norm(unc[:, :3], axis=1)
        unc[:, 3] = refl

        unc = unc[inten > intensity_threshold]
        c = c[inten > intensity_threshold]

        unc = torch.from_numpy(unc)
        c = torch.from_numpy(c)

        total_points += unc.shape[0]

        data.append(Data(x=unc, y=c))


    print("Total points: {}".format(total_points))
    print("Avg points: {}".format(total_points / num_files))    

    return data


def visualize(x, y):
    y = y.detach().numpy()
    y = np.clip(y, 0, 1)

    x = x.detach().numpy() * 75.0

    # visualize with matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y, s=1)

    # set equal axis
    plt.axis('equal')
    plt.show()


def self_weighted_mse(y_pred, y_true):
    """
    y_pred: (N, 3) (b, g, r)
    y_true: (N, 3) (b, g, r)
    """
    b_mean = torch.mean(y_true[:, 0])
    g_mean = torch.mean(y_true[:, 1])
    r_mean = torch.mean(y_true[:, 2])

    b_diff = y_true[:, 0] - b_mean
    g_diff = y_true[:, 1] - g_mean
    r_diff = y_true[:, 2] - r_mean

    b_diff = torch.pow(b_diff, 2)
    g_diff = torch.pow(g_diff, 2)
    r_diff = torch.pow(r_diff, 2)

    b_diff = b_diff / torch.sum(b_diff)
    g_diff = g_diff / torch.sum(g_diff)
    r_diff = r_diff / torch.sum(r_diff)

    b_diff = torch.sum(b_diff * torch.pow(y_pred[:, 0] - y_true[:, 0], 2))
    g_diff = torch.sum(g_diff * torch.pow(y_pred[:, 1] - y_true[:, 1], 2))
    r_diff = torch.sum(r_diff * torch.pow(y_pred[:, 2] - y_true[:, 2], 2))

    return b_diff + g_diff + r_diff


def distance_weighted_mse(y_pred, y_true, xyz):
    """
    y_pred: (N, 3) (b, g, r)
    y_true: (N, 3) (b, g, r)
    xyz: (N, 3)
    """
    dist = torch.norm(xyz, dim=1)
    dist = 1 / dist
    dist = dist / torch.sum(dist)

    b_diff = torch.sum(dist * torch.pow(y_pred[:, 0] - y_true[:, 0], 2))
    g_diff = torch.sum(dist * torch.pow(y_pred[:, 1] - y_true[:, 1], 2))
    r_diff = torch.sum(dist * torch.pow(y_pred[:, 2] - y_true[:, 2], 2))

    return b_diff + g_diff + r_diff



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = load_data()

    train_data = data # [:int(len(data) * 0.9)]
    val_data = data # [int(len(data) * 0.9):]

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=True)

    model = PointNet()
    old_state_dict = torch.load('models/model_pointnet.pth', map_location=device)
    model.load_state_dict(old_state_dict)
    model.to(device)

    
    #visualize(train_data[0].x, train_data[0].y)

    # x_in = train_data[0].x
    # batch = torch.zeros(x_in.shape[0], dtype=torch.long)
    # out = model(x_in, batch)
    # visualize(x_in, out)


    # print number of parameters
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("Number of parameters: {}".format(num_params))

    learning_rate = 0.0005
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    wandb.init(
        project="KEX",
        config={
            "learning_rate": learning_rate,
            "dataset": "KITTI 25k",
            "architecture": "PointNet_attention",
            "dataset_size": len(data),
            "batch_size": 1,
            "loss_function": "MSE",
            "optimizer": "Adam",
            "num_params": num_params,
        }
    )

    print("Starting training")

    epochs = 1000
    count = 0

    mse_moving_avg_list = []
    self_moving_avg_list = []
    dist_moving_avg_list = []
    tot_moving_avg_list = []

    try:
        for epoch in range(epochs):
            for i, data in enumerate(train_loader):
                data = data.to(device)

                optimizer.zero_grad()
                out = model(data.x, data.batch)

                mse_loss = criterion(out, data.y)
                self_loss = self_weighted_mse(out, data.y)
                dist_loss = distance_weighted_mse(out, data.y, data.x[:, :3])

                tot_loss = mse_loss * 0.2 + self_loss * 0.05 + dist_loss

                #print("MSE Loss: {} | Self Loss: {} | Dist Loss {} | Tot Loss {}".format(mse_loss.item(), self_loss.item(), dist_loss.item(), tot_loss.item()))

                tot_loss.backward()
                optimizer.step()

                count += 1

                if len(mse_moving_avg_list) < 100:
                    mse_moving_avg_list.append(mse_loss.item())
                    self_moving_avg_list.append(self_loss.item())
                    dist_moving_avg_list.append(dist_loss.item())
                    tot_moving_avg_list.append(tot_loss.item())
                else:
                    mse_moving_avg_list.pop(0)
                    self_moving_avg_list.pop(0)
                    dist_moving_avg_list.pop(0)
                    tot_moving_avg_list.pop(0)
                    mse_moving_avg_list.append(mse_loss.item())
                    self_moving_avg_list.append(self_loss.item())
                    dist_moving_avg_list.append(dist_loss.item())
                    tot_moving_avg_list.append(tot_loss.item())
                    wandb.log({
                        "train-mse-loss": np.mean(mse_moving_avg_list),
                        "train-self_w_mse-loss": np.mean(self_moving_avg_list),
                        "train-dist_w_mse-loss": np.mean(dist_moving_avg_list),
                        "train-tot-loss": np.mean(tot_moving_avg_list)
                    }, step=count)

                    print("Epoch: {} | Batch: {} | MSE Loss: {} | RGB W MSE Loss: {} | Dist W MSE Loss {} | Tot Loss {}".format(epoch, i, np.mean(mse_moving_avg_list), np.mean(self_moving_avg_list), np.mean(dist_moving_avg_list), np.mean(tot_moving_avg_list)))


            if epoch % 1000 == 0:
                val_losses = []
                val_self_losses = []
                val_dist_losses = []
                val_tot_losses = []
                for val_data in val_loader:
                    with torch.no_grad():
                        val_data = val_data.to(device)
                        val_out = model(val_data.x, val_data.batch)
                        val_loss = criterion(val_out, val_data.y)
                        val_self_loss = self_weighted_mse(val_out, val_data.y)
                        val_dist_loss = distance_weighted_mse(val_out, val_data.y, val_data.x[:, :3])
                        val_tot_loss = val_loss * 0.2 + val_self_loss * 0.05 + val_dist_loss
                        val_losses.append(val_loss.item())
                        val_self_losses.append(val_self_loss.item())

                wandb.log({
                    "val-mse-loss": np.mean(val_losses),
                    "val-self_w_mse-loss": np.mean(val_self_losses),
                    "val-dist_w_mse-loss": np.mean(val_dist_losses),
                    "val-tot-loss": np.mean(val_tot_losses)
                }, step=count)
                #print("Epoch: {} | Val MSE Loss: {} | Val Self Loss: {}".format(epoch, np.mean(val_losses), np.mean(val_self_losses)))

            torch.save(model.state_dict(), 'models/model_pointnet.pth')
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'models/model_pointnet.pth')
        print("Saved model")


if __name__ == '__main__':
    main()


    