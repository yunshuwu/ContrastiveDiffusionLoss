# NOTE: this file is for manifold metric, which is defined to measure the ability of learning density/score
# By sampling from the flow model, we are able to get the ground-truth score/density. 
# Why not used in the final paper? Since the flow generated images/2d data-samples are far away from data manifold, so cannot well represent real-world data distributions. 

import argparse
import shutil
import json

import torch
import numpy as np
import normflows as nf
from sklearn.datasets import make_moons
from matplotlib import pyplot as plt
import random
from functools import partial

from tqdm import tqdm

# import sys
# sys.path.append('.')
# from data_generator import *
from datasets.data_generator import *


def train(dataset, output_dir):
    """Train flow model."""
    # 1. Define flow model
    K = 16
    torch.manual_seed(0)

    latent_size = 2
    hidden_units = 128
    hidden_layers = 2

    flows = []
    for i in range(K):
        flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units)]
        flows += [nf.flows.LULinearPermute(latent_size)]

    # set base distribution
    q0 = nf.distributions.DiagGaussian(2, trainable=False)

    # construct flow
    nfm = nf.NormalizingFlow(q0=q0, flows=flows)

    # move model on GPU
    enable_cuda = True
    device = torch.device('cuda:0' if torch.cuda.is_available() and enable_cuda else 'cpu')
    nfm = nfm.to(device)

    # 2. Plot data
    # plot the training data
    if dataset == "dino":
        sample_fn = partial(dino, path_2d="./assets/DatasaurusDozen.tsv")
    elif dataset == "moons":
        sample_fn = moons
    elif dataset == "eight_gaussian":
        sample_fn = eight_gaussians
    elif dataset == "spirals": # no seed needed
        sample_fn = two_spirals
    elif dataset == "checkerboard": # no seed needed
        sample_fn = checkerboard
    elif dataset == "circle": # need seed
        sample_fn = circle

    print(f"Plotting ground truth data...")
    x_np = sample_fn(2 ** 20)
    print(f'{dataset} type = {type(x_np)}, shape = {x_np.shape}')
    plt.figure(figsize=(15, 15))
    plt.hist2d(x_np[:, 0], x_np[:, 1], bins=200, cmap='viridis')
    plt.savefig("./figs/{}_gt.pdf".format(dataset))

    # plot the initial flow distribution, standard gaussian
    # make the grid
    grid_size = 100
    xx, yy = torch.meshgrid(torch.linspace(-3.5, 3.5, grid_size), torch.linspace(-3.5, 3.5, grid_size))
    zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
    zz = zz.to(device)

    nfm.eval()
    log_prob = nfm.log_prob(zz).to('cpu').view(*xx.shape) # reshape log-prob to xx's shape
    nfm.train()
    prob = torch.exp(log_prob)
    prob[torch.isnan(prob)] = 0 # all points on grid which are not having values, should be assigned prob=0

    plt.figure(figsize=(15, 15))
    plt.pcolormesh(xx, yy, prob.data.numpy(), cmap='viridis')
    plt.gca().set_aspect('equal', 'box')
    plt.savefig("./figs/{}_init.pdf".format(dataset))

    # 3. Train
    max_iter = 10000
    num_samples = 2 ** 9 # per iteration
    show_iter = 500

    loss_hist = np.array([])

    optimizer = torch.optim.Adam(nfm.parameters(), lr=1e-3, weight_decay=1e-5)
    for it in tqdm(range(max_iter)):
        optimizer.zero_grad()

        # get training samples
        if dataset == "dino" or dataset == "moons" or dataset == "eight_gaussian" or dataset == "circle":
            rand_seed = random.randint(1, max_iter)
            x_np = sample_fn(num_samples, seed=rand_seed)
        elif dataset == "spirals" or dataset == "checkerboard":
            x_np = sample_fn(num_samples)
        x = torch.tensor(x_np).float().to(device)

        # calculate loss
        loss = nfm.forward_kld(x)

        # backprop and optimizer step
        if ~(torch.isnan(loss) | torch.isinf(loss)): # if loss is normal
            loss.backward()
            optimizer.step()
        
        # log loss
        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())

        # plot leaned distribution
        if (it + 1) % show_iter == 0:
            nfm.eval()
            log_prob = nfm.log_prob(zz) # calculate the log-prob on points zz
            nfm.train()
            prob = torch.exp(log_prob.to('cpu').view(*xx.shape))
            prob[torch.isnan(prob)] = 0 # assign prob=0 to those untrained points, in order to plot
            
            plt.figure(figsize=(15, 15))
            plt.pcolormesh(xx, yy, prob.data.numpy(), cmap='viridis')
            plt.gca().set_aspect('equal', 'box')
    # plot the loss
    plt.figure(figsize=(10, 10))
    plt.plot(loss_hist, label="loss")
    plt.savefig("./figs/{}_loss.pdf".format(dataset))

    # Save the checkpoint
    torch.save(nfm.state_dict(), f"../datasets/{output_dir}/{dataset}_{it}")

    # 4. Sanity check the generated data
    # plot the learned distribution
    nfm.eval()
    log_prob = nfm.log_prob(zz).to('cpu').view(*xx.shape)
    nfm.train()
    prob = torch.exp(log_prob)
    prob[torch.isnan(prob)] = 0

    plt.figure(figsize=(15, 15))
    plt.pcolormesh(xx, yy, prob.data.numpy(), cmap='viridis')
    plt.gca().set_aspect('equal', 'box')
    plt.savefig("./figs/{}_est.pdf".format(dataset))



def sample_2d_synthetic(dataset, num_samples=100000, output_dir="./flow_synthetic_2d_checkpoints/"):
    # 1. Define flow model
    K = 16
    torch.manual_seed(0)

    latent_size = 2
    hidden_units = 128
    hidden_layers = 2

    flows = []
    for i in range(K):
        flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units)]
        flows += [nf.flows.LULinearPermute(latent_size)]

    # 2. Load pre-trained model
    # Set base distribution
    q0 = nf.distributions.DiagGaussian(2, trainable=False)

    # Construct flow and load pre-trained models
    nfm = nf.NormalizingFlow(q0=q0, flows=flows)
    if dataset == "dino":
        nfm.load_state_dict(torch.load("../datasets/flow_synthetic_2d_checkpoints/dino_9999"))
    elif dataset == "moons":
        nfm.load_state_dict(torch.load("../datasets/flow_synthetic_2d_checkpoints/moons_9999"))
    elif dataset == "eight_gaussian":
        nfm.load_state_dict(torch.load("../datasets/flow_synthetic_2d_checkpoints/eight_gaussian_9999"))
    elif dataset == "spirals": # no seed needed
        nfm.load_state_dict(torch.load("../datasets/flow_synthetic_2d_checkpoints/spirals_9999"))
    elif dataset == "checkerboard": # no seed needed
        nfm.load_state_dict(torch.load("../datasets/flow_synthetic_2d_checkpoints/checkerboard_9999"))
    elif dataset == "circle":
        nfm.load_state_dict(torch.load("../datasets/flow_synthetic_2d_checkpoints/circle_9999"))
    
    # # move model on GPU for generation, Yunshu: is this necessarily? stay on CPU also works
    # enable_cuda = True
    # device = torch.device('cuda:0' if torch.cuda.is_available() and enable_cuda else 'cpu')
    # nfm = nfm.to(device)    

    # 3. Generate training/validation datasets
    nfm.eval() # now nfm model is on CPU
    with torch.no_grad():
        train_tensor, _ = nfm.sample(num_samples)
        val_tensor, _ = nfm.sample(num_samples)
    x_shape = train_tensor.shape[1:] # take sample shape, the first dim is the number of samples
    train_ds = TensorDataset(train_tensor)
    val_ds = TensorDataset(val_tensor)
    
    return train_ds, val_ds, x_shape



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_or_sample", type=str, default="train", choices=["train", "sample"], help="Train flow model")
    parser.add_argument("--dataset", type=str, default="dino", choices=["dino", "spirals", "moons", "eight_gaussian", "checkerboard", "circle"], help="Type of the dataset to be used.")
    parser.add_argument("--output_dir", default="./flow_synthetic_2d_checkpoints/", help="Directory to output logs and model checkpoints")

    args = parser.parse_args()

    if args.train_or_sample == "train":
        kwargs = vars(args)
        del kwargs["train_or_sample"]
        with open(os.path.join(args.output_dir, f"{args.dataset}_hparams.json"), "w") as fp:
            json.dump(kwargs, fp, sort_keys=True, indent=4)

        train(**kwargs)
    else: # sample, generate from pre-trained models
        kwargs = vars(args)
        del kwargs["train_or_sample"]
        # No need to save sampling hparams
        # with open(os.path.join(args.output_dir, f"{args.dataset}_hparams.json"), "w") as fp:
        #     json.dump(kwargs, fp, sort_keys=True, indent=4)
        
        sample_2d_synthetic(**kwargs)