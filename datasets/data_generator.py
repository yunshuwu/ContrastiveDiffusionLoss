import torch as t
import torchvision
import torch.nn as nn
import torch.distributions as td
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random

from sklearn.datasets import make_moons
import seaborn as sns
sns.set_style('white')

# from gmm import GMMDist
from datasets.gmm import GMMDist


# Yunshu: not sure if this mixture of gaussian dataset generator will stay here in the future, 
# Hmmm..... feels like it's not a good place to have this function....
def mix_gauss_generator(gauss_dim=2,
                        train_nsamples=100000, 
                        val_nsamples=100000,
                        batch_size=1200
                        ):
    '''
    Args: 
        train_nsamples: number of samples to be generated from mix-gauss distribution
        val_nsamples: number of samples to be generated from mix-gauss distribution
        gauss_dim: dimension
        batch_size: the larger, the better!!!
    '''
    mix_gauss = GMMDist(dim=gauss_dim)
    
    # training and validation samples
    t.manual_seed(11) 
    train_samples = mix_gauss.sample((train_nsamples,))
    t.manual_seed(42) 
    val_samples = mix_gauss.sample((val_nsamples,))
    
    # sample size
    x_shape = train_samples.shape[1:]
    print(f'sample size={x_shape}')

    train_dl = DataLoader(TensorDataset(train_samples), batch_size=batch_size, num_workers=32, persistent_workers=True)
    val_dl = DataLoader(TensorDataset(val_samples), batch_size=batch_size, num_workers=32, persistent_workers=True)

    return train_dl, val_dl, x_shape, mix_gauss

#------------------------------------------------------------------------------------------ 
def CIFAR10_dataloader(batch_size, data_fp='./datasets/CIFAR10'):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x * 2. - 1.)  # Default is [0,1] range, U-Net expects [-1,1]
    ])
    train_ds = torchvision.datasets.CIFAR10(root=data_fp, train=True, transform=transform, download=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_ds = torchvision.datasets.CIFAR10(root=data_fp, train=False, transform=transform, download=True) # only test data, for inference debugging
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    CIFAR_shape = (3, 32, 32)

    return train_loader, train_ds, test_loader, test_ds, CIFAR_shape


# NOTE: this fake CIFAR10 dataset was created for self-defined manifold metric, which is not in teh final paper (since flow generated images are not close to the manifold, according to the DINO exp)
def FakeCIFAR_dataloader(config, data_fp='../datasets/'):
    '''
    Because the whole Fake CIFAR10 is too large for our server with 48GB, 
    I generate 50000 training data samples in 50 rounds, each round generate 1000 data samples
    I generate 10000 testing data samples in 10 rounds, each round generate 1000 data samples
    The diffusionmodel.py assumes the data batch is a list, batch[0]==data/image and batch[1]==y
    Args: 
        config: configeration 

    Return: 
        train_dataloader, test_dataloader
        CIFAR_shape: the shape of FakeCIFAR10 data samples
        TODO: also return ground truth nll and score
    '''
    def custom_collate(batch):
        x = t.stack([item[0] for item in batch], dim=0)
        return x, None

    # 1. training dataloader
    train_tensors_list = []
    for i in range(50):
        file_path = os.path.join(data_fp, 'FakeCIFAR/train_fakeCIFAR10/train_FakeCIFAR_{i}.pt'.format(i=i))
        # train_tensor_i = t.load('../datasets/FakeCIFAR/train_fakeCIFAR10/train_FakeCIFAR_{i}.pt'.format(i=i))
        train_tensor_i = t.load(file_path)
        train_tensors_list.append(train_tensor_i)
    train_CIFAR10_tensor = t.cat(train_tensors_list, dim=0) # pixels in [0, 255]
    # scaled such that all x lies in between [-1, 1], since U-Net expects [-1,1]
    n_bits = 8
    n_bits = 2 ** n_bits # should be 256
    train_CIFAR10_tensor = (train_CIFAR10_tensor / n_bits) * 2. - 1.

    range_checker = (train_CIFAR10_tensor >= -1.) & (train_CIFAR10_tensor <= 1.)
    if range_checker.all():
        print(f'Train data preprocessing, all pixels in range [-1, 1]')

    # y_train = t.zeros_like(train_CIFAR10_tensor)
    # save memory
    del train_tensors_list
    # train dataloader, each batch should be a tuple, batch[0]=data/image, batch[1]=None (no y given here)
    train_dl = DataLoader(TensorDataset(train_CIFAR10_tensor), 
                        #   TensorDataset(train_CIFAR10_tensor, y_train), 
                          batch_size=config.training.batch_size, 
                          num_workers=config.training.num_cpu_works, 
                          persistent_workers=True, 
                          collate_fn=custom_collate)

    # 2. testing dataloader
    test_tensors_list = []
    for i in range(10):
        file_path = os.path.join(data_fp, 'FakeCIFAR/test_fakeCIFAR10/test_FakeCIFAR_{i}.pt'.format(i=i))
        # test_tensor_i = t.load('../datasets/FakeCIFAR/test_fakeCIFAR10/test_FakeCIFAR_{i}.pt'.format(i=i))
        test_tensor_i = t.load(file_path)
        test_tensors_list.append(test_tensor_i)
    test_CIFAR10_tensor = t.cat(test_tensors_list, dim=0)

    # scaled such that all x lies in between [-1, 1], since U-Net expects [-1,1]
    test_CIFAR10_tensor = (test_CIFAR10_tensor / n_bits) * 2. - 1.

    range_checker = (test_CIFAR10_tensor >= -1.) & (test_CIFAR10_tensor <= 1.)
    if range_checker.all():
        print(f'Test preprocessing, all pixels in range [-1, 1]')

    # save memory
    del test_tensors_list
    test_dl = DataLoader(TensorDataset(test_CIFAR10_tensor), 
                        #  TensorDataset(test_CIFAR10_tensor, y_test), 
                         batch_size=config.training.batch_size, 
                         num_workers=config.training.num_cpu_works, 
                         persistent_workers=True, 
                         collate_fn=custom_collate)

    # 3. data shape, used by diffusion_model class
    CIFAR_shape = (3, 32, 32)

    return train_dl, test_dl, CIFAR_shape

#------------------------------------------------------------------------------------------
def line_dataloader(n=8000, batch_size=1200):
    # This ends up looking like a square because we standardize the data

    # training dataloader
    rng = np.random.default_rng(6)
    x = rng.uniform(-0.5, 0.5, n)
    y = rng.uniform(-1, 1, n)
    X = np.stack((x, y), axis=1)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)

    data_tensor = t.from_numpy(X.astype(np.float32))
    x_shape = data_tensor.shape[1:]
    train_ds = TensorDataset(data_tensor)
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=32, persistent_workers=True)
    
    # validation dataloader
    rng = np.random.default_rng(16)
    x = rng.uniform(-0.5, 0.5, n)
    y = rng.uniform(-1, 1, n)
    X = np.stack((x, y), axis=1)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)

    data_tensor = t.from_numpy(X.astype(np.float32))
    x_shape = data_tensor.shape[1:]
    val_ds = TensorDataset(data_tensor)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=32, persistent_workers=True)

    return train_dl, val_dl, x_shape, train_ds


def dino_dataset(n=8000, batch_size=100, path_2d="assets/DatasaurusDozen.tsv"):
    df = pd.read_csv(path_2d, sep="\t")
    df = df[df["dataset"] == "dino"]

    def sample(rng):
        ix = rng.integers(0, len(df), n)
        x = df["x"].iloc[ix].tolist()
        x = np.array(x) + rng.normal(size=len(x)) * 0.15
        y = df["y"].iloc[ix].tolist()
        y = np.array(y) + rng.normal(size=len(x)) * 0.15
        x = (x/54 - 1) * 4
        y = (y/48 - 1) * 4
        X = np.stack((x, y), axis=1)
        X -= X.mean(axis=0)
        X /= X.std(axis=0)
        return X

    # training data
    rng_train = np.random.default_rng(42)
    X = sample(rng_train)
    data_tensor = t.from_numpy(X.astype(np.float32))
    x_shape = data_tensor.shape[1:]
    train_ds = TensorDataset(data_tensor)
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=32, persistent_workers=True)

    # validation data
    rng_val = np.random.default_rng(6)
    X = sample(rng_val)
    data_tensor = t.from_numpy(X.astype(np.float32))
    # x_shape = data_tensor.shape[1:]
    val_ds = TensorDataset(data_tensor)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=32, persistent_workers=True)

    return train_dl, val_dl, x_shape, train_ds # return train_ds to plot the dataset


def dino(n=2 ** 9, seed=42, path_2d="assets/DatasaurusDozen.tsv"):
    """Need to give different rng every time while training."""
    df = pd.read_csv(path_2d, sep="\t")
    df = df[df["dataset"] == "dino"]

    rng = np.random.default_rng(seed)

    ix = rng.integers(0, len(df), n)
    x = df["x"].iloc[ix].tolist()
    x = np.array(x) + rng.normal(size=len(x)) * 0.15
    y = df["y"].iloc[ix].tolist()
    y = np.array(y) + rng.normal(size=len(x)) * 0.15
    x = (x/54 - 1) * 4
    y = (y/48 - 1) * 4
    X = np.stack((x, y), axis=1)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    return X


def moons_dataset(n=8000, batch_size=100):
    def sample(rng=42):
        X, _ = make_moons(n_samples=n, random_state=rng, noise=0.03)
        X[:, 0] = (X[:, 0] + 0.3) * 2 - 1
        X[:, 1] = (X[:, 1] + 0.3) * 3 - 1
        X -= X.mean(axis=0)
        X /= X.std(axis=0)
        return X

    # training data
    X = sample(rng=42) 
    data_tensor = t.from_numpy(X.astype(np.float32))
    x_shape = data_tensor.shape[1:]
    train_ds = TensorDataset(data_tensor)
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=32, persistent_workers=True)

    # validation data
    X = sample(rng=6)
    data_tensor = t.from_numpy(X.astype(np.float32))
    val_ds = TensorDataset(data_tensor)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=32, persistent_workers=True)
    
    return train_dl, val_dl, x_shape, train_ds # return train_ds to plot the dataset


def moons(n=2 ** 9, seed=42):
    """Need to give different rng every time while training."""
    rng = random.randint(0, 4294967295) # np.random.default_rng(seed)

    X, _ = make_moons(n_samples=n, random_state=rng, noise=0.03)
    X[:, 0] = (X[:, 0] + 0.3) * 2 - 1
    X[:, 1] = (X[:, 1] + 0.3) * 3 - 1
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    return X


def circle_dataset(n=8000, batch_size=1200):
    def sample(seed):
        rng = np.random.default_rng(seed)
        x = np.round(rng.uniform(-0.5, 0.5, n)/2, 1)*2
        y = np.round(rng.uniform(-0.5, 0.5, n)/2, 1)*2
        norm = np.sqrt(x**2 + y**2) + 1e-10
        x /= norm
        y /= norm
        theta = 2 * np.pi * rng.uniform(0, 1, n)
        r = rng.uniform(0, 0.03, n)
        x += r * np.cos(theta)
        y += r * np.sin(theta)
        X = np.stack((x, y), axis=1)
        X -= X.mean(axis=0)
        X /= X.std(axis=0)
        data_tensor = t.from_numpy(X.astype(np.float32))
        return data_tensor

    # training dataloader
    data_tensor = sample(6)
    # print(f'training: data_tensor.shape = {data_tensor.shape}, data_tensor={data_tensor}')
    x_shape = data_tensor.shape[1:]
    train_ds = TensorDataset(data_tensor)
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=32, persistent_workers=True)

    # val dataloader
    data_tensor = sample(16)
    val_ds = TensorDataset(data_tensor)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=32, persistent_workers=True)

    return train_dl, val_dl, x_shape, train_ds


def circle(n=2 ** 9, seed=42):
    rng = np.random.default_rng(seed)
    
    x = np.round(rng.uniform(-0.5, 0.5, n)/2, 1)*2
    y = np.round(rng.uniform(-0.5, 0.5, n)/2, 1)*2
    norm = np.sqrt(x**2 + y**2) + 1e-10
    x /= norm
    y /= norm
    theta = 2 * np.pi * rng.uniform(0, 1, n)
    r = rng.uniform(0, 0.03, n)
    x += r * np.cos(theta)
    y += r * np.sin(theta)
    X = np.stack((x, y), axis=1)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    return X.astype(np.float32)


def eight_gaussians_dataset(n=8000, batch_size=100):
    def sample(seed=42):
        rng = np.random.default_rng(seed)
        scale = 4.0
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        ]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(n):
            point = rng.normal(loc=0, scale=0.5, size=2) * 0.5
            # print(f'point.shape={point.shape}, point={point}')
            idx = rng.integers(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset

    # training dataloader
    data_tensor = t.from_numpy(sample(6))
    # print(f'training: data_tensor.shape = {data_tensor.shape}, data_tensor={data_tensor}')
    x_shape = data_tensor.shape[1:]
    train_ds = TensorDataset(data_tensor)
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=32, persistent_workers=True)

    # val dataloader
    data_tensor = t.from_numpy(sample(16))
    # print(f'validation: data_tensor.shape = {data_tensor.shape}, data_tensor={data_tensor}')
    val_ds = TensorDataset(data_tensor)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=32, persistent_workers=True)

    return train_dl, val_dl, x_shape, train_ds


def eight_gaussians(n=2 ** 9, seed=42):
    rng = np.random.default_rng(seed)

    scale = 4.0
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
    ]
    centers = [(scale * x, scale * y) for x, y in centers]

    dataset = []
    for i in range(n):
        point = rng.normal(loc=0, scale=0.5, size=2) * 0.5
        # print(f'point.shape={point.shape}, point={point}')
        idx = rng.integers(8)
        center = centers[idx]
        point[0] += center[0]
        point[1] += center[1]
        dataset.append(point)
    dataset = np.array(dataset, dtype="float32")
    dataset /= 1.414
    return dataset


def two_spirals_dataset(num_points=8000, batch_size=100):
    def sample():
        n = np.sqrt(np.random.rand(num_points // 2, 1)) * 540 * (2 * np.pi) / 360 # Yunshu: here n is the degree
        d1x = -np.cos(n) * n + np.random.rand(num_points // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(num_points // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        return x
    
    # training dataloader
    data_tensor = t.from_numpy(sample())
    # print(f'training: data_tensor.shape = {data_tensor.shape}, data_tensor={data_tensor}')
    x_shape = data_tensor.shape[1:]
    train_ds = TensorDataset(data_tensor)
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=32, persistent_workers=True)

    # val dataloader
    data_tensor = t.from_numpy(sample())
    # print(f'validation: data_tensor.shape = {data_tensor.shape}, data_tensor={data_tensor}')
    val_ds = TensorDataset(data_tensor)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=32, persistent_workers=True)

    return train_dl, val_dl, x_shape, train_ds 


def two_spirals(num_points=2 ** 9):
    n = np.sqrt(np.random.rand(num_points // 2, 1)) * 540 * (2 * np.pi) / 360 # Yunshu: here n is the degree
    d1x = -np.cos(n) * n + np.random.rand(num_points // 2, 1) * 0.5
    d1y = np.sin(n) * n + np.random.rand(num_points // 2, 1) * 0.5
    x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
    x += np.random.randn(*x.shape) * 0.1
    return x


def checkerboard_dataset(num_points=8000, batch_size=100):
    def sample():
        x1 = np.random.rand(num_points) * 4 - 2
        x2_ = np.random.rand(num_points) - np.random.randint(0, 2, num_points) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        return np.concatenate([x1[:, None], x2[:, None]], 1) * 2

    # training dataloader
    data_tensor = t.from_numpy(sample())
    # print(f'training: data_tensor.shape = {data_tensor.shape}, data_tensor={data_tensor}')
    x_shape = data_tensor.shape[1:]
    train_ds = TensorDataset(data_tensor)
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=32, persistent_workers=True)

    # val dataloader
    data_tensor = t.from_numpy(sample())
    # print(f'validation: data_tensor.shape = {data_tensor.shape}, data_tensor={data_tensor}')
    val_ds = TensorDataset(data_tensor)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=32, persistent_workers=True)

    return train_dl, val_dl, x_shape, train_ds 


def checkerboard(num_points=2 ** 9):
    x1 = np.random.rand(num_points) * 4 - 2
    x2_ = np.random.rand(num_points) - np.random.randint(0, 2, num_points) * 2
    x2 = x2_ + (np.floor(x1) % 2)
    return np.concatenate([x1[:, None], x2[:, None]], 1) * 2


#------------------------------------------------------------------------------------------
# Following code from: https://github.com/conormdurkan/autoregressive-energy-machines 
from skimage import color, io as imageio, transform
from torch.utils import data

class InfiniteLoader:
    """A data loader that can load a dataset repeatedly."""

    def __init__(self, dataset, batch_size=1, shuffle=True,
                 drop_last=True, num_epochs=None):
        """Constructor.
        Args:
            dataset: A `Dataset` object to be loaded.
            batch_size: int, the size of each batch.
            shuffle: bool, whether to shuffle the dataset after each epoch.
            drop_last: bool, whether to drop last batch if its size is less than
                `batch_size`.
            num_epochs: int or None, number of epochs to iterate over the dataset.
                If None, defaults to infinity.
        """
        self.loader = data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last, 
            num_workers=32, 
            persistent_workers=True
        )
        self.finite_iterable = iter(self.loader)
        self.counter = 0
        self.num_epochs = float('inf') if num_epochs is None else num_epochs

    def __next__(self):
        try:
            return next(self.finite_iterable)
        except StopIteration:
            self.counter += 1
            if self.counter >= self.num_epochs:
                raise StopIteration
            self.finite_iterable = iter(self.loader)
            return next(self.finite_iterable)

    def __iter__(self):
        return self
    

def create_einstein_data(n, face='einstein'):
    root = "../datasets/figs/"
    path = os.path.join(root, face + '.jpg')
    image = imageio.imread(path)
    image = color.rgb2gray(image)
    image = transform.resize(image, (512, 512))

    grid = np.array([
        (x, y) for x in range(image.shape[0]) for y in range(image.shape[1])
    ])

    rotation_matrix = np.array([
        [0, -1],
        [1, 0]
    ])
    p = image.reshape(-1) / sum(image.reshape(-1))
    ix = np.random.choice(range(len(grid)), size=n, replace=True, p=p)
    points = grid[ix].astype(np.float32)
    points += np.random.rand(n, 2)  # dequantize
    points /= (image.shape[0])  # scale to [0, 1]

    data = (points @ rotation_matrix).astype(np.float32)
    data[:, 1] += 1
    return data


from torch.utils.data import Dataset

class PlaneDataset(Dataset):
    def __init__(self, n):
        self.n = n
        self.data = None
        self.create_data()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.n

    def reset(self):
        self.create_data()

    def create_data(self):
        raise NotImplementedError

class FaceDataset(PlaneDataset):
    def __init__(self, n, face='einstein'):
        self.face = face
        self.image = None
        super().__init__(n)

    def create_data(self):
        self.data = create_einstein_data(self.n, self.face)

    def __getitem__(self, idx):
        data_point = self.data[idx]
        label = t.empty(0)
        return data_point, label


#------------------------------------------------------------------------------------------
def plot_samples(true_dist, 
                 nsamples=100,  
                 savefig=r'./figs/sampling_test/mix_Gauss',
                 left_bound=-8, right_bound=8):
    '''
    Yunshu: function is copied from Yang Song's NCSN repo, modified to plot my plots
    Args:
        true_dist: true distribution
    '''
    sns.set(font_scale=1.3)
    sns.set_style('white')
    savefig = savefig
    
    # ready for calling dynamics
    true_samples = true_dist.sample((nsamples,))


    # output true samples from ground-truth distribution
    samples = true_samples
    samples = samples.detach().cpu().numpy()
    plt.scatter(samples[:, 0], samples[:, 1], s=0.2)
    plt.axis('square')
    plt.title('i.i.d samples')
    plt.xlim([left_bound, right_bound])
    plt.ylim([left_bound, right_bound])
    if savefig is not None:
        plt.savefig(savefig + "/iid_samples.png", bbox_inches='tight')
        plt.show()
    else:
        plt.show()