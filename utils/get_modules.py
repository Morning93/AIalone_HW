from torchvision.transforms import Resize
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from torch.utils.data import random_split
import torch
from torch.utils.data import DataLoader
from PIL import Image

def get_transform(args):
    transform = Compose([Resize((args.image_size, 
                             args.image_size)), 
                     ToTensor()])
    return transform

def get_dataset(args):
    transform = get_transform(args)
    train_val_dataset = MNIST(root='../../data', train=True, download=True, transform=transform)
    train_dataset, val_dataset = random_split(train_val_dataset, 
                                            [50000, 10000], 
                                            torch.Generator().manual_seed(42))
    test_dataset = MNIST(root='../../data', train=False, download=True, transform=transform)
    return train_dataset, val_dataset, test_dataset

def get_dataloader(args):
    transform = get_transform(args)
    train_dataset, val_dataset, test_dataset = get_dataset(args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader

def get_model(args):
    from networks.mlp import myMLP
    model = myMLP(image_size = args.image_size, 
                  hidden_size = args.hidden_size, 
                  num_class = args.num_class).to(args.device)
    return model

def prepare_image(args):
    image = Image.open(args.target_image_path)
    image = image.convert('L')

    return image