import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import Resize
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize

def get_transform(args): 
    if args.data == 'MNIST':
        transform = Compose([Resize((args.image_size, 
                                args.image_size)), 
                        ToTensor()])
    elif args.data == 'CIFAR':
        mean = [0.491, 0.482, 0.447]
        std = [0.247, 0.244, 0.262]
        transform = Compose([Resize((args.image_size,
                                     args.image_size)),
                                     ToTensor(),
                                     Normalize(mean, std)])
    return transform 

def get_datasets(transform, args): 
    if args.data == 'MNIST':
        from torchvision.datasets import MNIST
        train_val_dataset = MNIST(root='../../data', train=True, download=True, transform=transform)
        train_dataset, val_dataset = random_split(train_val_dataset, 
                                                [50000, 10000], 
                                                torch.Generator().manual_seed(42))
        test_dataset = MNIST(root='../../data', train=False, download=True, transform=transform)
    elif args.data == 'CIFAR':
        from torchvision.datasets import CIFAR10
        train_val_dataset = CIFAR10(root='../../data', train=True, download=True, transform=transform)
        train_dataset, val_dataset = random_split(train_val_dataset, 
                                                [40000, 10000], 
                                                torch.Generator().manual_seed(42))
        test_dataset = CIFAR10(root='../../data', train=False, download=True, transform=transform)
    return train_dataset, val_dataset, test_dataset

def get_dataloaders(args): 
    # 데이터 불러오기 
    # dataset 만들기 & 전처리하는 코드도 같이 작성 
    transform = get_transform(args)
    train_dataset, val_dataset, test_dataset = get_datasets(transform, args)
    # dataloader 만들기 
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader



def get_model(args): 
    if args.model == 'mlp':
        from networks.mlp import myMLP
        model = myMLP(args).to(args.device) 
    elif args.model == 'lenet':
        from networks.lenet import LeNet
        model = LeNet(args).to(args.device)
    elif args.model == 'lenet_inj':
        from networks.lenet import LeNet_inj
        model = LeNet_inj(args).to(args.device)
    elif args.model == 'lenet_multiconv':
        from networks.lenet import LeNet_multiconv
        model = LeNet_multiconv(args).to(args.device)
    elif args.model == 'lenet_incep':
        from networks.lenet import LeNet_incep
        model = LeNet_incep(args).to(args.device)
    elif args.model == 'lenet_nh':
        from networks.lenet import LeNet_nh
        model = LeNet_nh(args).to(args.device)
        
    return model 

def get_loss(): 
    return loss 

def get_optim():
    return optim 