import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import os


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def fc_model():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(784,10)
    )
    return model

def mnist_2layer_fc(img_size=7,num_hidden=10,activation='relu'):
    if activation == 'relu':
        act = nn.ReLU()
    elif activation == 'gelu':
        act = nn.GELU()
    elif activation == 'softplus':
        act = nn.Softplus()
    model = nn.Sequential(
        Flatten(),
        nn.Linear(img_size * img_size, num_hidden),
        act,
        nn.Linear(num_hidden, 10),
    )
    return model

def mnist_5layer_fc(img_size=7,num_hidden=100,activation='relu'):
    if activation == 'relu':
        act = nn.ReLU()
    elif activation == 'gelu':
        act = nn.GELU()
    elif activation == 'softplus':
        act = nn.Softplus()
    model = nn.Sequential(
        Flatten(),
        nn.Linear(img_size * img_size, num_hidden),
        act,
        nn.Linear(num_hidden, num_hidden),
        act,
        nn.Linear(num_hidden, num_hidden),
        act,
        nn.Linear(num_hidden, num_hidden),
        act,
        nn.Linear(num_hidden, 10),
    )
    return model

def resiz_mnist_net(num_hidden=64,activation='relu'):
    if activation == 'relu':
        act = nn.ReLU()
    elif activation == 'gelu':
        act = nn.GELU()
    elif activation == 'softplus':
        act = nn.Softplus()
    model = nn.Sequential(
        nn.Conv2d(1, 16, 3, stride=1, padding=1),
        act,
        nn.Conv2d(16, 32, 3, stride=2, padding=1),
        act,
        Flatten(),
        nn.Linear(32*4*4,num_hidden),
        act,
        nn.Linear(num_hidden, 10)
    )
    return model

def mnist_net():
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

def mnist_conv_big():
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*7*7,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model

def load_mnist(path, train_size, test_size):
    mnist_train = datasets.MNIST(path, train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=train_size, shuffle=True)
    mnist_test = datasets.MNIST(path, train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=test_size, shuffle=False)
    return train_loader, test_loader

def load_model(model_dir, model_name, device):
    ckpt = torch.load(os.path.join(model_dir,model_name+'.pth'))
    if model_name == 'fc_1':
        model = fc_model().to(device)
    elif model_name in ['lenet', 'adv_lenet']:
        model = mnist_net().to(device)
    elif '/model' in model_dir and model_name+'.pth' in os.listdir(model_dir):
        return ckpt.to(device)
    elif 'large' in model_name:
        if 'relu' in model_name:
            model = mnist_5layer_fc(img_size=7,num_hidden=100,activation='relu').to(device)
        if 'gelu' in model_name:
            model = mnist_5layer_fc(img_size=7,num_hidden=100,activation='gelu').to(device)
        if 'softplus' in model_name:
            model = mnist_5layer_fc(img_size=7,num_hidden=100,activation='softplus').to(device)
    elif 'conv' in model_name:
        if 'relu' in model_name:
            model = resiz_mnist_net(num_hidden=64,activation='relu').to(device)
        if 'gelu' in model_name:
            model = resiz_mnist_net(num_hidden=64,activation='gelu').to(device)
        if 'softplus' in model_name:
            model = resiz_mnist_net(num_hidden=64,activation='softplus').to(device)
    else:
        if 'relu' in model_name:
            model = mnist_2layer_fc(img_size=7,num_hidden=10,activation='relu').to(device)
        if 'gelu' in model_name:
            model = mnist_2layer_fc(img_size=7,num_hidden=10,activation='gelu').to(device)
        if 'softplus' in model_name:
            model = mnist_2layer_fc(img_size=7,num_hidden=10,activation='softplus').to(device)
    model.load_state_dict(ckpt)
    return model

def clean_evaluate(model, test_loader):
    total_loss = 0
    total_acc = 0
    n = 0
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = F.cross_entropy(output, y)
            total_loss += loss.item() * y.size(0)
            total_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)

    return total_loss/n, total_acc/n

def load_resized_mnist(path, train_size, test_size, resize=7):
    resize_transform = transforms.Compose([
        transforms.Resize(size=resize),
        transforms.ToTensor(),
    ])
    mnist_train = datasets.MNIST(path, train=True, download=True, transform=resize_transform)
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=train_size, shuffle=True)
    mnist_test = datasets.MNIST(path, train=False, download=True, transform=resize_transform)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=test_size, shuffle=False)
    return train_loader, test_loader