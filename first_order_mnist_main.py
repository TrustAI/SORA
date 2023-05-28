
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
import numpy as np

import os
import argparse
import time

from geo_transf_verifications import obstacle_bound,reachability_loss
from mnist_utils import load_mnist, load_model, load_resized_mnist

upper_limit = 1.
lower_limit = 0.

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def attack_pgd_masked(model, X, y, mask, epsilon, 
               alpha, attack_iters, restarts, device, use_CWloss=False):
    mask = mask.to(device)
    y = torch.tensor(y).view(-1).to(device)
    max_loss = torch.zeros(y.shape[0]).to(device)
    max_delta = torch.zeros_like(X).to(device)
    for zz in range(restarts):
        delta = torch.zeros_like(X).to(device)
        delta.uniform_(-epsilon, epsilon)
        delta = delta * mask
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            if use_CWloss:
                loss = CW_loss(output, y)
            else:
                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            # print(grad.shape)
            d = delta.data
            g = grad.data

            d = torch.clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = d * mask
            d = clamp(d, lower_limit - X, upper_limit - X)
            delta.data = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def CW_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    
    loss_value = -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind))
    return loss_value.mean()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--example-idx', default=506, type=int)
    parser.add_argument('--data-dir', default='/mnt/storage0_8/torch_datasets/mnist-data/', type=str)
    parser.add_argument('--model-dir', default='./model', type=str)
    parser.add_argument('--model-name', default='onnx_mnist_256x2', type=str)
    parser.add_argument('--device', default='cuda', type=str)

    # Transformation
    parser.add_argument('--l-inf-bound', default=0.29999, type=float)
    parser.add_argument('--topleft-x', default=0, type=int)
    parser.add_argument('--topleft-y', default=0, type=int)
    parser.add_argument('--width', default=0, type=int)
    parser.add_argument('--height', default=0, type=int)
    parser.add_argument('--resize', action='store_true')
    parser.add_argument('--size', default=7, type=int)

    # PGD
    parser.add_argument('--max-iteration', default=100, type=int)
    parser.add_argument('--step-size', default=0.1, type=float)
    parser.add_argument('--restart', default=5, type=int)
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()

def main():
    args = get_args()
    if args.resize:
        _, test_loader = load_resized_mnist(args.data_dir, 100, 100, resize=args.size)
    else:
        _, test_loader = load_mnist(args.data_dir,100,100)
    img, label = test_loader.dataset.__getitem__(args.example_idx)
    img = img.to(args.device)
    data_size = tuple(img.shape)

    model = load_model(args.model_dir, args.model_name, args.device)
    ori_out = model(img.unsqueeze(0))
    ori_conf = reachability_loss(ori_out, label).item()
    if torch.argmax(ori_out).item() == label:
        correctness = 1
    else:
        correctness = 0

    coefs = []
    for i in range(args.width):
        for j in range(args.height):
            coefs.append((args.topleft_y + j, args.topleft_x + i))
    mask = torch.zeros_like(img.squeeze())

    for co in coefs:
        mask[co] = 1.
    start_time = time.time()
    opt_patch = attack_pgd_masked(model, img.unsqueeze(0), label, mask, args.l_inf_bound, args.step_size, args.max_iteration, args.restart, args.device, True)
    end_time = time.time()
    opt_out = model(opt_patch + img)
    if torch.argmax(opt_out).item() == label:
        post_correctness = 1
    else:
        post_correctness = 0

    print(f'{args.example_idx},{correctness},{ori_conf:.6f},{post_correctness},{opt_out[0][label].item()},{(end_time - start_time):.2f}')
    print([opt_patch.cpu().squeeze()[co].item() for co in sorted(coefs, key=lambda x: x[0])])
    if args.debug:
        print(opt_out.data[0])
    


if __name__ == "__main__":
    main()