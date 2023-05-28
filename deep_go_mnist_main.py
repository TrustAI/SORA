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
from deep_go import nested_lip_opt, Recorder
from mnist_utils import load_mnist, load_model, load_resized_mnist


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--example-idx', default=506, type=int)
    parser.add_argument('--data-dir', default='/mnt/storage0_8/torch_datasets/mnist-data/', type=str)
    parser.add_argument('--model-dir', default='./model/', type=str)
    parser.add_argument('--model-name', default='onnx_mnist_256x2', type=str)
    parser.add_argument('--device', default='cuda', type=str)

    # Transformation
    parser.add_argument('--resize', action='store_true')
    parser.add_argument('--size', default=7, type=int)
    parser.add_argument('--l-inf-bound', default=0.3, type=float)
    parser.add_argument('--topleft-x', default=0, type=int)
    parser.add_argument('--topleft-y', default=0, type=int)
    parser.add_argument('--width', default=0, type=int)
    parser.add_argument('--height', default=0, type=int)

    # DeepGo
    parser.add_argument('--init-k', default=0.9, type=float)
    parser.add_argument('--max-evaluation', default=10000, type=int)
    parser.add_argument('--max-iteration', default=500, type=int)
    parser.add_argument('--bound-error', default=1e-4, type=float)
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()

def main():
    args = get_args()

    if args.resize:
        _, test_loader = load_resized_mnist(args.data_dir, 100, 100, resize=args.size)
    else:
        _, test_loader = load_mnist(args.data_dir,100,100)

    img, label = test_loader.dataset.__getitem__(args.example_idx)
    data_size = tuple(img.shape)

    model = load_model(args.model_dir, args.model_name, args.device)
    ori_out = model(img.unsqueeze(0).to(args.device))
    ori_conf = reachability_loss(ori_out, label).item()
    if torch.argmax(ori_out).item() == label:
        correctness = 1
    else:
        correctness = 0

    nb_pixel = args.width * args.height
    assert nb_pixel != 0 and args.l_inf_bound > 0
    location_dist = {
        'tl_x':args.topleft_x,
        'tl_y':args.topleft_y,
        'width':args.width,
        'height':args.height,
    }
    bound = obstacle_bound(img, args.topleft_x, args.topleft_y, args.width, args.height, args.l_inf_bound)
    coefs = []
    for i in range(args.width):
        for j in range(args.height):
            coefs.append((args.topleft_y + j, args.topleft_x + i))
    coefs = sorted(coefs, key=lambda x: x[0])
    # print(coefs)
    recorder = Recorder(nb_pixel, coefs, bound, args.bound_error, args.init_k, args.max_iteration, args.max_evaluation)
    nest_depth = nb_pixel
    x_loc = coefs[nest_depth-1]
    # x = img[0, x_loc[0], x_loc[1]].item()
    x = 0.

    p = torch.zeros_like(img).squeeze()
    start_time = time.time()
    nested_lip_opt(model, recorder, nest_depth, x=x, sampled_data=img, patch=p, ground_true_label=label)
    end_time = time.time()
    _, low_bound_example = min(recorder.net_val, key=lambda item: item[0])
    _, up_bound_example = max(recorder.net_val, key=lambda item: item[0])
    # print(bound)
    # print((img + low_bound_example).max(),(img + low_bound_example).min())
    opt_out = model((img + low_bound_example).unsqueeze(0).to(args.device))
    if torch.argmax(opt_out).item() == label:
        post_correctness = 1
    else:
        post_correctness = 0
    # save_image(low_bound_example, '/home/fu/workspace/DeepRA/tmp1.png')
    print(f'{args.example_idx},{correctness},{ori_conf:.6f},{post_correctness},{opt_out[0][label].item()},{recorder.cur_feval}, {(end_time - start_time):.2f}')
    print([low_bound_example.cpu().squeeze()[co].item() for co in sorted(coefs, key=lambda x: x[0])])
    if args.debug:
        # print(f'Number of evaluation: {recorder.cur_feval}, used {(end_time - start_time):.2f}s.')
        print('Output of low bounded image: ',end='')
        print(model(low_bound_example.unsqueeze(0)).data)
        # print('Output of up bounded image: ',end='')
        # print(model(up_bound_example.unsqueeze(0)).data)
        # print('Output of original image: ',end='')
        # print(ori_out.data)
        print(recorder.break_recorder)


if __name__ == "__main__":
    main()