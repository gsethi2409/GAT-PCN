import argparse
import time
import torch
from model import PCN

import numpy as np

from pytorch3d.loss import chamfer_distance

from pytorch3d.datasets import ShapeNetCore, collate_batched_meshes, r2n2_shapenet_dataset
from torch.utils.data import DataLoader

from pdb import set_trace as bp

def get_args_parser():
    parser = argparse.ArgumentParser('GAT_PCN', add_help=False)
    # Model parameters
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--max_iter', default=10000, type=int)
    parser.add_argument('--log_freq', default=1000, type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=0, type=str)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=1.0, type=float)
    parser.add_argument('--save_freq', default=500, type=int)    
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--load_feat', action='store_true') 
    parser.add_argument('--load_checkpoint', action='store_true')            
    return parser

def preprocess(feed_dict, args):
    images = feed_dict['images'].squeeze(1)
    mesh = feed_dict['mesh']
    pointclouds_tgt = sample_points_from_meshes(mesh, args.n_points)    
    ground_truth_3d = pointclouds_tgt
    return images.to(args.device), ground_truth_3d.to(args.device)

def calculate_loss(predictions, ground_truth, args):
    loss_chamfer, _ = chamfer_distance(ground_truth, predictions)
    return loss

def train_model(args):
    SHAPENET_PATH = "/home/pointcloud/Gunjan/3dlearning/data/a2/r2n2_shapenet_dataset/shapenet"
    shapenet_dataset = ShapeNetCore(SHAPENET_PATH)
    loader = DataLoader(shapenet_dataset, batch_size=args.batch_size, collate_fn=collate_batched_meshes)
    train_loader = iter(loader)

    bp()
    
    model = PCN(args)
    model.to(args.device)
    model.train()

    # ============ preparing optimizer ... ============
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)  # to use with ViTs
    start_iter = 0
    start_time = time.time()

    losses = []

    if args.load_checkpoint:
        checkpoint = torch.load(f'checkpoint_{args.type}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint['step']
        print(f"Succesfully loaded iter {start_iter}")
    
    print("Starting training !")

    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        if step % len(train_loader) == 0: #restart after one epoch
            train_loader = iter(loader)

        read_start_time = time.time()

        feed_dict = next(train_loader)    

        images_gt, ground_truth_3d = preprocess(feed_dict,args)
        read_time = time.time() - read_start_time

        prediction_3d = model(images_gt, args)
        loss = calculate_loss(prediction_3d, ground_truth_3d, args)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        if (step % args.save_freq) == 0:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, f'{step}_checkpoint_{args.type}.pth')

            losses.append(loss_vis)

        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f" % (step, args.max_iter, total_time, read_time, iter_time, loss_vis))
    
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }, f'final_checkpoint_{args.type}.pth')

    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('GAT_PCN', parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
