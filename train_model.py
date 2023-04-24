import argparse
import time
import torch
from model import SingleViewto3D
from r2n2_custom import R2N2
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
import dataset_location
from pytorch3d.ops import sample_points_from_meshes
import losses
import matplotlib.pyplot as plt 

def get_args_parser():

    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    # Model parameters
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--lr', default=4e-4, type=str)
    parser.add_argument('--max_iter', default=200, type=str)
    parser.add_argument('--log_freq', default=10, type=str)
    parser.add_argument('--batch_size', default=32, type=str)
    parser.add_argument('--num_workers', default=0, type=str)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=600, type=float)
    parser.add_argument('--save_freq', default=1, type=int)    
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--add_occlusion', action='store_true') 
    parser.add_argument('--load_feat', action='store_true') 
    # parser.add_argument('--load_checkpoint', action='store_true')            
    
    return parser

def preprocess(feed_dict,args):
    images = feed_dict['images'].squeeze(1)
    if args.type == "point":
        mesh = feed_dict['mesh']
        pointclouds_tgt = sample_points_from_meshes(mesh, args.n_points)    
        ground_truth_3d = pointclouds_tgt  

    return images.to(args.device), ground_truth_3d.to(args.device)

def calculate_loss(predictions, ground_truth, args):

    if args.type == 'point':
        loss = losses.chamfer_loss(predictions, ground_truth)   
    return loss

def train_model(args):

    r2n2_dataset = R2N2("train", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True, add_occlusion=args.add_occlusion, occlusion_patch_size=40, occlusion_type="white")
    
    # shuffle the dataset so that images are not just from one category
    
    r2n2_dataset = torch.utils.data.Subset(r2n2_dataset, torch.randperm(len(r2n2_dataset))) 

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True)
    
    print(f"\nDataset size: {len(r2n2_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of batches: {len(loader)}")
    
    train_loader = iter(loader)
    model =  SingleViewto3D(args)
    model.to(args.device)
    model.train()

    # ============ preparing optimizer ... ============
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)  # to use with ViTs
    start_iter = 0
    start_time = time.time()

    # if args.load_checkpoint:
    #     checkpoint = torch.load(f'./checkpoints/checkpoint_{args.batch_size}_{step}.pth')
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     start_iter = checkpoint['step']
    #     print(f"Succesfully loaded iter {start_iter}")
    
    print("Starting training !")
    
    for i in range(0, args.max_iter+1):

        loss = 0

    # so as to go through the whole dataset once
        for step in range(start_iter, len(loader)):

            iter_start_time = time.time()

            if step % len(train_loader) == 0: #restart after one epoch
                train_loader = iter(loader)

            read_start_time = time.time()

            feed_dict = next(train_loader)

            images_gt, ground_truth_3d = preprocess(feed_dict,args)

            # for i in range(0,args.batch_size):
            #     plt.imsave(f'./training_images/{step}_{i}.png', images_gt[i].squeeze().detach().cpu().numpy())

            read_time = time.time() - read_start_time

            prediction_3d = model(images_gt, args)

            loss = calculate_loss(prediction_3d, ground_truth_3d, args)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()        

            total_time = time.time() - start_time
            iter_time = time.time() - iter_start_time

            loss_vis = loss.cpu().item()
            loss += loss_vis
            print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f" % (step, len(loader), total_time, read_time, iter_time, loss))


        if (i % args.save_freq) == 0:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, f'./checkpoints/checkpoint_{args.batch_size}_{i}_{loss}.pth')

        # print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f" % (step, args.max_iter, total_time, read_time, iter_time, loss_vis))
        print("----------[%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f" % (i, args.max_iter, total_time, read_time, iter_time, loss))


    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)