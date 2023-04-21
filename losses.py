import torch
import torch.nn as nn
import pytorch3d
import pytorch3d.ops
import pytorch3d.loss


# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	m = nn.Sigmoid()
	loss = nn.BCELoss()
	input = voxel_src
	target = voxel_tgt
	output = loss(m(input), target)	
	return output

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	dists1 , idxs1, _ = pytorch3d.ops.knn_points(point_cloud_src, point_cloud_tgt, norm = 2,  K=1)
	dists2 , idxs2, _ = pytorch3d.ops.knn_points(point_cloud_tgt, point_cloud_src, norm = 2, K=1)
	c_loss = torch.sum(dists1) + torch.sum(dists2)
	return c_loss

def smoothness_loss(mesh_src):
	loss_laplacian = pytorch3d.loss.mesh_laplacian_smoothing(mesh_src, method='uniform')
	return loss_laplacian