import torch
import pytorch3d
from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
)
import pytorch3d.io
from pytorch3d.vis.plotly_vis import plot_scene
from tqdm.auto import tqdm
import imageio
from pytorch3d.io import load_obj
import numpy as np
import pytorch3d.ops

def get_points_renderer(
    image_size=512, device=None, radius=0.01, background_color=(0, 0, 0)):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


def get_mesh_renderer(image_size=512, lights=None, device=None):
    """
    Returns a Pytorch3D Mesh Renderer
    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """
    # if device is None:
    #     if torch.cuda.is_available():
    device = torch.device("cuda:0")
        # else:
        #     device = torch.device("cpu")

    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer


def visualize_point_clouds(points, output_file , device=None):

    """
    Visualizes a point cloud using Pytorch3D.

    Args:
        points (torch.Tensor): A tensor of shape (N, 3) containing the point cloud
            coordinates.
        renderer (PointsRenderer): A Pytorch3D renderer for point clouds.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """

    device = torch.device("cuda:0")
    # Add random colors to the points
    rgb = torch.rand_like(points, device=device)
    pointss = pytorch3d.structures.Pointclouds(points=points, features=rgb)
    num_views = 40

    R, T = pytorch3d.renderer.look_at_view_transform(
    dist=3,
    elev=0,
    azim=np.linspace(-180, 180, num_views, endpoint=False),
    )

    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        device=device
    )

    renderer = get_points_renderer(device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
    images = renderer(pointss.extend(num_views), cameras=many_cameras, lights=lights)
    images = images.detach().cpu().numpy()
    
    imageio.mimsave(output_file, images, fps=20)

def visualize_meshes(meshes, output_file, device=None):
    """
    Visualizes a mesh using Pytorch3D.

    Args:
        meshes (pytorch3d.structures.Meshes): A Pytorch3D Meshes object containing the
            mesh to visualize.
        renderer (MeshRenderer): A Pytorch3D renderer for meshes.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """
    device = torch.device("cuda:0")

    vertices = meshes.verts_packed()
    vertices = vertices.unsqueeze(0)
    faces = meshes.faces_packed()
    faces = faces.unsqueeze(0)
    texture_rgb = torch.rand_like(vertices, device=device)
    textures = pytorch3d.renderer.TexturesVertex(texture_rgb)

    meshes = pytorch3d.structures.Meshes(
    verts=vertices,
    faces=faces,
            textures=textures,
    )


    num_views = 40
    R, T = pytorch3d.renderer.look_at_view_transform(
    dist=3,
    elev=0,
    azim=np.linspace(-180, 180, num_views, endpoint=False),
    )
    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        device=device
    )
    renderer = get_mesh_renderer(device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
    images = renderer(meshes.extend(num_views), cameras=many_cameras, lights=lights)
    images = images.detach().cpu().numpy()
    imageio.mimsave(output_file, images, fps=20)

    