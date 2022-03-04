import math

import numpy as np
import pyvista
import torch
from torch.utils.tensorboard import SummaryWriter


def _add_mesh(plotter, c, v, n_cubes):
    if v.sum() == 0:
        return

    x = np.arange(0, n_cubes + 1, dtype=float)
    y = np.arange(0, n_cubes + 1, dtype=float)
    z = np.arange(0, n_cubes + 1, dtype=float)
    x, y, z = np.meshgrid(x, y, z)

    grid = pyvista.StructuredGrid(y, z, x)  # WARNING : messed up the axes so the pkmns look at the left
    ugrid = pyvista.UnstructuredGrid(grid)
    vox = ugrid.extract_cells(v.view(-1).numpy())

    vox.cell_data["colors"] = c.view(-1, 3)[v.view(-1)].numpy()
    plotter.add_mesh(mesh=vox, scalars="colors", rgb=True)
    return


def show_voxels(voxel_grids, color_grids=None, save_as_file_on=None, writer_params=None, image_size=512, packed=False):
    """
    Visualization function
    """
    voxel_grids = voxel_grids.byte().to_dense().bool() if voxel_grids.is_sparse else voxel_grids
    if color_grids is None:
        color_grids = 200 * torch.ones([*voxel_grids.shape, 3], dtype=torch.uint8, device=voxel_grids.device)
    color_grids = color_grids.to_dense() if color_grids.is_sparse else color_grids
    color_grids = color_grids.detach().cpu()
    voxel_grids = voxel_grids.detach().cpu()

    n_cubes = voxel_grids.shape[-1]
    n_objects = voxel_grids.shape[0]
    if packed:
        w = math.ceil(math.sqrt(n_objects))
        h = math.ceil(n_objects / w)
    else:
        w, h = n_objects, 1
    as_img = save_as_file_on is not None

    plotter = pyvista.Plotter(shape=(h, w), window_size=(w * image_size, h * image_size), off_screen=as_img)
    plotter.set_background('white')
    for i in range(n_objects):
        plotter.subplot(i // w, i % w)
        _add_mesh(plotter, c=color_grids[i], v=voxel_grids[i], n_cubes=n_cubes)

    if as_img:
        if type(save_as_file_on) is SummaryWriter:
            _, img = plotter.show(screenshot=True, return_cpos=True)
            img = (torch.tensor(img.copy()).float() / 255.).permute(2, 0, 1)
            if writer_params is None:
                writer_params = ('images', 0)
            save_as_file_on.add_image(writer_params[0], img, writer_params[1])
        else:
            plotter.show(screenshot=save_as_file_on)

    else:
        plotter.show()


def format_and_show_voxels(x, save_as_file_on=None, writer_params=None, image_size=512, packed=False):
    x = x.detach()
    y = torch.clamp((x[:, :3] / 2 + 0.5) * 255, min=0, max=255).round().to(torch.uint8).permute(0, 2, 3, 4, 1)
    x = x[:, 3] > 0.
    x = x[:16]  # 16 images max is enough
    y = y[:16]
    show_voxels(voxel_grids=x, color_grids=y, save_as_file_on=save_as_file_on,
                writer_params=writer_params, image_size=image_size,
                packed=packed)
