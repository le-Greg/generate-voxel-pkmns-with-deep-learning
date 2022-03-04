import torch
import numpy as np


def fibonacci_sphere(samples: int = 1000):
    """
    Samples points uniformly on an unit sphere
    See https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    :param samples: Number of points to sample
    :return: torch.tensor, shape [N, 3], XYZ coordinates of the N points on the sphere
    """

    increment = np.linspace(0, samples - 1, samples)
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    y = 1 - (increment / float(samples - 1)) * 2  # y goes from 1 to -1
    radius = np.sqrt(1 - y * y)  # radius at y

    theta = phi * increment  # golden angle increment

    x = np.cos(theta) * radius
    z = np.sin(theta) * radius

    return np.stack([x, y, z], axis=1)


def interpolate_colors(colorgrid, voxelgrid, darken_outside=False):
    """
    The colors of the mesh can only be known from the voxels on the surface, so this interpolate
    fake colors in colorgrid from existing points
    :param colorgrid: B X Y Z 3 uint8 torch.tensor, colors of the grid
    :param voxelgrid: B X Y Z bool torch.tensor, surface boolean voxel grid
    :param darken_outside: Add black points on the edges to make a little decay effect
    :return: B X Y Z 3 uint8 torch.tensor
    """
    from scipy.interpolate import RBFInterpolator, NearestNDInterpolator
    batch_size = voxelgrid.shape[0]
    cube_size = voxelgrid.shape[1:]
    n_pts_sphere = 100

    if not voxelgrid.is_sparse:
        points = [torch.nonzero(voxelgrid[i]).T for i in range(batch_size)]
    else:
        points = [voxelgrid[i].indices() for i in range(batch_size)]  # tuple of len B, [Ni, 3] array of each x, y, z
    if colorgrid.is_sparse:
        colorgrid.to_dense()
    colorgrid = colorgrid.float() / 255.
    val = [colorgrid[i, points[i][0], points[i][1], points[i][2]] for i in range(batch_size)]  # tuple of [Ni, 3] colors

    xs = torch.linspace(0, cube_size[0] - 1, steps=cube_size[0]).view(-1, 1, 1)
    ys = torch.linspace(0, cube_size[1] - 1, steps=cube_size[1]).view(1, -1, 1)
    zs = torch.linspace(0, cube_size[2] - 1, steps=cube_size[2]).view(1, 1, -1)
    xi = torch.stack([xs.expand(*cube_size).flatten(),
                      ys.expand(*cube_size).flatten(),
                      zs.expand(*cube_size).flatten()], dim=1)

    external_sphere = fibonacci_sphere(n_pts_sphere)
    # Radius of the sphere needs to be >> sqrt(2)*cube_size. Here radius == 3*cube_size
    external_sphere = (external_sphere * 1.5 + 0.5) * (np.array(cube_size)[None] - 1)

    for b in range(batch_size):
        b_points = np.array(points[b].cpu().T)
        b_values = np.array(val[b].cpu())
        if len(b_points) == 0:  # No points
            continue

        if darken_outside:
            b_points = np.concatenate((b_points, external_sphere), axis=0)
            b_values = np.concatenate((b_values, np.zeros([n_pts_sphere, 3])), axis=0)
        # rbfi = RBFInterpolator(b_points, b_values)  # radial basis function interpolator instance
        rbfi = NearestNDInterpolator(b_points, b_values)
        di = rbfi(xi)
        colorgrid[b] = torch.from_numpy(di.reshape(*cube_size, 3)).to(device=colorgrid.device)

    return torch.clamp(colorgrid * 255., min=0, max=255).round().to(torch.uint8)  # Less precise, more memory efficient
