# All the functions of this file are modified versions coming from the Kaolin v0.9.1 library,
# to take into account color voxels. The differentiability has probably been lost during the modifications.
# You can find here some link and the license of Kaolin.

# https://github.com/NVIDIAGameWorks/kaolin/blob/v0.9.1/kaolin/ops/conversions/trianglemesh.py
# https://github.com/NVIDIAGameWorks/kaolin/blob/v0.9.1/kaolin/ops/mesh/trianglemesh.py
# https://github.com/NVIDIAGameWorks/kaolin/blob/v0.9.1/kaolin/ops/conversions/pointcloud.py
# https://github.com/NVIDIAGameWorks/kaolin/blob/v0.9.1/kaolin/ops/voxelgrid.py
# https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/ops/conversions/voxelgrid.py

# Copyright (c) 2019,20-21 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
from scipy import ndimage
from .colored_voxels import corresponding_unique_mean, uv2vertex_color


def _unbatched_color_subdivide_vertices(vertices, faces, resolution, verts_attributes=None):
    """
    Modified version of Kaolin v0.9.1's _unbatched_subdivide_vertices
    It includes attributes for vertices (such as vertex color), which are linearly interpolated
    :param vertices: (torch.tensor) unbatched vertices of shape (V, 3) of mesh.
    :param faces: (torch.LongTensor) unbatched faces of shape (F, 3) of mesh.
    :param resolution: (int) target resolution to upsample to.
    :param verts_attributes: (torch.tensor) unbatched vertices attributes of shape (V, X) of mesh.
    :return: (torch.Tensor): upsampled vertices of shape (W, 3)
        if verts_attributes is given, return a tuple of :
            (torch.Tensor): upsampled vertices of shape (W, 3)
            (torch.Tensor): upsampled vertices attributes of shape (W, X)
    """
    assert resolution > 1
    min_edge_length = ((resolution - 1) / (resolution ** 2)) ** 2

    if verts_attributes is not None:
        verts_attributes_dtype = verts_attributes.dtype
        vertices = torch.cat([vertices, verts_attributes.float()], dim=1)

    v1 = torch.index_select(vertices, 0, faces[:, 0])  # shape of (F, 3)
    v2 = torch.index_select(vertices, 0, faces[:, 1])
    v3 = torch.index_select(vertices, 0, faces[:, 2])

    while True:
        edge1_length = torch.sum((v1[:, :3] - v2[:, :3]) ** 2, dim=1).unsqueeze(1)  # shape (F, 1)
        edge2_length = torch.sum((v2[:, :3] - v3[:, :3]) ** 2, dim=1).unsqueeze(1)
        edge3_length = torch.sum((v3[:, :3] - v1[:, :3]) ** 2, dim=1).unsqueeze(1)

        total_edges_length = torch.cat((edge1_length, edge2_length, edge3_length), dim=1)
        max_edges_length = torch.max(total_edges_length, dim=1)[0]

        # Choose the edges that is greater than the min_edge_length
        keep = max_edges_length > min_edge_length

        # if all the edges are smaller than the min_edge_length, stop upsampling
        K = torch.sum(keep)

        if K == 0:
            break

        v1 = v1[keep]  # shape of (K, 3 + 3 optional), where K is number of edges that has been kept
        v2 = v2[keep]
        v3 = v3[keep]

        # New vertices is placed at the middle of the edge
        v4 = (v1 + v3) / 2  # shape of (K, 3), where K is number of edges that has been kept
        v5 = (v1 + v2) / 2
        v6 = (v2 + v3) / 2

        # update vertices
        vertices = torch.cat((vertices, v4, v5, v6), dim=0)

        # Get rid of repeated vertices
        vertices, unique_indices = torch.unique(vertices, return_inverse=True, dim=0)

        # Update v1, v2, v3
        v1 = torch.cat((v1, v2, v4, v3))
        v2 = torch.cat((v4, v5, v5, v4))
        v3 = torch.cat((v5, v6, v6, v6))

    if verts_attributes is None:
        return vertices
    else:
        return vertices[:, :3], vertices[:, 3:].to(verts_attributes_dtype)


def _extended_points_to_voxelgrids(position, resolution, values=None, return_sparse=False):
    """
    Modified version of Kaolin v0.9.1's _base_points_to_voxelgrids
    Converts points to voxelgrids. Only points within range [0, 1] are used for voxelization. Points outside
    of [0, 1] will be discarded. You can specify the value of each voxel (could be multidimensional, such as RGB colors)
    :param position: (torch.Tensor) Exact batched points with shape (batch_size, P, 3)
    :param resolution: (int) Resolution of output voxelgrids
    :param values: (torch.Tensor) Values of each points with shape (batch_size, P, X)
    :param return_sparse: (bool) Whether to return a sparse voxelgrids or not.
    :return: (torch.Tensor): Exact batched voxelgrids with shape (batch_size, resolution, resolution, resolution, X)
        If return_sparse == True, sparse COO tensor is returned.
    """

    batch_size = position.shape[0]
    num_p = position.shape[1]

    device = position.device
    dtype = position.dtype

    if values is not None:
        assert position.shape[:-1] == values.shape[:-1], "position and values need the same dimensions except last.\n" + \
                                                     f"Found {position.shape[:-1]} and {values.shape[:-1]}."

    mult = torch.ones(batch_size, device=device, dtype=dtype) * (resolution - 1)  # size of (batch_size)

    prefix_index = torch.arange(start=0, end=batch_size, device=device, dtype=torch.long
                                ).repeat(num_p, 1).T.reshape(-1, 1)

    pc_index = torch.round((position * mult.view(-1, 1, 1))).long()
    pc_index = torch.cat((prefix_index, pc_index.reshape(-1, 3)), dim=1)
    pc_index, pc_inverse, pc_counts = torch.unique(pc_index, dim=0, return_inverse=True, return_counts=True)
    if values is not None:
        pc_values = corresponding_unique_mean(pc_index, pc_inverse, pc_counts, values.reshape(-1, *values.shape[2:]))

    # filter point that is outside of range 0 and resolution - 1
    condition = pc_index[:, 1:] <= (resolution - 1)
    condition = torch.logical_and(condition, pc_index[:, 1:] >= 0)
    row_cond = condition.all(1)

    pc_index = pc_index[row_cond, :]
    pc_index = pc_index.reshape(-1, 4)  # In case size is 0

    vg = torch.sparse_coo_tensor(pc_index.T, torch.ones(pc_index.shape[0], device=pc_index.device, dtype=torch.bool),
                                 size=(batch_size, resolution, resolution, resolution), dtype=torch.bool)

    if not return_sparse:
        vg = vg.to_dense()

    if values is not None:
        pc_values = pc_values[row_cond, :]
        pc_values = pc_values.reshape(-1, *values.shape[2:])

        vg_val = torch.sparse_coo_tensor(pc_index.T, pc_values.to(values.dtype),
                                         size=(batch_size, resolution, resolution, resolution, *values.shape[2:]),
                                         dtype=values.dtype)
        if not return_sparse:
            vg_val = vg_val.to_dense()
        return vg, vg_val

    return vg


def trianglemeshes_to_colored_voxelgrids(vertices, faces, resolution, verts_uvs=None, textures=None,
                                         origin=None, scale=None, return_sparse=False):
    """
    Modified version of Kaolin v0.9.1's trianglemeshes_to_voxelgrids

    Converts meshes to surface voxelgrids of a given resolution. It first upsamples
    triangle mesh's vertices to given resolution, then it performs a box test.
    If a voxel contains a triangle vertex, set that voxel to 1. Vertex will be
    offset and scaled as following:
    normalized_vertices = (vertices - origin) / scale
    the voxelgrids will only be generated in the range [0, 1] of normalized_vertices.
    Each voxels can have a value, such as a color, and it will be returned as colorgrids
    There are 2 ways to provide color information :
        - Using a texture image and a UV coordinates for each vertex, we can use UV mapping
        - Using one color for each vertex
    :param vertices: (list of torch.tensor) list of vertices of shape (V, 3) of mesh to convert.
    :param faces: (list of torch.tensor) list of faces of shape (F, 3) of mesh to convert.
    :param resolution: (int) desired resolution of generated voxelgrid.
    :param verts_uvs: (list of torch.tensor) list of vertices UV coordinates of shape (V, 2) of mesh to convert.
                                            Only useful when using UV mapping colors
    :param textures: It structures change depending on the use:
        - if UV mapping, (list of torch.tensor) with shapes (Ix, Iy, 3), the texture image
        - if vertex wise, (list of torch.tensor) with shapes (V, 3), colors for each vertex
    :param origin: (torch.tensor) batched origin of the voxelgrid in the mesh coordinates, with shape (batch_size, 3)
                               Default: origin = torch.min(vertices, dim=1)[0]
    :param scale: (torch.tensor) batched scale by which we divide the vertex position, with shape (batch_size)
                              Default: scale = torch.max(torch.max(vertices, dim=1)[0] - origin, dim=1)[0]
    :param return_sparse: (bool) If True, sparse COO tensor is returned.
    :return: (torch.Tensor): Binary batched voxelgrids of shape (B, resolution, resolution, resolution).
            If return_sparse == True, sparse COO tensor is returned.
            If colors are provided, return a tuple of :
                (torch.Tensor, float): Binary batched voxelgrids of shape (B, resolution, resolution, resolution)
                (torch.Tensor, uint8): Color batched voxelgrids of shape (B, resolution, resolution, resolution, 3)
    """
    assert isinstance(vertices, list) and isinstance(faces, list), "vertices and faces need to be batched in lists"
    vertices_device = vertices[0].device

    if not isinstance(resolution, int):
        raise TypeError(f"Expected resolution to be int "
                        f"but got {type(resolution)}.")

    if origin is None:
        min_val = torch.stack([i.min(0)[0] for i in vertices], dim=0)
        origin = min_val

    if scale is None:
        max_val = torch.stack([i.max(0)[0] for i in vertices], dim=0)
        scale = torch.max(max_val - origin, dim=1)[0]

    batch_size = len(vertices)

    origin = origin.to(vertices_device)
    scale = scale.to(vertices_device)
    batched_points = [(v - origin[i].unsqueeze(0)) / scale[i] for i, v in enumerate(vertices)]

    voxelgrids = list()

    if textures is not None:

        colorgrids = list()

        for i in range(batch_size):
            if verts_uvs is not None:
                # Case UV mapping
                points, uv_coords = _unbatched_color_subdivide_vertices(batched_points[i], faces[i], resolution,
                                                                        verts_attributes=verts_uvs[i])
                v_colors = uv2vertex_color(verts_uvs=uv_coords, uv_texture=textures[i])
            else:
                # Case vertex-wise colors
                points, v_colors = _unbatched_color_subdivide_vertices(batched_points[i], faces[i], resolution,
                                                                       verts_attributes=textures[i])
            voxelgrid, colorgrid = _extended_points_to_voxelgrids(points.unsqueeze(0), resolution,
                                                                  values=v_colors.unsqueeze(0), return_sparse=True)

            voxelgrids.append(voxelgrid)
            colorgrids.append(colorgrid)

        voxelgrids = torch.cat(voxelgrids, dim=0).to(device=vertices_device)
        colorgrids = torch.cat(colorgrids, dim=0).to(device=vertices_device)

        if not return_sparse:
            # Little bug in 1.8.1, fixed in later version of pytorch
            voxelgrids = voxelgrids.to(torch.uint8).to_dense().to(torch.bool)
            colorgrids = colorgrids.to_dense()

        return voxelgrids, colorgrids

    else:
        for i in range(batch_size):
            points = _unbatched_color_subdivide_vertices(batched_points[i], faces[i], resolution)
            voxelgrid = _extended_points_to_voxelgrids(points.unsqueeze(0), resolution, return_sparse=True)

            voxelgrids.append(voxelgrid)

        voxelgrids = torch.cat(voxelgrids, dim=0).to(device=vertices_device)
        if not return_sparse:
            # Little bug in 1.8.1, fixed later version of pytorch
            voxelgrids = voxelgrids.to(torch.uint8).to_dense().to(torch.bool)
        return voxelgrids


def extract_surface(voxelgrids):
    """
    Simpler version of Kaolin v0.9.1's extract_surface
    Removes any internal structure(s) from a voxelgrids.

    param: voxelgrids (torch.Tensor): Binary voxelgrids of shape (N, X, Y ,Z) from which to extract surface
    return: (torch.Tensor): Binary voxelgrids of shape (N, X, Y ,Z)
    """
    voxelgrids = voxelgrids.float()

    try:
        # output = F.avg_pool3d(voxelgrids.unsqueeze(1),
        #                       kernel_size=(3, 3, 3), padding=1, stride=1).squeeze(1)
        # output = (output < 1) * voxelgrids.bool()

        output_z = F.avg_pool3d(voxelgrids.unsqueeze(1), kernel_size=(1, 1, 3), padding=(0, 0, 1), stride=1).squeeze(1)
        output_y = F.avg_pool3d(voxelgrids.unsqueeze(1), kernel_size=(1, 3, 1), padding=(0, 1, 0), stride=1).squeeze(1)
        output_x = F.avg_pool3d(voxelgrids.unsqueeze(1), kernel_size=(3, 1, 1), padding=(1, 0, 0), stride=1).squeeze(1)

        output = ((output_x < 1) | (output_y < 1) | (output_z < 1)) * voxelgrids.bool()
    except RuntimeError as err:
        if voxelgrids.ndim != 4:
            voxelgrids_dim = voxelgrids.ndim
            raise ValueError(f"Expected voxelgrids to have 4 dimensions "
                             f"but got {voxelgrids_dim} dimensions.")

        raise err  # Unknown error

    return output


def fill(voxelgrids):
    """
    from Kaolin v0.9.1's fill
    :param voxelgrids: binary voxelgrids of size (N, X, Y, Z) to be filled
    :return: filled binary voxelgrids of size (N, X, Y, Z)
    """
    if voxelgrids.ndim != 4:
        voxelgrids_dim = voxelgrids.ndim
        raise ValueError(f"Expected voxelgrids to have 4 dimensions "
                         f"but got {voxelgrids_dim} dimensions.")

    device = voxelgrids.device

    if voxelgrids.is_cuda:
        raise NotImplementedError("Fill function is not supported on GPU yet.")

    voxelgrids = voxelgrids.data.cpu()

    output = torch.empty_like(voxelgrids)
    for i in range(voxelgrids.shape[0]):
        on = ndimage.binary_fill_holes(voxelgrids[i])
        output[i] = torch.from_numpy(on).to(torch.bool).to(device)

    return output


def voxelgrids_to_colored_cubic_meshes(voxelgrids, colorgrids=None, is_trimesh=True, origin=(0, 0, 0), scale=1):
    """
    Modified version of Kaolin v0.9.1's voxelgrids_to_cubic_meshes

    Convert voxelgrids to meshes by replacing each occupied voxel with a cuboid mesh (unit cube).
    Each cube has 8 vertices and 12 faces. Internal faces are ignored.
    If `is_trimesh==True`, this function performs the same operation as "Cubify" defined in the
    ICCV 2019 paper "Mesh R-CNN": https://arxiv.org/abs/1906.02739.
    It can take a colorgrid matrix which represent face colors
    :param voxelgrids: (torch.Tensor) binary voxel array containing valid position with shape (B, X, Y, Z).
    :param colorgrids: (torch.Tensor) uint8 voxel array containing colors with shape (B, X, Y, Z, C).
    :param is_trimesh: (bool) the outputs are triangular meshes if True. Otherwise quadmeshes are returned.
    :param origin: (tuple) tuple containing 3 floats for X,Y,Z initial origin of the mesh
    :param scale: (float) initial scale of the mesh
    :return (list[torch.Tensor], list[torch.LongTensor], list[torch.LongTensor]):
            tuple containing the list of vertices, the list of faces and the list of faces colors for each mesh.
    """
    verts_template = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0]
        ],
        dtype=torch.float
    )

    faces_template = torch.tensor(
        [
            [0, 2, 1, 3],
            [0, 1, 4, 5],
            [0, 4, 2, 6]
        ],
        dtype=torch.int64
    )
    faces_3x4x3 = verts_template[faces_template]
    for i in range(3):
        faces_3x4x3[i, :, (i - 1) % 3] -= 1
        faces_3x4x3[i, :, (i + 1) % 3] -= 1

    quad_face = torch.LongTensor([[0, 1, 3, 2]])
    kernel = torch.zeros((1, 1, 2, 2, 2))
    kernel[..., 0, 0, 0] = -1
    kernel[..., 1, 0, 0] = 1
    kernels = torch.cat([kernel, kernel.transpose(
        2, 3), kernel.transpose(2, 4)], 0)  # (3,1,2,2,2)

    device = voxelgrids.device
    n_cubes = voxelgrids.shape[1]
    voxelgrids = voxelgrids.unsqueeze(1)
    batch_size = voxelgrids.shape[0]

    voxelgrids_sum = voxelgrids.sum()
    if voxelgrids_sum == 0 or voxelgrids_sum.isnan():  # In case the whole voxelgrid is empty, or NaNs
        verts_batch = list([torch.zeros([0, 3], dtype=torch.float, device=device) for i in range(batch_size)])
        faces_batch = list([torch.zeros([0, 3], dtype=torch.long, device=device) for i in range(batch_size)])
        texatlas_batch = list([torch.zeros([0, 1, 1, 3], dtype=torch.float, device=device) for i in range(batch_size)])
        return verts_batch, faces_batch, texatlas_batch

    face = quad_face.to(device)

    if device == 'cpu':
        k = kernels.to(device).half()
        voxelgrids = voxelgrids.half()
    else:
        k = kernels.to(device).float()
        voxelgrids = voxelgrids.float()

    conv_results = torch.nn.functional.conv3d(
        voxelgrids, k, padding=1)  # (B, 3, r, r, r)

    indices = torch.nonzero(conv_results.transpose(
        0, 1), as_tuple=True)  # (N, 5)
    dim, batch, loc = indices[0], indices[1], torch.stack(
        indices[2:], -1)  # (N,) , (N, ), (N, 3)

    invert = conv_results.transpose(0, 1)[indices] == -1
    _, counts = torch.unique(dim, sorted=True, return_counts=True)

    faces_loc = (torch.repeat_interleave(faces_3x4x3.to(device), counts, dim=0) +
                 loc.unsqueeze(1).float())  # (N, 4, 3)

    faces_batch = []
    verts_batch = []
    texatlas_batch = []

    for b in range(batch_size):
        verts = faces_loc[torch.nonzero(batch == b)].view(-1, 3)
        if verts.shape[0] == 0:
            faces_batch.append(torch.zeros((0, 3 if is_trimesh else 4), device=device, dtype=torch.long))
            verts_batch.append(torch.zeros((0, 3), device=device))
            if colorgrids is not None:
                texatlas_batch.append(torch.zeros((0, 1, 1, 3), device=device))
            continue
        invert_batch = torch.repeat_interleave(
            invert[batch == b], face.shape[0], dim=0)
        N = verts.shape[0] // 4

        if colorgrids is not None:
            texatlas = verts.view(-1, 4, 3).mean(1)
            texatlas[torch.arange(texatlas.shape[0]), dim[batch == b]] += 0.5 - invert_batch.float()
            texatlas = (texatlas - 0.5).round().long()
            texatlas = colorgrids[b, texatlas[:, 0], texatlas[:, 1], texatlas[:, 2]]
            if is_trimesh:
                texatlas = texatlas.repeat(2, 1)
            texatlas = texatlas[:, None, None]
            texatlas_batch.append(texatlas)

        shift = torch.arange(N, device=device).unsqueeze(1) * 4  # (N,1)
        faces = (face.unsqueeze(0) + shift.unsqueeze(1)
                 ).view(-1, face.shape[-1])  # (N, 4) or (2N, 3)
        faces[invert_batch] = torch.flip(faces[invert_batch], [-1])

        if is_trimesh:
            faces = torch.cat(
                [faces[:, [0, 3, 1]], faces[:, [2, 1, 3]]], dim=0)

        verts, v = torch.unique(
            verts, return_inverse=True, dim=0)
        faces = v[faces.reshape(-1)].reshape((-1, 3 if is_trimesh else 4))

        verts = verts / n_cubes * scale + torch.tensor([origin], device=verts.device, dtype=verts.dtype)

        faces_batch.append(faces)
        verts_batch.append(verts)

    return verts_batch, faces_batch, texatlas_batch


def sdf_from_voxels(query, voxelgrids, colorgrid=None, origin=(0, 0, 0), scale=1.):
    """
    Gets the SDF (signed distance field) of several points from a voxelgrid, and its
    nearest color if colorgrid is provided
    :param query: (Nx3) torch.float
    :param voxelgrids: (RxRxR) torch.bool
    :param colorgrid: (RxRxRx3) torch.uint8
    :param origin: tuple of 3 floats for X, Y, Z origin
    :param scale: float of scale
    :return: dist: (RxRxR) torch.float
        [optional, if colorgrid is provided] colors: (RxRxRx3) torch.uint8
    """
    n_cubes = voxelgrids.shape[1]
    origin = torch.tensor([origin], dtype=torch.float, device=voxelgrids.device)

    surface = extract_surface(voxelgrids.unsqueeze(0)).squeeze(0)
    xyz = surface.nonzero()
    xyz_norm = (xyz.float() + 0.5) / n_cubes * scale + origin

    dist = ((query.unsqueeze(1) - xyz_norm.unsqueeze(0)) ** 2)
    dist, idx = torch.min(dist.sum(2), dim=1)

    query = ((query - origin) / scale).maximum(
        torch.zeros([1], device=query.device)).minimum(torch.ones([1], device=query.device) - 1e-12)
    query = torch.round(query * n_cubes - 0.5).long()
    dist[voxelgrids[query[:, 0], query[:, 1], query[:, 2]]] *= -1

    if colorgrid is not None:
        color_id = xyz[idx]
        colors = colorgrid[color_id[:, 0], color_id[:, 1], color_id[:, 2]]
        return dist, colors
    return dist
