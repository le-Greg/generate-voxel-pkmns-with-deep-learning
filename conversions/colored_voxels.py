import torch


def corresponding_unique_mean(uniques, inverse, counts, values):
    """
    Average the properties of values array following the torch.unique filtering of another indices array
    It uses sparse tensors for speed/memory tradeoff
    :param uniques, inverse, counts: output of torch.unique function on a vector X on one specific dimension
    :param values: torch tensor, first dimension is the same size as the specific dimension
    :return: torch tensor
    """
    device = values.device
    dtype = values.dtype
    values = values.float()

    coo_idx = torch.stack([torch.linspace(0, inverse.shape[0] - 1, inverse.shape[0], device=device), inverse], dim=0)
    x = torch.sparse_coo_tensor(coo_idx, values, (inverse.shape[0], uniques.shape[0], *values.shape[1:]))
    x = torch.sparse.sum(x, dim=0)
    x = x.to_dense()
    x = x / counts.view(-1, *(1,) * len(values.shape[1:]))

    if dtype is torch.uint8:
        x = x.clamp(0, 255).round()
    return x.type(dtype)


def atlas2vertex_color(vertices, faces, atlas_texture, use_mean=True):
    """
    Convert "Atlas texture" representation to "Color per vertex"
    :param vertices: Vx3, torch.tensor, float
    :param faces: Fx3, torch.tensor, long
    :param atlas_texture: FxIxIx3, float (0 to 1)
    :param use_mean: if True, average the color of a vertex using all faces that contains it
        If you are sure that the color of a vertex is the same for each faces, set to False
    :return: torch tensor Vx3, float (0 to 1), RGB colors per vertex
    """
    colors = torch.zeros([vertices.shape[0], 3], dtype=atlas_texture.dtype, device=atlas_texture.device)
    face_verts_colors = torch.stack([atlas_texture[:, 0, -1], atlas_texture[:, -1, 0], atlas_texture[:, 0, 0]], dim=1)

    if use_mean:
        # Get the vertex wise texture by taking the points on the limits of the textures atlases, and
        # keeping the mean for each faces that share the vertex
        colors = colors.index_add_(0, faces.view(-1), face_verts_colors.view(-1, 3))
        count = torch.zeros([vertices.shape[0]], dtype=torch.long, device=atlas_texture.device)
        count = count.index_add_(0, faces.view(-1), torch.ones_like(faces.view(-1)))
        colors = colors / count.clamp(min=1).view(-1, 1)
    else:
        # One vertex is in multiple places in faces. Here is the easy way, where we take anyone of them, because
        # we assume all the others have the same color.
        colors[faces.flatten()] = face_verts_colors.view(-1, 3)

    return colors


def uv2vertex_color(verts_uvs, uv_texture, force_byte=True):
    """
    Convert "UV texture" representation to "Color per vertex"
    :param verts_uvs: torch tensor Vx2, float (0 to 1)
    :param uv_texture: torch tensor HxWx3, uint8 (0 to 255) or float (0 to 1)
    :param force_byte: force output to be uint8
    :return: torch tensor Vx3, uint8 (0 to 255) or float (0 to 1), RGB colors per vertex
    """
    # Switch V coords, it works like that in pkmn dataset, don't know why
    verts_uvs = verts_uvs.clone()
    verts_uvs[:, 1] = 1 - verts_uvs[:, 1]

    colors = torch.round(
        verts_uvs * (torch.tensor(uv_texture.shape[1::-1], device=verts_uvs.device)[None] - 1)
    ).long()
    colors = uv_texture[colors[:, 1], colors[:, 0]]
    if force_byte:
        colors = torch.round(colors * 255.).byte()
    return colors
