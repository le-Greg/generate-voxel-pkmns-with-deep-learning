import os

import numpy as np
import rpack
import torch
from PIL import Image
from collada import Collada
from collada.material import Map


# Functions to make the conversion from COLLADA dumped files to Pytorch 3D exploitable ones

def load_model(path):
    """
    Load model data from decrypted .gfpak folders
    :param path: Path to the folder containing the Collada file of the pkmn mesh, and all its textures
    :return: A dict containing all different vertices, faces, and UV coords array, and all useful textures paths
    """
    files = [os.path.join(path, f) for f in os.listdir(path)]

    dae_path = [j for j in files if j.endswith('.dae')]
    assert len(dae_path) == 1, 'There should be only 1 .dae model per folder, found ' + str(len(dae_path))
    dae = Collada(dae_path[0])

    # dae.controller => Its used for animations, skinning properties, vertex weights etc
    # dae.effect => It contains aesthetics
    # dae.materials => Link to effect

    texture_name = dict()
    for m in dae.materials:
        # Only diffuse map (= real colors) are used
        if type(m.effect.diffuse) == Map:
            texture_name[m.id] = os.path.join(path, m.effect.diffuse.sampler.surface.image.path)

    model = dict()

    for g in dae.geometries:
        assert not g.double_sided
        meshes = list()
        for triangles in g.primitives:
            # In COLLADA representation, there are vertices that are all arranged in triangles
            # But vertices repeats themselves multiple times for each time there are in a face
            # In Pytorch 3D representation, we need to have a list of vertices, and a list of
            # vertices indices which constitutes the faces
            vertex_index = triangles.vertex_index
            valid_idx = np.unique(vertex_index.flatten())
            vertices = triangles.vertex[valid_idx]

            faces = np.zeros(vertex_index.shape, dtype=np.int32)
            for idx, f in enumerate(valid_idx):
                faces[vertex_index == f] = idx
            # Check that all the faces indexes are in the range of 0 to N-1 (N = vertices number)
            assert np.array_equal(
                np.linspace(0, vertices.shape[0] - 1, vertices.shape[0], dtype=faces.dtype),
                np.unique(faces)
            )

            # To map the meshes to their texture, in Pytorch 3D, we need one (and only) texture,
            # and the U and V values for each vertex
            texcoord_indexset = triangles.texcoord_indexset
            texcoordset = triangles.texcoordset

            if len(texcoordset) == 0 or len(texcoord_indexset) == 0:
                # When there are no textures, it is generally flames or smokes. Since there is
                # no good way to represent it in pytorch3D, we get rid of it
                continue

            if triangles.material not in texture_name.keys():
                # If the texture is there, but is not diffuse, we skip too
                continue

            # Sometimes there are multiple textures for one mesh. We can use only one with Pytorch3D,
            # so we just assume the first one is the main one.
            faces_uv = texcoordset[0][texcoord_indexset[0]]

            verts_uvs = np.zeros([vertices.shape[0], 2], dtype=np.float32)
            for idx in range(vertices.shape[0]):
                uvs = faces_uv[faces == idx]
                # Most vertices share many faces, so they appear multiple times in tex_uv
                # We must check that their UVs are the same
                assert np.unique(uvs, axis=0).shape == (1, 2)
                verts_uvs[idx] = uvs[0]

            meshes.append({
                "vertices": vertices,
                "faces": faces,
                "verts_uv": verts_uvs,
                "texture": texture_name[triangles.material]
            })
        model[g.name] = meshes

    return model


def list_folders(root_path):
    """
    Get all available pkmns folders
    """
    all_models = list()
    folders = [os.path.join(root_path, i) for i in os.listdir(root_path)]
    for i in folders:
        all_models += [os.path.join(i, f) for f in os.listdir(i)]
    return all_models


def pack_textures(textures: list):
    """
    COLLADA files handle multiple image textures, Pytorch3D only handle one. This function fuse them up
    into one
    :param textures: list of numpy array (texture images)
    :return: im, numpy array of fused textures
            res, list of shifted origin position for each texture in the fused one
            bbox, shape of im
    """
    h_w_rects = [list(r.shape[:2]) for r in textures]
    res = rpack.pack(h_w_rects)
    bbox = rpack.bbox_size(res, h_w_rects)

    im = np.zeros([*bbox, *textures[0].shape[2:]], dtype=textures[0].dtype)
    for r in range(len(textures)):
        h, w = textures[r].shape[:2]
        im[res[r][0]:res[r][0] + h, res[r][1]:res[r][1] + w] = textures[r]
    return im, res, bbox


def repeat_mirror_texture(tex, max_x, max_y):
    """
    In pkmns models, uv coords can be out of the 0-1 range, so we must apply padding. Uv coords are pre-normalized
    to be positive and near 0,0. This function does it manually by expanding the textures. They are repeated in
    the up-down axis, and are mirrored in the left-right axis
    :param tex: numpy array of texture which needs to be expanded
    :param max_x: float, max value on the right side
    :param max_y: float, max value on the up side
    :return: tex : np.array, new expanded texture
        tiling_right : int, number of repetitions in the right direction
        tiling_up : int, number of repetitions in the up direction
    """
    if max_x > 1.0:
        tex = np.concatenate([tex, tex[:, ::-1]], axis=1)

    tiling_up = int(np.ceil(max_y))
    tiling_right = int(np.ceil(max_x / 2) * 2) if max_x > 1.0 else 1
    tex = np.tile(tex, (
        tiling_up,
        int(tiling_right / 2) if max_x > 1.0 else 1,
        1
    ))

    return tex, (tiling_right, tiling_up)


def fuse_geometries(model):
    """
    Create one big mesh, with one big set of vertices, one big set of faces, one big set of uv coordinates,
    and one big texture, so that it is usable by pytorch 3D
    :param model: dict, output dict from load_model function
    :return: vertices, faces, verts_uvs, full_tex (as torch tensors)
    """
    vertice = list()
    face = list()
    verts_uv = list()

    textures = dict()
    n_face = 0
    for geo_name in model.keys():
        for i in range(len(model[geo_name])):
            mesh = model[geo_name][i]
            vertice.append(torch.from_numpy(mesh["vertices"]))
            face.append(torch.from_numpy(mesh["faces"]) + n_face)

            n_face += mesh["vertices"].shape[0]

            im_path = model[geo_name][i]['texture']
            if im_path not in textures:
                textures[im_path] = np.array(Image.open(im_path))

            v_uvs = mesh["verts_uv"]
            verts_uv.append([im_path, v_uvs])

    vertices = torch.cat(vertice, dim=0)
    faces = torch.cat(face, dim=0)

    # Texture is the tricky part, because we need to fuse all texture images into one, and then re-shift
    # and re-normalize the UV coordinates on the new picture, without changing the order in the array,
    # which needs to be the same as in the "vertices" array. We also needs to take into consideration that
    # textures are repeated in the up-down axis, and mirrored in the right-left axis, and we want to get them back
    # in the 0-1 range
    textures_keys = list(textures.keys())
    for im_key in textures_keys:
        max_pos, min_pos = np.array([-1e9, -1e9]), np.array([1e9, 1e9])
        for i in range(len(verts_uv)):
            if verts_uv[i][0] == im_key:
                verts_uv[i][1][:, 0] -= 2 * np.floor(verts_uv[i][1][:, 0].min() / 2)
                verts_uv[i][1][:, 1] -= np.floor(verts_uv[i][1][:, 1].min())

                max_pos = np.maximum(max_pos, np.max(verts_uv[i][1], axis=0))
                min_pos = np.minimum(min_pos, np.min(verts_uv[i][1], axis=0))

        assert not np.any(max_pos < -1e8), max_pos
        assert not np.any(min_pos > 1e8), min_pos

        tex, tiling = repeat_mirror_texture(textures[im_key], max_x=max_pos[0], max_y=max_pos[1])
        textures[im_key] = tex
        for i in range(len(verts_uv)):
            if verts_uv[i][0] == im_key:
                verts_uv[i][1] /= np.expand_dims(np.array(tiling), 0)

    full_tex, res, bbox = pack_textures([textures[im_key] for im_key in textures_keys])

    for im_key in range(len(textures_keys)):
        for i in range(len(verts_uv)):
            if verts_uv[i][0] == textures_keys[im_key]:
                verts_uv[i][1] *= np.expand_dims(np.array(textures[textures_keys[im_key]].shape[1::-1]), 0)
                verts_uv[i][1] += np.expand_dims(np.array(res[im_key][::-1]), 0)
                verts_uv[i][1] /= np.expand_dims(np.array(bbox[::-1]), 0)

    verts_uvs = torch.cat([torch.from_numpy(i[1]) for i in verts_uv], dim=0)

    # It needs to be flipped vertically
    verts_uvs[:, 1] = 1 - verts_uvs[:, 1]
    return vertices, faces, verts_uvs, full_tex
