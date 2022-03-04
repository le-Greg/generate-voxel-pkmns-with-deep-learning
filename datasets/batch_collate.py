# Taken from Pytorch 3D v0.5.0 pytorch3d.datasets.utils.py, and slightly modified it to take into account UV textures

# https://github.com/facebookresearch/pytorch3d/blob/v0.5.0/pytorch3d/datasets/utils.py

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

from pytorch3d.renderer import TexturesUV, TexturesAtlas
from pytorch3d.structures import Meshes


def collate_batched_meshes(batch: List[Dict]):  # pragma: no cover
    """
    Take a list of objects in the form of dictionaries and merge them
    into a single dictionary. This function can be used with a Dataset
    object to create a torch.utils.data.Dataloader which directly
    returns Meshes objects.

    Args:
        batch: List of dictionaries containing information about objects
            in the dataset.

    Returns:
        collated_dict: Dictionary of collated lists. If batch contains both
            verts and faces, a collated mesh batch is also returned.
    """
    if batch is None or len(batch) == 0:
        return None
    collated_dict = {}
    for k in batch[0].keys():
        collated_dict[k] = [d[k] for d in batch]

    collated_dict["mesh"] = None
    if {"verts", "faces"}.issubset(collated_dict.keys()):

        textures = None
        if "textures" in collated_dict:
            textures = TexturesAtlas(atlas=collated_dict["textures"])
        # Added these lines to take care of TextureUV for our needs
        elif "texturesUV" in collated_dict and "verts_uv" in collated_dict.keys():
            textures = TexturesUV(maps=collated_dict["texturesUV"],
                                  faces_uvs=collated_dict["faces"],
                                  verts_uvs=collated_dict["verts_uv"],
                                  padding_mode="border", align_corners=True)

        collated_dict["mesh"] = Meshes(
            verts=collated_dict["verts"],
            faces=collated_dict["faces"],
            textures=textures,
        )

    return collated_dict
