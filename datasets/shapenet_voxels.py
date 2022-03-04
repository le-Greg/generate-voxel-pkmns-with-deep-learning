import os
import warnings
from os import path
from typing import Dict

import torch
from pytorch3d.datasets.shapenet.shapenet_core import ShapeNetCore

from conversions.color_interpolation import interpolate_colors
from conversions.colored_voxels import atlas2vertex_color
from conversions.kaolin_mesh_to_voxels import fill, extract_surface, trianglemeshes_to_colored_voxelgrids


class ShapeNetVoxels(ShapeNetCore):
    def __init__(self, data_dir, synsets=None, filled=True) -> None:
        """
        Loads pre-calculated voxels objects from ShapeNet, in binvox format
        Store each object's synset id and models id from data_dir. It uses Trimesh library to encode voxels.

        Args:
            data_dir: Path to ShapeNetCore data.
            synsets: List of synset categories to load from ShapeNetCore in the form of
                synset offsets or labels. A combination of both is also accepted.
                When no category is specified, all categories in data_dir are loaded.
            filled: internal part are filled
        """
        super().__init__(data_dir=data_dir,
                         synsets=synsets,
                         version=2,
                         load_textures=False,
                         texture_resolution=1)
        from trimesh.exchange.binvox import load_binvox
        self.load_binvox = load_binvox

        self.filled = filled
        self.model_dir = path.join("models", f"model_normalized.{'solid' if self.filled else 'surface'}.binvox")

        # Re-extract model_id, but for .binvox files
        self.synset_ids = []
        self.model_ids = []
        self.synset_num_models = {}
        for synset in self.synset_start_idxs.keys():
            for model in os.listdir(path.join(data_dir, synset)):
                if not path.exists(path.join(data_dir, synset, model, self.model_dir)):
                    msg = ("Object file not found in the model directory %s "
                           "under synset directory %s."
                           ) % (model, synset)
                    warnings.warn(msg)
                    continue
                self.synset_ids.append(synset)
                self.model_ids.append(model)
            model_count = len(self.synset_ids) - self.synset_start_idxs[synset]
            self.synset_num_models[synset] = model_count

    def __getitem__(self, idx: int) -> Dict:
        """
        Read a model by the given index.

        Args:
            idx: The idx of the model to be retrieved in the dataset.

        Returns:
            dictionary with following keys:
            - voxel (str): voxel representation using Trimesh VoxelGrid class for encoding
            - synset_id (str): synset id
            - model_id (str): model id
            - label (str): synset label.
        """
        model = self._get_item_ids(idx)
        model_path = path.join(
            self.shapenet_dir, model["synset_id"], model["model_id"], self.model_dir
        )
        with open(model_path, 'rb') as file_obj:
            model["voxel"] = self.load_binvox(file_obj, resolver=None, axis_order='xzy', file_type=None)
        model["label"] = self.synset_dict[model["synset_id"]]
        return model


class RawVoxelsShapeNetDataset(ShapeNetCore):
    """
    The voxels are calculated from the ShapeNet mesh in the same way as for the Pkmn dataset,
    to be able to use transfer learning between the 2. This also allows to include the color of the voxels.
    """

    def __init__(self, root_path, n_cubes, synsets=None):
        super(RawVoxelsShapeNetDataset, self).__init__(root_path, synsets=synsets, version=2, load_textures=True,
                                                       texture_resolution=4)
        self.n_cubes = n_cubes

    def __getitem__(self, item):
        x = super().__getitem__(item)
        verts = x["verts"]
        faces = x["faces"]
        atlas_texture = x["textures"]

        # Min-max normalization in [-1, 1], centered in 0
        verts = (verts - verts.min(dim=0)[0]) / (verts - verts.min(dim=0)[0]).max()
        verts = 2 * (verts - verts.max(dim=0)[0] / 2)

        # We use vertex color on the initial mesh, so faces size must be small enough in comparison of voxel size
        v_colors = atlas2vertex_color(vertices=verts, faces=faces, atlas_texture=atlas_texture, use_mean=True)
        v_colors = (v_colors * 255).clamp(min=0, max=255).to(torch.uint8)

        voxels, colored_voxels = trianglemeshes_to_colored_voxelgrids(
            [verts], [faces], self.n_cubes, verts_uvs=None, textures=[v_colors],
            origin=torch.tensor([[-1., -1., -1.]]),
            scale=torch.tensor([2.]),
            return_sparse=False
        )
        voxels = fill(voxels)
        colored_voxels = interpolate_colors(colorgrid=colored_voxels,
                                            voxelgrid=extract_surface(voxels),
                                            darken_outside=True)

        return {'synset_id': x['synset_id'], 'model_id': x['model_id'], 'label': x['label'],
                'voxelgrid': voxels[0], 'colorgrid': colored_voxels[0]}


def save_entire_shapenet_voxels_dataset_on_disk(root_path, voxels_folder, n_cubes=64, synsets=None):
    """
    Pre-computes colored voxels of the ShapeNet dataset, and store them on disk. They can then be loaded
    by VoxelsDataset class (see pkmn_dataset.py)
    :param root_path: ShapeNet path
    :param voxels_folder: Folder path on which to save voxels
    :param n_cubes: Dimension of the voxel grid (total number of voxels = n_cubes*n_cubes*n_cubes)
    :param synsets: ShapeNet synsets, see Pytorch3d's ShapeNetCore
    """
    from tqdm import tqdm
    d = RawVoxelsShapeNetDataset(root_path=root_path, n_cubes=n_cubes, synsets=synsets)
    for i in tqdm(range(len(d))):
        try:
            data = d[i]
        except Exception as e:  # Some samples fails, so it just skip them
            print(i, e)
            continue
        new_name = path.join(voxels_folder,
                             data['synset_id'] + '_' + data['model_id'] + '_' + data['label'] +
                             '_nc' + str(int(n_cubes)) + '.pt')
        indices = extract_surface(data['voxelgrid'].unsqueeze(0)).squeeze(0).nonzero()
        colors = data['colorgrid']

        torch.save(f=new_name, obj=(indices, colors))
