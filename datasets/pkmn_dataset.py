import os
import os.path

import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset

from conversions.color_interpolation import interpolate_colors
from conversions.kaolin_mesh_to_voxels import trianglemeshes_to_colored_voxelgrids, fill, extract_surface
from .pkmns_funcs import list_folders, fuse_geometries, load_model
from .voxel_augments import custom_augment


class PkmnDataset(Dataset):
    # These are means and std of pkmn meshes calculated on all the dataset
    MEAN = [-0.08825463, 59.789803, 6.796502]
    STD = 45.0774

    def __init__(self, root_path):
        """
        Basic dataset that loads pkmn meshes
        :param root_path: Path to uncrypted pkmn meshes
        """
        super(PkmnDataset, self).__init__()
        self.root_path = root_path
        self.model_folders = list_folders(self.root_path)

    def __len__(self):
        return len(self.model_folders)

    def __getitem__(self, item):
        model = self.load_data(item)
        return model

    def load_data(self, item):
        chosen_model = self.model_folders[item]
        model = load_model(chosen_model)

        verts, faces, verts_uvs, packed_tex = fuse_geometries(model)

        # Classic normalize, by global means and stds
        # verts = (verts - torch.tensor([self.MEAN], dtype=verts.dtype)) / torch.tensor(self.STD, dtype=verts.dtype)
        # Min-max normalization in [-1, 1], centered in 0, get rid of pkmn size info, but easier to deal with
        verts = (verts - verts.min(dim=0)[0]) / (verts - verts.min(dim=0)[0]).max()
        verts = 2 * (verts - verts.max(dim=0)[0] / 2)

        faces = faces.long()
        packed_tex = torch.from_numpy(packed_tex)[..., :3].float() / 255.

        return {"verts": verts, "faces": faces, "verts_uv": verts_uvs, "texturesUV": packed_tex}


class RawVoxelsPkmnDataset(PkmnDataset):
    def __init__(self, root_path, n_cubes):
        """
        Loads pkmn meshes, then transform them into voxels, and apply some augmentations
        :param root_path: Path to uncrypted pkmn meshes
        :param n_cubes: Dimension of the voxel grid (total number of voxels = n_cubes*n_cubes*n_cubes)
        """
        super(RawVoxelsPkmnDataset, self).__init__(root_path)
        self.n_cubes = n_cubes

    def __getitem__(self, item):
        x = super().__getitem__(item)
        voxels, colored_voxels = trianglemeshes_to_colored_voxelgrids(
            [x['verts']], [x['faces']], self.n_cubes, verts_uvs=[x['verts_uv']], textures=[x['texturesUV']],
            origin=torch.tensor([[-1., -1., -1.]]),
            scale=torch.tensor([2.]),
            return_sparse=False
        )
        voxels = fill(voxels)
        colored_voxels = interpolate_colors(colorgrid=colored_voxels,
                                            voxelgrid=extract_surface(voxels),
                                            darken_outside=True)

        return voxels[0], colored_voxels[0], self.model_folders[item]


def save_entire_pkmn_voxels_dataset_on_disk(root_path, voxels_folder, n_cubes=64):
    """
    Pre-computes colored voxels of the Pkmn dataset, and store them on disk. They can then be loaded
    by VoxelsDataset class
    :param root_path: Path to uncrypted pkmn meshes
    :param voxels_folder: Folder path on which to save voxels
    :param n_cubes: Dimension of the voxel grid (total number of voxels = n_cubes*n_cubes*n_cubes)
    """
    from tqdm import tqdm
    d = RawVoxelsPkmnDataset(root_path=root_path, n_cubes=n_cubes)
    for data in tqdm(d):
        new_name = os.path.join(voxels_folder, data[2].split('/')[-1] + '_nc' + str(int(n_cubes)) + '.pt')
        indices = extract_surface(data[0].unsqueeze(0)).squeeze(0).nonzero()
        colors = data[1]

        torch.save(f=new_name, obj=(indices, colors))


def load_voxels(path):
    """
    Load from the disk a filled boolean voxelgrid and an interpolated uint8 colorgrid
    """
    indices, colors = torch.load(f=path)
    n_cubes = colors.shape[1]
    voxels = torch.zeros([n_cubes, n_cubes, n_cubes], dtype=torch.bool)
    voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
    voxels = fill(voxels.unsqueeze(0)).squeeze(0)
    return voxels, colors


class VoxelsDataset(Dataset):
    def __init__(self, root_path, item_n_cubes: int, files_n_cubes: int = 64, noise_level: float = 1.):
        """
        Loads preprocessed data from the disk
        :param root_path: Path to uncrypted pkmn meshes
        :param item_n_cubes: (int) Desired dimension of the returned voxel grid
        :param files_n_cubes: (int) Dimension of the voxel grid of the input saved files
        :param noise_level: (float) Augmentation noie level, 0. = no noise, 1. and more = lot of noise
        """
        super(VoxelsDataset, self).__init__()
        self.root_path = root_path
        self.model_folders = [i for i in os.listdir(root_path) if i[-5:] == str(int(files_n_cubes)) + '.pt']
        if len(self.model_folders) == 0:
            raise Warning('Your data folder is empty. You need to call "save_entire_pkmn_voxels_dataset_on_disk" to'
                          'fill it.')
        self.n_cubes = item_n_cubes
        self.noise_level = noise_level

    def __len__(self):
        return len(self.model_folders)

    def __getitem__(self, item):
        model = self.load_data(item)
        return model

    def load_data(self, item):
        path = os.path.join(self.root_path, self.model_folders[item])
        voxelgrid, colorgrid = load_voxels(path)

        res = torch.cat([colorgrid.permute(3, 0, 1, 2).float() / 255.,
                         voxelgrid.float().unsqueeze(0)],
                        dim=0)
        res = self.downsample(res)
        res = custom_augment(res, level=self.noise_level)
        res = (res*2-1).clamp(-1, 1)  # Normalize between -1 and 1
        return res

    def downsample(self, x):
        scale = x.shape[1] // self.n_cubes
        assert x.shape[1] % self.n_cubes == 0
        x = torch.cat([functional.avg_pool3d(x[:3], scale, stride=scale, padding=0),
                       functional.max_pool3d(x[3].unsqueeze(0), scale, stride=scale, padding=0)], dim=0)
        return x

    def set_size(self, n_cubes):
        self.n_cubes = n_cubes

    def set_noise_level(self, noise_level):
        self.noise_level = noise_level
