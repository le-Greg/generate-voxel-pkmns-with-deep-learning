# Pokemon dataset
from datasets.pkmn_dataset import save_entire_pkmn_voxels_dataset_on_disk

pkmn_3d_data = '/path/to/your/pkmn/mesh/folder'
pkmn_voxel_data = 'data/pkmn_voxels/'

n_cubes = 32
save_entire_pkmn_voxels_dataset_on_disk(root_path=pkmn_3d_data,
                                        voxels_folder=pkmn_voxel_data,
                                        n_cubes=n_cubes)

# ShapeNet dataset
from datasets.shapenet_voxels import save_entire_shapenet_voxels_dataset_on_disk

shapenet_3d_data = '/path/to/your/shapenet/mesh/folder'
shapenet_voxel_data = 'data/shapenet_voxels/'

n_cubes = 32
save_entire_shapenet_voxels_dataset_on_disk(root_path=shapenet_3d_data,
                                            voxels_folder=shapenet_voxel_data,
                                            n_cubes=n_cubes,
                                            synsets=None)

print('Done !')
