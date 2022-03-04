## Generating New 3D Voxel Pokemon Using Deep Learning

I generated new pokemons in 3D using GAN, VAE and score-based models. Check out [this post](https://le-greg.github.io/generate-voxel-pkmns-with-deep-learning/) to learn more

### Running Experiments

##### Datasets

**Getting the dataset containing the 3D models of pokemons is quite a pain. You have to dig into Pokemon Sword/Shield cartridges to get them, and Nintendo hasn't made the process easy for us. If you just want to try training without generating pokemon specifically, I'd recommend just using the ShapeNet dataset instead.**

To get the ShapeNet dataset, you must first create an account on [their site](https://shapenet.org/signup/), which must be approved. Then download the dataset. The file tree should look like this:
```
ShapeNetCore.v2
 |-- 02691156
 |-- 02747177
 ...
 |-- 04554684
 |-- taxonomy.json
```

Then change the appropriate paths in preprocess_data.py and run `python preprocess_data.py` to transform the shapenet meshes into voxels. It will store the new files at the path specified in preprocess_data.py. The original ShapeNet is about 100 GB, and the preprocessed data is an additional 7 GB.

##### Dependencies

```
pycollada                 0.7.2
python                    3.9.7
pytorch3d                 0.6.1
pyvista                   0.33.2
scipy                     1.7.0
tensorboard               2.7.0
```

##### Trainings

Once you data is ready, adjust the parameters in the python scripts, and run them :
```
python train_on_basic_gan_16.py
python train_on_basic_score_32.py
python train_on_basic_vae_32.py
```

Check the result on tensorboard with the command `tensorboard --logdir logs/tensorboard/`
