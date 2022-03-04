import torch
from torch.utils.tensorboard import SummaryWriter

from datasets.pkmn_dataset import VoxelsDataset
from models.basic_vae3d import BasicVAE3d
from training import train_basicvae

if __name__ == '__main__':
    model = BasicVAE3d(in_channels=4, latent_dim=50)
    device = torch.device('cuda:0')

    # Pretrain on ShapeNet
    dataset = VoxelsDataset(root_path='data/shapenet_voxels', item_n_cubes=-1, files_n_cubes=32)
    dataset.set_size(32)
    writer = SummaryWriter('logs/tensorboard/basicvae32_shapenet')

    model = train_basicvae(
        dataset,
        model,
        writer,
        lr=0.001,
        batch_size=16,
        n_epochs=30,
        device=device,
        noise_coeff=1,
        samples_per_log=4000,
        weight_decay=0.0,
        kld_weight=1e-4,
        scheduler_gamma=0.89,
    )

    torch.save(obj=model.state_dict(), f='logs/models/basicvae32_shapenet.pt')
    model.load_state_dict(torch.load('logs/models/basicvae32_shapenet.pt'))

    # Train on PkmnDataset
    dataset = VoxelsDataset(root_path='data/pkmn_voxels', item_n_cubes=-1, files_n_cubes=32)
    dataset.set_size(32)
    writer = SummaryWriter('logs/tensorboard/basicvae32')

    model = train_basicvae(
        dataset,
        model,
        writer,
        lr=1e-3,
        batch_size=16,
        n_epochs=300,
        device=device,
        noise_coeff=1,
        samples_per_log=4000,
        weight_decay=0.0,
        kld_weight=1e-5,
        scheduler_gamma=0.985,
    )

    torch.save(obj=model.state_dict(), f='logs/models/basicvae32.pt')
