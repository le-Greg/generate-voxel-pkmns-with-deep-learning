import functools

import torch
from torch.utils.tensorboard import SummaryWriter

from datasets.pkmn_dataset import VoxelsDataset
from models.basic_score3d import ScoreNet
from training.train_basicscore import train_basicscore, marginal_prob_std, diffusion_coeff

if __name__ == '__main__':
    dataset = VoxelsDataset(root_path='data/shapenet_voxels', item_n_cubes=-1, files_n_cubes=32)
    dataset.set_size(32)
    writer = SummaryWriter('logs/tensorboard/basicscore32_shapenet')

    sigma, device = 35., torch.device('cuda:0')
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma, device=device)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma, device=device)

    model = ScoreNet(marginal_prob_std=marginal_prob_std_fn)

    # Pretrain on ShapeNet
    model = train_basicscore(
        model=model,
        dataset=dataset,
        writer=writer,
        marginal_prob_std_fn=marginal_prob_std_fn,
        diffusion_coeff_fn=diffusion_coeff_fn,
        n_epochs=15,
        batch_size=32,
        lr=1e-3,
        device=device,
        noise_coeff=1.,
        samples_per_log=10000,
        scheduler_gamma=0.88
    )

    torch.save(obj=model.state_dict(), f='logs/models/basicscore32_shapenet.pt')
    model.load_state_dict(torch.load('logs/models/basicscore32_shapenet.pt'))

    dataset = VoxelsDataset(root_path='data/pkmn_voxels', item_n_cubes=-1, files_n_cubes=32)
    dataset.set_size(32)
    writer = SummaryWriter('logs/tensorboard/basicscore32')

    model = train_basicscore(
        model=model,
        dataset=dataset,
        writer=writer,
        marginal_prob_std_fn=marginal_prob_std_fn,
        diffusion_coeff_fn=diffusion_coeff_fn,
        n_epochs=500,
        batch_size=32,
        lr=1e-3,
        device=device,
        noise_coeff=1.,
        samples_per_log=10000,
        scheduler_gamma=0.99
    )

    torch.save(obj=model.state_dict(), f='logs/models/basicscore32.pt')
