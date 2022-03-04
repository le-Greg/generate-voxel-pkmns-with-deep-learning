import torch

from models.basic_gan3d import BasicGenerator, BasicDiscriminator
from training import train_basicgan
from datasets.pkmn_dataset import VoxelsDataset
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    nc, nz, ngf, ndf = 4, 32, 16, 16
    gen = BasicGenerator(nc=nc, nz=nz, ngf=ngf, leak=0.1)
    dis = BasicDiscriminator(nc=nc, ndf=ndf, leak=0.1)

    dataset = VoxelsDataset(root_path='data/pkmn_voxels', item_n_cubes=-1, files_n_cubes=32)
    dataset.set_size(16)
    writer = SummaryWriter('logs/tensorboard/basicgan')

    models = train_basicgan(
        dataset=dataset,  # Voxel dataset, or a dataset with set_size and set_noise_level
        generator=gen,  # Generator net
        discriminator=dis,  # Discriminator net
        writer=writer,  # Tensorboard writer
        glr=1e-5,  # generator learning rate
        dlr=4e-5,  # discriminator learning rate
        beta1=0.5,  # Beta for Adam optimizers
        batch_size=32,  # Batch size
        n_epochs=700,  # Number of epochs
        device="cuda:0",  # Device
        noise_coeff=.5,  # Evolution of noise following 1-x**noise_coeff, x between 0 (start) and 1 (end)
        samples_per_log=5000,  # Frequency of saving images
        # Schedulers
        learning_rate_max_d=8e-3,  # Max LR for discriminator's OneCycle scheduler
        learning_rate_max_g=2e-3,  # Max LR for generator's OneCycle scheduler
        pct_start_d=0.25,  # The percentage of the cycle (in number of steps) spent increasing the learning rate.
        pct_start_g=0.25,  # The percentage of the cycle (in number of steps) spent increasing the learning rate.
        train_dis_every_x_times=2,  # Training discriminator less than generator
    )

    torch.save(obj=models['generator'].state_dict(), f='logs/models/basicgan_gen16.pt')
    torch.save(obj=models['discriminator'].state_dict(), f='logs/models/basicgan_dis16.pt')
