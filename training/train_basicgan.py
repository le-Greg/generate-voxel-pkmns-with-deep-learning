# Inspired by https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan/wgan.py


from math import ceil

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils_for_visualization import format_and_show_voxels


def train_basicgan(
        dataset,  # Voxel dataset, or a dataset with set_size and set_noise_level
        generator,  # Generator net
        discriminator,  # Discriminator net
        writer,  # Tensorboard writer
        glr=1e-4,  # generator learning rate
        dlr=1e-4,  # discriminator learning rate
        beta1=0.5,  # Beta for Adam optimizers
        batch_size=16,  # Batch sizes for each voxel resolution
        n_epochs=10,  # Number of samples shown to the network
        device="cuda:0",  # Device
        noise_coeff=3,  # Evolution of noise following 1-x**noise_coeff, x between 0 (start) and 1 (end)
        samples_per_log=1000,  # Frequency of saving images
        # Schedulers
        learning_rate_max_d=1e-2,  # Max LR for discriminator's OneCycle scheduler
        learning_rate_max_g=2.5e-3,  # Max LR for generator's OneCycle scheduler
        pct_start_d=0.4,  # The percentage of the cycle (in number of steps) spent increasing the learning rate.
        pct_start_g=0.4,  # The percentage of the cycle (in number of steps) spent increasing the learning rate.
        train_dis_every_x_times=1,
):
    generator = generator.to(device).train()
    discriminator = discriminator.to(device).train()
    assert dataset.n_cubes == 16

    timestamp = 0
    time_to_log_imgs = 0
    criterion = torch.nn.BCELoss()

    nz = generator.nz
    fixed_noise = torch.randn(batch_size, nz, 1, 1, 1, device=device)
    real_label = 0.9  # Label smoothing
    fake_label = 0

    # setup optimizer
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=dlr, betas=(beta1, 0.999))
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=glr, betas=(beta1, 0.999))
    len_dataloader = ceil(len(dataset) / batch_size)
    scheduler_d = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_d, max_lr=learning_rate_max_d, steps_per_epoch=len_dataloader, epochs=n_epochs, pct_start=pct_start_d
    )
    scheduler_g = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_g, max_lr=learning_rate_max_g, steps_per_epoch=len_dataloader, epochs=n_epochs, pct_start=pct_start_g
    )

    # Loop through data
    for epoch in range(n_epochs):
        noise_level = 1. - (epoch / (n_epochs - 1)) ** noise_coeff
        dataset.set_noise_level(noise_level)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False)

        writer.add_scalar('BasicGAN/LR_D', optimizer_d.param_groups[0]["lr"], timestamp)
        writer.add_scalar('BasicGAN/LR_G', optimizer_g.param_groups[0]["lr"], timestamp)
        writer.add_scalar('BasicGAN/Noise_level', noise_level, timestamp)

        for step, x in tqdm(enumerate(dataloader), desc=f"Epoch {epoch}", total=len(dataloader) - 1):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            real = x.to(device)
            current_batch_size = real.size(0)
            label = torch.full((current_batch_size,), real_label,
                               dtype=real.dtype, device=device)
            noise = torch.randn(current_batch_size, nz, 1, 1, 1, device=device)
            fake = generator(noise)
            if step % train_dis_every_x_times == 0:
                discriminator.zero_grad()

                output = discriminator(real)
                err_d_real = criterion(output, label)
                err_d_real.backward()

                # train with fake
                label.fill_(fake_label)
                output = discriminator(fake.detach())
                err_d_fake = criterion(output, label)
                err_d_fake.backward()
                err_d = err_d_real + err_d_fake
                optimizer_d.step()

                writer.add_scalar('BasicGAN/Loss_D', err_d.item(), timestamp)

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            discriminator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = discriminator(fake)
            err_g = criterion(output, label)
            err_g.backward()
            optimizer_g.step()

            writer.add_scalar('BasicGAN/Loss_G', err_g.item(), timestamp)

            if timestamp >= time_to_log_imgs:
                with torch.no_grad():
                    fixed_noise_fake = generator(fixed_noise)

                format_and_show_voxels(fixed_noise_fake, save_as_file_on=writer,
                                       writer_params=('BasicGAN/Fixed_Noise_Fake', timestamp), image_size=256,
                                       packed=True)

                format_and_show_voxels(fake, save_as_file_on=writer,
                                       writer_params=('BasicGAN/Fake', timestamp), image_size=256,
                                       packed=True)

                format_and_show_voxels(real, save_as_file_on=writer,
                                       writer_params=('BasicGAN/Real', timestamp), image_size=256,
                                       packed=True)

                while timestamp >= time_to_log_imgs:
                    time_to_log_imgs += samples_per_log

            timestamp += current_batch_size

            scheduler_d.step()
            scheduler_g.step()

    return {'discriminator': discriminator.cpu(), 'generator': generator.cpu()}
