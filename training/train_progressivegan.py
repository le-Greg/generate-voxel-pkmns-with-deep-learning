import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from .utils_for_visualization import show_voxels
from tqdm import tqdm


def train_progressivegan(
        dataset,  # Voxel dataset, or a dataset with set_size and set_noise_level
        generator,  # Generator net
        discriminator,  # Discriminator net
        loss_criterion,  # Loss criterion
        writer,  # Tensorboard writer
        glr=1e-4,  # generator learning rate
        dlr=1e-4,  # discriminator learning rate
        batch_sizes=(32, 32, 16, 8, 4, 4),  # Batch sizes for each voxel resolution
        resolutions=(2, 4, 8, 16, 32, 64),  # Voxel grid resolution
        n_samples=(2000, 4000, 16000, 64000, 64000, 64000),  # Number of samples shown to the network before scaling
        noise_coeff=1,  # Augmentation level of data
        device="cuda:0",  # Device
        samples_per_log=512,  # Log an image in tensorboard every samples_per_log samples
        fade_factor=3,  # Fading during (total number of samples)/fade_factor
        epsilonD=0,  # Epsilon loss coefficient
        # Schedulers
        learning_rate_max_d=1e-2,  # Max LR for discriminator's OneCycle scheduler
        learning_rate_max_g=2.5e-3,  # Max LR for generator's OneCycle scheduler
        pct_start=0.3,  # The percentage of the cycle (in number of steps) spent increasing the learning rate.
):

    generator = generator.to(device).train()
    discriminator = discriminator.to(device).train()

    assert len(resolutions) == len(batch_sizes) == len(n_samples)

    # for every resolution
    timestamp = 0
    time_to_log_imgs = 0
    optimizer_d = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()),
                             betas=(0, 0.99), lr=dlr)
    optimizer_g = optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()),
                             betas=(0, 0.99), lr=glr)

    for i, res in enumerate(resolutions):
        batch_size = batch_sizes[i]
        n_iterations = n_samples[i] // batch_size
        print('res = ', res, ' / batch size = ', batch_size, ' / n samples = ', n_iterations*batch_size)

        dataset.set_size(res)
        dataset.set_noise_level(1)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
        data_iterator = iter(dataloader)

        scheduler_d = torch.optim.lr_scheduler.OneCycleLR(
            optimizer_d, max_lr=learning_rate_max_d, total_steps=n_iterations, pct_start=pct_start
        )
        scheduler_g = torch.optim.lr_scheduler.OneCycleLR(
            optimizer_g, max_lr=learning_rate_max_g, total_steps=n_iterations, pct_start=pct_start
        )

        if i != 0:
            generator.addScale(res)
            discriminator.addScale(res)
            generator = generator.to(device)
            discriminator = discriminator.to(device)

        for n in tqdm(range(1, n_iterations + 1)):

            try:
                data = data_iterator.next()
            except StopIteration:
                noise_level = 1. - (n / n_iterations) ** noise_coeff
                dataset.set_noise_level(noise_level)
                data_iterator = iter(
                    DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
                )
                data = data_iterator.next()

            if i != 0:
                alpha = max(1-(n / n_iterations) * fade_factor, 0)
            else:
                alpha = 0
            generator.setNewAlpha(alpha)
            discriminator.setNewAlpha(alpha)

            real_input = data.to(device)
            current_batch = real_input.size()[0]

            # Update the discriminator
            optimizer_d.zero_grad()

            # #1 Real data
            pred_real_d = discriminator(real_input, False)
            loss_d = loss_criterion.getCriterion(pred_real_d, True)

            # #2 Fake data
            input_latent = torch.randn(current_batch, generator.dimLatent).to(device)
            pred_fake_g = generator(input_latent).detach()
            pred_fake_d = discriminator(pred_fake_g, False)

            loss_d_fake = loss_criterion.getCriterion(pred_fake_d, False)
            loss_d += loss_d_fake

            # #3 Epsilon loss
            loss_epsilon = (pred_real_d[:, 0] ** 2).sum() * epsilonD
            loss_d += loss_epsilon

            loss_d.backward(retain_graph=True)
            optimizer_d.step()

            # Update the generator
            optimizer_g.zero_grad()
            optimizer_d.zero_grad()

            # #1 Image generation
            input_noise = torch.randn(current_batch, generator.dimLatent).to(device)
            pred_fake_g = generator(input_noise)

            # #2 Status evaluation
            pred_fake_d, phi_g_fake = discriminator(pred_fake_g, True)

            # #3 GAN criterion
            loss_g_fake = loss_criterion.getCriterion(pred_fake_d, True)
            loss_g_fake.backward(retain_graph=True)

            optimizer_g.step()

            writer.add_scalar('ProgressiveGAN/alpha', alpha, timestamp)
            writer.add_scalar('ProgressiveGAN/Loss_G_fake', loss_g_fake, timestamp)
            writer.add_scalar('ProgressiveGAN/Loss_D_fake', loss_d_fake, timestamp)
            writer.add_scalar('ProgressiveGAN/Loss_D', loss_d, timestamp)
            writer.add_scalar('ProgressiveGAN/LR_D', optimizer_d.param_groups[0]["lr"], timestamp)
            writer.add_scalar('ProgressiveGAN/LR_G', optimizer_g.param_groups[0]["lr"], timestamp)

            if timestamp >= time_to_log_imgs:
                x = pred_fake_g.detach()
                y = torch.clamp((x[:, :3] / 2 + 0.5) * 255, min=0, max=255).round().to(torch.uint8).permute(0, 2, 3,
                                                                                                            4, 1)
                x = x[:, 3] > 0.
                x = x[:16]  # 16 images max is enough
                y = y[:16]
                show_voxels(voxel_grids=x, color_grids=y, save_as_file_on=writer,
                            writer_params=(f'ProgressiveGAN/Fake_{res}', timestamp), image_size=256,
                            packed=True)

                x = real_input.detach()
                y = torch.clamp((x[:, :3] / 2 + 0.5) * 255, min=0, max=255).round().to(torch.uint8).permute(0, 2, 3,
                                                                                                            4, 1)
                x = x[:, 3] > 0.
                x = x[:16]  # 16 images max is enough
                y = y[:16]
                show_voxels(voxel_grids=x, color_grids=y, save_as_file_on=writer,
                            writer_params=(f'ProgressiveGAN/Real_{res}', timestamp), image_size=256,
                            packed=True)
                while timestamp >= time_to_log_imgs:
                    time_to_log_imgs += samples_per_log

            timestamp += batch_size

            scheduler_d.step()
            scheduler_g.step()

    return {'discriminator': discriminator.cpu(), 'generator': generator.cpu()}
