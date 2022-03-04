# NVAE https://github.com/NVlabs/NVAE


import torch
from torch.utils.data import DataLoader
import numpy as np
from math import ceil
from tqdm import tqdm

from .utils_for_visualization import show_voxels
from models.nvae3d.spectral_norm import SpectralNormCalculator
from models.nvae3d.training_functions import kl_balancer_coeff, AvgrageMeter, vae_terms, kl_coeff, reconstruction_loss,\
    kl_balancer


def train_nvae(
        dataset,  # Dataset
        writer,  # Tensorboard writer
        nvae,  # Model
        samples_per_log=512,  # Num training samples before logging
        device='cuda:0',  # Device
        learning_rate_vae=1e-2,  # LR VAE optimizers
        learning_rate_min_vae=1e-4,  # Minimal learning rate for the scheduler
        weight_decay=3e-4,  # weight decay VAE scheduler (CosineAnnealing)
        weight_decay_norm=3e-2,  # The lambda parameter for spectral regularization.
        weight_decay_norm_init=10.,  # The initial lambda parameter
        weight_decay_norm_anneal=False,  # This flag enables annealing the lambda coefficient from
        kl_const_portion=0.0001,  # The portions epochs that KL is constant at kl_const_coeff
        kl_const_coeff=0.0001,  # The constant value used for min KL coeff
        n_epochs=10,  # Number of epochs
        n_warmup_epochs=2,  # Number of warm up epochs for the scheduler
        n_warmup_iters=10,  # Number of warm up batches before scheduling
        batch_size=4,  # Batch size
        kl_anneal_portion=1.,
        kl_max_coeff=1.,
        noise_coeff=2,  # Evolution of noise following 1-x**noise_coeff, x between 0 (start) and 1 (end)
):
    nvae = nvae.to(device)

    # Logging
    stat_holder = AvgrageMeter()
    timestamp = 0
    num_total_iter = (len(dataset) // batch_size) * n_epochs

    # Set optimizers and schedulers
    vae_optimizer = torch.optim.Adamax(nvae.parameters(), lr=learning_rate_vae, weight_decay=weight_decay, eps=1e-3)
    vae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        vae_optimizer, n_epochs - n_warmup_epochs - 1, eta_min=learning_rate_min_vae
    )

    # create SN calculator
    sn_calculator = SpectralNormCalculator()
    sn_calculator.add_conv_layers(nvae)
    sn_calculator.add_bn_layers(nvae)

    nvae.train()
    # Loop through data
    for epoch in range(n_epochs):

        if epoch > n_warmup_epochs:
            vae_scheduler.step()

        alpha_i = kl_balancer_coeff(num_scales=nvae.num_latent_scales,
                                    groups_per_scale=nvae.groups_per_scale, fun='square').to(device)

        noise_level = 1. - (epoch / (n_epochs - 1)) ** noise_coeff
        dataset.set_noise_level(noise_level)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

        writer.add_scalar('NVAE/Noise_level', noise_level, timestamp * batch_size)
        writer.add_scalar('NVAE/LR', vae_optimizer.param_groups[0]["lr"], timestamp * batch_size)

        for step, x in tqdm(enumerate(dataloader), desc=f"Epoch {epoch}", total=len(dataloader)-1):
            x = x.to(device)

            # warm-up lr
            if timestamp < n_warmup_iters:
                lr = learning_rate_vae * timestamp / n_warmup_iters
                for param_group in vae_optimizer.param_groups:
                    param_group['lr'] = lr

            vae_optimizer.zero_grad()

            logits, all_log_q, all_eps = nvae(x)
            log_q, log_p, kl_all, kl_diag = vae_terms(all_log_q, all_eps)
            output = nvae.decoder_output(logits)
            kl_c = kl_coeff(timestamp, kl_anneal_portion * num_total_iter,
                            kl_const_portion * num_total_iter, kl_const_coeff,
                            kl_max_coeff)

            recon_loss = reconstruction_loss(output, x, crop=nvae.crop_output)
            balanced_kl, kl_coeffs, kl_vals = kl_balancer(kl_all, kl_c, kl_balance=True, alpha_i=alpha_i)

            nelbo_batch = recon_loss + balanced_kl
            loss = torch.mean(nelbo_batch)
            norm_loss = sn_calculator.spectral_norm_parallel()
            bn_loss = sn_calculator.batchnorm_loss()
            # get spectral regularization coefficient (lambda)
            if weight_decay_norm_anneal:
                assert weight_decay_norm_init > 0 and weight_decay_norm > 0, 'init and final wdn should be positive.'
                wdn_coeff = (1. - kl_c) * np.log(weight_decay_norm_init) + kl_c * np.log(weight_decay_norm)
                wdn_coeff = np.exp(wdn_coeff)
            else:
                wdn_coeff = weight_decay_norm

            loss += norm_loss * wdn_coeff + bn_loss * wdn_coeff

            loss.backward()
            vae_optimizer.step()
            stat_holder.update(loss.data, 1)
            writer.add_scalar('NVAE/loss_batch', loss.data, timestamp * batch_size)
            writer.add_scalar('NVAE/norm_loss', norm_loss, timestamp * batch_size)
            writer.add_scalar('NVAE/bn_loss', bn_loss, timestamp * batch_size)
            if (timestamp+1) % samples_per_log == 0:
                x = x.detach()
                y = torch.clamp((x[:, :3] / 2 + 0.5) * 255, min=0, max=255).round().to(torch.uint8).permute(0, 2, 3,
                                                                                                            4, 1)
                x = x[:, 3] > 0.
                x = x[:16]  # 16 images max is enough
                y = y[:16]
                show_voxels(voxel_grids=x, color_grids=y, save_as_file_on=writer,
                            writer_params=(f'NVAE/Real', timestamp * batch_size), image_size=256,
                            packed=True)

                # Log images
                for t in [1.0, 0.7]:
                    img = nvae.sample(num_samples=batch_size, t=t, eps_z=None)

                    x = img.detach()
                    y = torch.clamp(x[:, :3] * 255, min=0, max=255).round().to(torch.uint8).permute(0, 2, 3, 4, 1)
                    x = x[:, 3] > 0.5
                    x = x[:16]  # 16 images max is enough
                    y = y[:16]
                    show_voxels(voxel_grids=x, color_grids=y, save_as_file_on=writer,
                                writer_params=(f'NVAE/Fake_{t}', timestamp * batch_size), image_size=256,
                                packed=True)

            timestamp += 1

        # Log data
        writer.add_scalar('NVAE/loss_epoch', stat_holder.avg, timestamp * batch_size)
        stat_holder.reset()

    return nvae
