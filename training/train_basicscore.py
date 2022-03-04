import functools

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils_for_visualization import format_and_show_voxels


# From Yang Song's blog on "Generative Modeling by Estimating Gradients of the Data Distribution"
# https://yang-song.github.io/blog/2021/score/
# And its tutorial :
# https://colab.research.google.com/drive/1SeXMpILhkJPjXUaesvzEhc3Ke6Zl_zxJ?usp=sharing


def marginal_prob_std(t, sigma, device):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The standard deviation.
    """
    try:
        t = torch.from_numpy(t).to(device=device)
    except TypeError:
        t = t.to(device)
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))


def diffusion_coeff(t, sigma, device):
    """Compute the diffusion coefficient of our SDE.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The vector of diffusion coefficients.
    """
    try:
        t = torch.from_numpy(t).to(device=device)
    except TypeError:
        t = t.to(device)
    return sigma ** t


def loss_fn(model, x, marginal_prob_std, eps=1e-5):
    """The loss function for training score-based generative models.

    Args:
      model: A PyTorch model instance that represents a
        time-dependent score-based model.
      x: A mini-batch of training data.
      marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
      eps: A tolerance value for numerical stability.
    """
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None, None, None, None]
    score = model(perturbed_x, random_t)
    loss = torch.mean(torch.sum((score * std[:, None, None, None, None] + z) ** 2, dim=(1, 2, 3, 4)))
    return loss


def pc_sampler(score_model,
               marginal_prob_std,
               diffusion_coeff,
               batch_size=64,
               vox_size=32,
                n_color=4,
               num_steps=500,
               snr=0.16,
               device='cuda',
               eps=1e-3):
    """Generate samples from score-based models with Predictor-Corrector method.

    Args:
      score_model: A PyTorch model that represents the time-dependent score-based model.
      marginal_prob_std: A function that gives the standard deviation
        of the perturbation kernel.
      diffusion_coeff: A function that gives the diffusion coefficient
        of the SDE.
      batch_size: The number of samplers to generate by calling this function once.
      vox_size: voxel dimension
      num_steps: The number of sampling steps.
        Equivalent to the number of discretized time steps.
      device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
      eps: The smallest time step for numerical stability.

    Returns:
      Samples.
    """
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, n_color, vox_size, vox_size, vox_size, device=device) * \
             marginal_prob_std(t)[:, None, None, None, None]
    time_steps = np.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
        for time_step in time_steps:
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            # Corrector step (Langevin MCMC)
            grad = score_model(x, batch_time_step)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm) ** 2
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)

            # Predictor step (Euler-Maruyama)
            g = diffusion_coeff(batch_time_step)
            x_mean = x + (g ** 2)[:, None, None, None, None] * score_model(x, batch_time_step) * step_size
            x = x_mean + torch.sqrt(g ** 2 * step_size)[:, None, None, None, None] * torch.randn_like(x)

            # The last step does not include any noise
        return x_mean


def train_basicscore(
        model,
        dataset,
        writer,
        marginal_prob_std_fn,
        diffusion_coeff_fn,
        n_epochs=50,
        batch_size=32,
        lr=1e-4,
        device='cuda',
        noise_coeff=1.,
        samples_per_log=512,
        scheduler_gamma=0.85
):
    score_model = model.train().to(device)
    assert dataset.n_cubes == 32

    optimizer = Adam(score_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)

    timestamp = 0
    time_to_log_imgs = 0

    # Loop through data
    for epoch in range(n_epochs):
        noise_level = 1. - (epoch / (n_epochs - 1)) ** noise_coeff
        dataset.set_noise_level(noise_level)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False)

        writer.add_scalar('BasicScore/LR', optimizer.param_groups[0]["lr"], timestamp)
        writer.add_scalar('BasicScore/Noise_level', noise_level, timestamp)

        for step, real in tqdm(enumerate(dataloader), desc=f"Epoch {epoch}", total=len(dataloader) - 1):

            real = real.to(device)
            current_batch_size = real.shape[0]
            loss = loss_fn(score_model, real, marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('BasicScore/Loss', loss.item(), timestamp)

            if timestamp >= time_to_log_imgs:

                format_and_show_voxels(real, save_as_file_on=writer,
                                       writer_params=('BasicScore/Real', timestamp), image_size=256,
                                       packed=True)

                fake = pc_sampler(score_model,
                                  marginal_prob_std=marginal_prob_std_fn,
                                  diffusion_coeff=diffusion_coeff_fn,
                                  batch_size=min(batch_size, 16),
                                  vox_size=32,
                                  n_color=4,
                                  num_steps=500,
                                  snr=0.16,
                                  device=device,
                                  eps=1e-3)

                format_and_show_voxels(fake, save_as_file_on=writer,
                                       writer_params=('BasicScore/Fake', timestamp), image_size=256,
                                       packed=True)

                while timestamp >= time_to_log_imgs:
                    time_to_log_imgs += samples_per_log

            timestamp += current_batch_size

        scheduler.step()

    return score_model
