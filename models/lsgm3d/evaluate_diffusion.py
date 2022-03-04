# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LSGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
from timeit import default_timer as timer

from .diffusion_continuous import DiffusionBase
from .diffusion_discretized import DiffusionDiscretized


def generate_samples_vada(dae, diffusion, vae, num_samples, ode_eps=None, ode_solver_tol=None,
                          ode_sample=False, prior_var=1.0, temp=1.0, vae_temp=1.0, noise=None):
    shape = [dae.num_input_channels, dae.input_size, dae.input_size, dae.input_size]
    with torch.no_grad():
        if ode_sample:
            assert isinstance(diffusion, DiffusionBase), 'ODE-based sampling requires cont. diffusion!'
            assert ode_eps is not None, 'ODE-based sampling requires integration cutoff ode_eps!'
            assert ode_solver_tol is not None, 'ODE-based sampling requires ode solver tolerance!'
            start = timer()
            eps, nfe, time_ode_solve = diffusion.sample_model_ode(dae, num_samples, shape, ode_eps, ode_solver_tol, temp, noise)
        else:
            assert isinstance(diffusion, DiffusionDiscretized), 'Regular sampling requires disc. diffusion!'
            assert noise is None, 'Noise is not used in ancestral sampling.'
            nfe = diffusion._diffusion_steps
            time_ode_solve = 999.999  # Yeah I know...
            start = timer()
            eps = diffusion.run_denoising_diffusion(dae, num_samples, shape, temp, is_image=False, prior_var=prior_var)
        decomposed_eps = vae.decompose_eps(eps)
        image = vae.sample(num_samples, vae_temp, decomposed_eps)
        end = timer()
        sampling_time = end - start
        # average over GPUs
        nfe_torch = torch.tensor(nfe * 1.0, device='cuda')
        sampling_time_torch = torch.tensor(sampling_time * 1.0, device='cuda')
        time_ode_solve_torch = torch.tensor(time_ode_solve * 1.0, device='cuda')
    return image, nfe_torch, time_ode_solve_torch, sampling_time_torch
