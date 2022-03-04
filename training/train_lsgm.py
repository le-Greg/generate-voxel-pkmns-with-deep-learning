# NVAE https://github.com/NVlabs/NVAE
# LSGM https://github.com/NVlabs/LSGM


import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from .utils_for_visualization import show_voxels
from tqdm import tqdm
from math import ceil

from models.lsgm3d.diffusion_continuous import make_diffusion
from models.lsgm3d.diffusion_discretized import DiffusionDiscretized
from models.lsgm3d.evaluate_diffusion import generate_samples_vada
from models.lsgm3d.ema import EMA
from models.lsgm3d.utils import cross_entropy_normal, get_mixed_prediction, kl_per_group_vada, infer_active_variables, update_lr, \
    dae_regularization, vae_regularization
from models.nvae3d.spectral_norm import SpectralNormCalculator
from models.nvae3d.training_functions import kl_balancer_coeff, reconstruction_loss, kl_coeff, kl_balancer


def train_lsgm(
        dataset,  # Dataset
        writer,  # Tensorboard writer
        vae,  # VAE model
        dae,  # DAE model
        samples_per_log=512,  # Num training samples before logging
        device='cuda:0',  # Device
        learning_rate_vae=1e-2,  # LR VAE optimizers
        learning_rate_min_vae=1e-4,  # Minimal learning rate for the scheduler
        weight_decay=3e-4,  # weight decay VAE scheduler (CosineAnnealing)
        n_epochs=10,  # Number of epochs
        n_warmup_epochs=2,  # Number of warm up epochs for the scheduler
        n_warmup_iters=10,  # Number of warm up batches before scheduling
        batch_size=4,  # Batch size
        ema_decay=0.9999,  # EMA decay factor')
        weight_decay_norm_vae=0.,  # The lambda parameter for spectral regularization.')
        grad_clip_max_norm=0.,  # The maximum norm used in gradient norm clipping (0 applies no clipping).')
        # Diffusion
        learning_rate_dae=3e-4,  # init learning rate
        learning_rate_min_dae=3e-4,  # min learning rate
        weight_decay_norm_dae=0.,  # The lambda parameter for spectral regularization.
        custom_conv_dae=False,  # Set this argument if conv layers in the SGM prior are custom layers from NVAE.
        diffusion_steps=1000,  # number of diffusion steps
        sigma2_0=0.0,  # initial SDE variance at t=0 (sort of represents Normal perturbation of input data)
        beta_start=0.1,  # initial beta variance value
        beta_end=20.0,  # final beta variance value
        sigma2_min=1e-4,  # initial beta variance value
        sigma2_max=0.99,  # final beta variance value
        #  what kind of sde type to use when training/evaluating in continuous manner.
        sde_type='geometric_sde',  # choices=['geometric_sde', 'vpsde', 'sub_vpsde']
        time_eps=1e-2,  # During training, t is sampled in [time_eps, 1.].
        # enables learning the conditional VAE decoder distribution standard deviations
        denoising_stddevs='beta',  # ['learn', 'beta', 'beta_post']
        # VADA
        # Specifies the weighting mechanism used for training p (sgm prior) and if it uses importance sampling
        # ll_uniform, ll_iw, drop_all_uniform, drop_all_iw, drop_sigma2t_iw, rescale_iw, drop_sigma2t_uniform,
        iw_sample_p='ll_uniform',
        # Specifies the weighting mechanism used for training q (vae) and whether or not to use importance sampling.
        # reweight_p_samples indicates reweight the t samples generated for the prior as done in Algorithm 3.
        iw_sample_q='reweight_p_samples',  # ['reweight_p_samples', 'll_uniform', 'll_iw'],
        iw_subvp_like_vp_sde=False,
        drop_inactive_var=False,  # Drops inactive latent variables.
        # When p (sgm prior) and q (vae) have different objectives, trains them in two separate forward calls
        update_q_ema=False,  # Enables updating q with EMA parameters of prior.
        # second stage VADA KL annealing
        cont_kl_anneal=False,  # If true, we continue KL annealing using below setup when training LSGM.
        kl_anneal_portion_vada=0.1,  # The portions epochs that KL is annealed
        kl_const_portion_vada=0.0,  # The portions epochs that KL is constant at kl_const_coeff
        kl_const_coeff_vada=0.7,  # The constant value used for min KL coeff
        kl_max_coeff_vada=1.,  # The constant value used for max KL coeff
        kl_balance_vada=False,  # If true, we use KL balancing during VADA KL annealing.
        train_ode_eps=1e-2,  # ODE can only be integrated up to some epsilon > 0.
        train_ode_solver_tol=1e-4,  # ODE solver error tolerance.
        noise_coeff=2,  # Evolution of noise following 1-x**noise_coeff, x between 0 (start) and 1 (end)
):
    vae = vae.to(device)
    dae = dae.to(device)

    # Logging
    timestamp = 0
    num_total_iter = (len(dataset) // batch_size) * n_epochs
    fast_ode_param = {'ode_eps': train_ode_eps, 'ode_solver_tol': train_ode_solver_tol}

    # Set optimizers and schedulers
    vae_optimizer = Adam(vae.parameters(), lr=learning_rate_vae, weight_decay=weight_decay, eps=1e-3)

    vae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        vae_optimizer, n_epochs - n_warmup_epochs - 1, eta_min=learning_rate_min_vae)
    dae_optimizer = Adam(dae.parameters(), learning_rate_dae, weight_decay=weight_decay, eps=1e-4)
    dae_optimizer = EMA(dae_optimizer, ema_decay=ema_decay)

    dae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        dae_optimizer, n_epochs - n_warmup_epochs - 1, eta_min=learning_rate_min_dae)

    # create SN calculator
    vae_sn_calculator = SpectralNormCalculator(custom_conv=True)  # NVAE consists of our own custom conv layer classes
    dae_sn_calculator = SpectralNormCalculator(custom_conv=custom_conv_dae)  # NCSN++ consists of pytorch conv layers
    vae_sn_calculator.add_conv_layers(vae)
    vae_sn_calculator.add_bn_layers(vae)
    dae_sn_calculator.add_conv_layers(dae)
    dae_sn_calculator.add_bn_layers(dae)

    # create diffusion
    diffusion_cont = make_diffusion(sde_type, sigma2_0, sigma2_min, sigma2_max, time_eps, beta_end, beta_start)
    diffusion_disc = DiffusionDiscretized(denoising_stddevs, diffusion_steps, var_fun=diffusion_cont.var)

    assert iw_sample_p != iw_sample_q or update_q_ema, \
        'disjoint training is for the case training objective of p and q are not the same unless q is ' \
        'updated with the EMA parameters.'
    assert iw_sample_q in ['ll_uniform', 'll_iw']

    # ----------
    #  Training
    # ----------

    dae.train()
    vae.train()
    # Loop through data
    for epoch in range(n_epochs):

        if epoch > n_warmup_epochs:
            dae_scheduler.step()
            vae_scheduler.step()

        # remove disabled latent variables by setting their mixing component to a small value
        if epoch == 0 and dae.mixed_prediction and drop_inactive_var:
            # inferring active latent variables.
            is_active = infer_active_variables(dataloader, vae, max_iter=1000)
            dae.mixing_logit.data[0, torch.logical_not(is_active), 0, 0] = -15
            dae.is_active = is_active.float().view(1, -1, 1, 1)

        alpha_i = kl_balancer_coeff(num_scales=vae.num_latent_scales,
                                    groups_per_scale=vae.groups_per_scale, fun='square')

        noise_level = 1. - (epoch / (n_epochs - 1)) ** noise_coeff
        dataset.set_noise_level(noise_level)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

        writer.add_scalar('LSGM/LR', vae_optimizer.param_groups[0]["lr"], timestamp * batch_size)
        writer.add_scalar('LSGM/Noise_level', noise_level, timestamp * batch_size)

        for step, x in tqdm(enumerate(dataloader), desc=f"Epoch {epoch}", total=len(dataloader)-1):
            x = x.to(device)
            dae.train()
            vae.train()

            # warm-up lr
            update_lr(learning_rate_dae, learning_rate_vae, True, timestamp, n_warmup_iters, dae_optimizer,
                      vae_optimizer)

            if update_q_ema and timestamp > 0:
                # switch to EMA parameters
                dae_optimizer.swap_parameters_with_ema(store_params_in_ema=True)

            vae_optimizer.zero_grad()

            # apply vae:
            with torch.set_grad_enabled(True):
                logits, all_log_q, all_eps = vae(x)
                eps = vae.concat_eps_per_scale(all_eps)[0]  # prior is applied at the top scale
                remaining_neg_log_p_total, remaining_neg_log_p_per_ver = \
                    cross_entropy_normal(all_eps[vae.num_groups_per_scale:])
                output = vae.decoder_output(logits)
                vae_recon_loss = reconstruction_loss(output, x, crop=vae.crop_output)

                ##############################################
                ###### Update the VAE encoder/decoder ########
                ##############################################

                noise_q = torch.randn(size=eps.size(), device='cuda')

                # apply diffusion model for samples generated for q (vae)
                t_q, var_t_q, m_t_q, obj_weight_t_q, _, g2_t_q = \
                    diffusion_cont.iw_quantities(batch_size, time_eps, iw_sample_q, iw_subvp_like_vp_sde)
                eps_t_q = diffusion_cont.sample_q(eps, noise_q, var_t_q, m_t_q)

                # run the score model
                mixing_component = diffusion_cont.mixing_component(eps_t_q, var_t_q, t_q, enabled=dae.mixed_prediction)
                pred_params_q = dae(eps_t_q, t_q)
                params = get_mixed_prediction(dae.mixed_prediction, pred_params_q, dae.mixing_logit,
                                              mixing_component)
                l2_term_q = torch.square(params - noise_q)
                cross_entropy_per_var = obj_weight_t_q * l2_term_q
                cross_entropy_per_var += diffusion_cont.cross_entropy_const(time_eps)
                all_neg_log_p = vae.decompose_eps(cross_entropy_per_var)
                all_neg_log_p.extend(remaining_neg_log_p_per_ver)  # add the remaining neg_log_p
                kl_all_list, kl_vals_per_group, kl_diag_list = kl_per_group_vada(all_log_q, all_neg_log_p)

                # kl coefficient
                if cont_kl_anneal:
                    kl_coeff_ = kl_coeff(step=timestamp,
                                         total_step=kl_anneal_portion_vada * num_total_iter,
                                         constant_step=kl_const_portion_vada * num_total_iter,
                                         min_kl_coeff=kl_const_coeff_vada,
                                         max_kl_coeff=kl_max_coeff_vada)
                else:
                    kl_coeff_ = 1.0

                # nelbo loss with kl balancing
                balanced_kl, kl_coeffs, kl_vals = kl_balancer(kl_all_list, kl_coeff_, kl_balance=kl_balance_vada,
                                                              alpha_i=alpha_i)
                nelbo_loss = balanced_kl + vae_recon_loss

                # compute regularization terms
                regularization_q, vae_norm_loss, vae_bn_loss, vae_wdn_coeff = vae_regularization(
                    True, weight_decay_norm_vae, vae_sn_calculator)
                q_loss = torch.mean(nelbo_loss) + regularization_q  # vae loss

            # backpropagate q_loss for vae and update vae params, if trained
            q_loss.backward()
            if grad_clip_max_norm > 0.:  # apply gradient clipping
                torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=grad_clip_max_norm)
            vae_optimizer.step()

            if update_q_ema and timestamp > 0:
                # switch back to original parameters
                dae_optimizer.swap_parameters_with_ema(store_params_in_ema=True)

            ####################################
            ######  Update the SGM prior #######
            ####################################

            # the interface between VAE and DAE is eps.
            eps = eps.detach()

            dae_optimizer.zero_grad()

            noise_p = torch.randn(size=eps.size(), device='cuda')
            # get diffusion quantities for p sampling scheme (sgm prior)
            t_p, var_t_p, m_t_p, obj_weight_t_p, _, g2_t_p = \
                diffusion_cont.iw_quantities(batch_size, time_eps, iw_sample_p, iw_subvp_like_vp_sde)
            eps_t_p = diffusion_cont.sample_q(eps, noise_p, var_t_p, m_t_p)
            # run the score model
            mixing_component = diffusion_cont.mixing_component(eps_t_p, var_t_p, t_p, enabled=dae.mixed_prediction)
            pred_params_p = dae(eps_t_p, t_p)
            params = get_mixed_prediction(dae.mixed_prediction, pred_params_p, dae.mixing_logit, mixing_component)
            l2_term_p = torch.square(params - noise_p)
            p_objective = torch.sum(obj_weight_t_p * l2_term_p, dim=[1, 2, 3])

            regularization_p, dae_norm_loss, dae_bn_loss, dae_wdn_coeff = dae_regularization(
                weight_decay_norm_dae, dae_sn_calculator)

            p_loss = torch.mean(p_objective) + regularization_p

            p_loss.backward()
            # update dae parameters
            if grad_clip_max_norm > 0.:  # apply gradient clipping
                torch.nn.utils.clip_grad_norm_(dae.parameters(), max_norm=grad_clip_max_norm)
            dae_optimizer.step()

            ########################
            ######  Logging  #######
            ########################

            writer.add_scalar('LSGM/p_loss', p_loss, timestamp * batch_size)
            writer.add_scalar('LSGM/q_loss', q_loss, timestamp * batch_size)
            if (timestamp + 1) % samples_per_log == 0:
                dae.eval()
                vae.eval()
                # switch to EMA parameters
                dae_optimizer.swap_parameters_with_ema(store_params_in_ema=True)

                # generate samples
                samples_disc, _, _, _ = generate_samples_vada(dae, diffusion_disc, vae, int(min(9, batch_size)),
                                                              prior_var=1.0)

                img_disc_x = samples_disc.detach()
                img_disc_y = torch.clamp(img_disc_x[:, :3] * 255, min=0, max=255
                                         ).round().to(torch.uint8).permute(0, 2, 3, 4, 1)
                img_disc_x = img_disc_x[:, 3] > 0.5
                show_voxels(voxel_grids=img_disc_x, color_grids=img_disc_y, save_as_file_on=writer,
                            writer_params=('LSGM/generated_disc', timestamp * batch_size), image_size=256,
                            packed=True)

                samples_ode, nfe, _, _ = generate_samples_vada(dae, diffusion_cont, vae, int(min(9, batch_size)),
                                                               ode_eps=fast_ode_param['ode_eps'],
                                                               ode_solver_tol=fast_ode_param['ode_solver_tol'],
                                                               ode_sample=True, prior_var=1.0)
                img_ode_x = samples_ode.detach()
                img_ode_y = torch.clamp(img_ode_x[:, :3] * 255, min=0, max=255
                                        ).round().to(torch.uint8).permute(0, 2, 3, 4, 1)
                img_ode_x = img_ode_x[:, 3] > 0.5
                show_voxels(voxel_grids=img_ode_x, color_grids=img_ode_y, save_as_file_on=writer,
                            writer_params=('LSGM/generated_ode', timestamp * batch_size), image_size=256,
                            packed=True)
                writer.add_scalar('LSGM/ode_sampling_nfe_single_batch', nfe, timestamp * batch_size)

                # switch back to original parameters
                dae_optimizer.swap_parameters_with_ema(store_params_in_ema=True)

            timestamp += 1

    return {'vae': vae.cpu(), 'dae': dae.cpu(), 'diffusion_disc': diffusion_disc}
